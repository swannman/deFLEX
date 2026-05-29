# flexdec — a from-scratch FLEX paging decoder

A single-file numpy/scipy decoder for the FLEX paging protocol (TIA-1500),
built from the spec to see whether a modern DSP front end could match or beat
`multimon-ng` on real off-air traffic. It is a **research / A-B tool**, not a
live receiver (see [Status](#status)).

The headline result: on a frozen 120 s benchmark capture, flexdec produces
**FEC-validated, garbage-free** alpha decodes that are cleaner than multimon's,
and it **decisively wins on weak carriers** — recovering a lab-temperature
broadcast (`TempTrak`) at −27.5 dB that multimon renders as noise.

---

## The signal

FLEX is a one-way paging protocol. The benchmark carrier (929.6125 MHz, the
Spok network in the Seattle area) is **mode 3: 3200 baud, 4-level FSK**,
**inverted polarity**.

| Parameter | Value | Notes |
|---|---|---|
| Frame period | 1.875 s = **30000 samples @ 16 kHz** | strictly periodic; 128 frames = 4-min cycle |
| Symbol rate | 3200 baud | → only **5 samples/symbol** at 16 kHz — the crux of every tuning decision |
| Modulation | 4-FSK, tones at **±4800 / ±1600 Hz** | mode-3 tone spacing (3200 Hz) = baud rate → tones are *orthogonal* over a symbol |
| Frame sync | 32-bit marker `0xA6C6AAAA` + mode codeword | packed into a 64-bit sync value |
| Frame data | 11 blocks × 8 words = **88 words** | each word BCH(31,21) + 1 parity |
| FEC | BCH(31,21,5) | corrects 1 error, detects 2 per word |

Pages come in types: ALN/SPN (**alpha** — text), NUM/NNM (**numeric** — digits),
TON (tone-only). This decoder is optimized for **alpha**.

---

## Pipeline

```
complex64 IQ (250 kS/s)
  → [optional] de-rotate carrier offset      (--carrier=HZ : select a neighbour channel)
  → firwin channel LPF (default 12 kHz)
  → 4-FSK noncoherent matched-filter bank     (mf_bank_mag: correlate raw baseband
                                               against the 4 FLEX tones per symbol)
  → per-frame carrier-offset null             (est_cfo: shift reference tones)
  → max-log soft bit metrics                  (mfbank_softbits)
  → frame acquisition                         (correlator OR hard sync state machine)
  → symbol-timing + parity sweep              (--sweep, --frac)
  → 8×32 deinterleave
  → BCH(31,21) decode + Chase-II soft FEC      (--soft)
  → frame → pages parse (BIW / capcodes / alpha / numeric)
  → per-page confidence tiering (A/B/C/D)
  → [--alpha] English-likeness quality gate
```

---

## What was learned (chronological — the parts that mattered)

### 1. The detector: noncoherent matched-filter bank beats FM discrimination
The first front end was an FM discriminator (`angle(x[1:]·conj(x[:-1]))`). It
worked but capped at ~2 clean ALN. Replacing it with a **4-FSK noncoherent
matched-filter bank** — correlating the raw complex baseband against the four
FLEX tones over each symbol — was the single biggest detection win
(clean ALN 2→7). This is the **spec-optimal** detector: because mode-3 tone
spacing equals the baud rate, the tones are exactly orthogonal over a symbol,
which is precisely the regime a noncoherent MF bank is built for.

**Key lesson: the detector that decodes must also drive the timing sweep.**
Feeding the MF bank the FM discriminator's "optimal" timing made it *worse*;
letting the MF bank pick its own sampling phase via the sweep is what unlocked
the gain.

### 2. Coherent phase tracking is a dead end here
Tried decision-directed, squaring-loop, and squaring+unwrap coherent detectors
— **all regress ~4×**. FLEX mode-3 is integer-h CPFSK (phase advances by odd
multiples of π per symbol), the regime where coherent's edge over noncoherent is
smallest (~1 dB) *and* phase tracking is hardest (π-ambiguity traps flip bits
globally on one bad decision). This is exactly why real pagers use noncoherent
FSK. Kept as a documented opt-in (`--coh`), not used.

### 3. Carrier-offset null (per-frame CFO) is worth it
`est_cfo()` estimates residual carrier per frame and shifts the reference tones
to null it: clean ALN 4→7, fewer BCH fails. There's a real residual offset worth
correcting frame by frame.

### 4. Acquisition: a data-aided correlator, not a hard threshold (`--corr`)
The original sync was a hard, zero-sliced bit correlator that accepted a frame
only when the 32-bit marker matched within 3 bit errors. On a strong carrier it
locked ~33 of ~64 frames; on a weak carrier only 3. Replaced with a
**normalized matched-filter correlator** on the known 64-bit sync template
(`corr_acquire`): NCC = `fftconvolve(d, template[::-1])` / sliding-RMS, peak-pick
per mode. The frame grid is exactly periodic, so a few high-confidence peaks fix
the global phase and we **comb every grid slot** (`--comb`).

Validated before wiring in: sync peaks sit at a rock-solid position
(≡ 1073 mod 30000, std 0.5 samples) — the shared transmitter frame clock — and
the peak→frame-start offset is a constant 1517 samples on both carriers.

### 5. The 88-vs-80-word bug (root cause of "0 pages on the weak carrier")
For a long time the +50k carrier emitted **zero** pages and I wrongly concluded
"content absence." It was a **parse bug**: every frame decoded to 80 words, not
88, so `parse_frame` bailed on `len != 88`. Cause: the parity sweep
(`n = nsyms - par`) dropped a symbol under parity=1, truncating 11 blocks (88
words) to 10 (80). Fewer words meant fewer BCH fails, so the broken alignment
looked "best." **Fix: 2-symbol slack — `nsyms = baud*1760//1000 + 2`.** This
unblocked the weak carrier entirely *and* quietly lifted the strong one.

> Lesson worth keeping: "no output" is a hypothesis, not a conclusion. The
> ground truth (multimon *does* decode it) was right; my decoder had a bug.

### 6. Confidence tiers instead of a binary clean/garbled flag
Two independent signals fused per page: per-word BCH status (clean / corrected /
chase / failed) and a FEC-independent **soft margin** (mean MF-bank decision
margin, normalized). Tiers: **A** all words verified, **B** FEC-corrected but
healthy, **C** low margin or low printable-ratio, **D** any uncorrectable word.

### 7. Soft-decision Chase FEC helps text, forges numbers
Chase-II (`--soft`) flips the L=4 least-reliable bits on every failed word and
keeps the min-distance valid codeword. On **alpha** it produces genuinely
cleaner text. But on **numeric** it *manufactures* valid-looking digit-soup —
NUM bodies have no structure beyond BCH and every BCD char is "printable," so
aggressive Chase passes any printable gate with garbage. This motivated the
alpha-only mode.

### 8. Alpha-only optimization + English-likeness gate (`--alpha`)
The insight (Matt's): *numeric pages can't be validated, but alpha can.* So:
discard NUM/NNM, then gate alpha bodies on an **`english_score(body)`** that
*fails on numeric by design*:

- core = (letters + spaces) / n
- small **capped** credit for digits + limited punctuation (dates/codes survive;
  a punct-soup like `&"Q$2i{Cf` cannot)
- heavy penalty for control characters
- penalty for long consonant runs (random text piles 6–10 consonants; English
  rarely exceeds 4)
- × a vowel-band score (English vowel ratio ~0.38; accept 0.18–0.55 so terse
  codes like "NOC"/"PCB" pass)

Floor `ALPHA_EN_OK = 0.60`: under `--alpha`, an A/B page scoring below that is
demoted to C. This makes the **aggressive full-grid comb safe** — Chase-forged
noise is rejected, real text is kept, and genuinely-English-but-garbled messages
(TempTrak) are promoted to trustworthy.

---

## A-B results vs multimon-ng

All on the same frozen capture `iq_929612500_250k.cfile` (929.6125 MHz center,
250 kS/s, 120 s). Only three FLEX carriers fall within the recording's ±125 kHz.
Both decoders measured with the **same** English gate (≥0.60) + dedup for
fairness.

| Carrier | Signal | flexdec `--alpha` (FEC-clean) | multimon (same gate) |
|---|---|---|---|
| −25k · 929.5875 | dead | **0** | 0 |
| center · 929.6125 | −16 dB | **24** | 28 |
| +50k · 929.6625 (NOC) | −27.5 dB | **3** | 2 |

**Verdict:** neck-and-neck on count (27 vs 30). flexdec trades a little center
recall for **higher fidelity** (its bodies are BCH/Chase-validated and
garbage-free by construction; multimon prints whatever it sliced, errors and
all) and a **clear win on the weak carrier**.

The starkest single example — same page, both decoders, at center:

```
flexdec : From: HGMS@onsemi.com Subject: Ignition Alarm Notification - At
          08:29:10, alarm "Alarm" at "default/HGMS/GCs/GC-52/Controller/Right/
          Alarms/Exhaust/Status is in Alarm" transitioned to Active.

multimon: oO:$LGMS@onseoa.com Swbjgat: I'nx4yon AlarM ^gthfisAtyol - At 08:29:10,
          alarm$"Alarm" at "default/HGMS/GCs/GC-5;/Condroloer/Right/Alarms/
          Exhaust/[tctuc ic in ELape" transktionmd }o Active.
```

And the weak-carrier win — `TempTrak` lab-temperature broadcast at +50k:

```
flexdec : From: TempTrak@luminishealth.ost - LAB ROOM TEMP 5/28/2026 11:30:00 PM
          19.8 20.0 23.9 5/28/2026 10:30:00 PM
multimon: 30:30z00 PM [62]Yof@cVfB]?3fYL}:3x}?~9ma?x{k...   (dissolves into noise)
```

multimon's center count is also inflated by base64/encrypted blobs and ~5
error-different copies of the same EZCall page; flexdec emits each message once,
clean, or not at all.

### Why multimon still leads on center recall
It runs **continuous** bit/frame tracking over the whole transmit cycle and
stays locked through weak/idle frames, so it catches a few more pages flexdec
leaves in tier C. The lever to close that gap is longer captures / better
acquisition, not better per-frame detection (which is already near-optimal).

---

## Phase 2: wideband, all 7 production carriers (240 s)

The 250 kS/s benchmark only contains one real carrier. To test all 7, a fresh
**240 s / 2.646 MS/s** capture was taken at the production center (930.7625 MHz),
then each carrier de-rotated to baseband via `--carrier` + `--samp-rate`
(`load_baseband` resamples the wideband input down to the internal 250 kS/s, so
the validated pipeline is untouched — `--samp-rate=250000` reproduces the
benchmark result bit-for-bit).

| Carrier (MHz) | Frames (of 127) | Trustworthy A+B | Note |
|---|---|---|---|
| 929.5875 | 0 | 0 | carried no decodable FLEX this window (lowest-power carrier) |
| **929.6125** | 127 | **131** | strongest carrier |
| 929.6625 | 127 | 2 | weak SNR off-center |
| 929.9375 | 127 | 7 | |
| 931.0625 | 0 | 0 | **POCSAG channel, not FLEX** — correctly skipped |
| 931.2125 | 127 | 4 | |
| 931.9375 | 127 | 19 | |

**Findings:**

- **Every page flexdec marks trustworthy is genuinely clean** — real
  hospital/enterprise alpha across all five FLEX carriers (Harborview "Rapid
  Response room 462", "ACTUAL EVENT … Full Trauma … ER 8", Epic EVS logistics,
  UW "ready for turnover", door/alarm notifications).
- **Calibration is near-perfect:** the count of readable bodies
  (`--dump-readable`, en≥0.60 & ≥85% printable, any tier) ≈ the trustworthy A+B
  count on every carrier → the large `C_suspect` bulk is genuinely garbled, not
  hidden good text the tiering is throwing away.
- **The two "0-frame" carriers are not flexdec failures:** 931.0625 is a POCSAG
  channel (flexdec doesn't do POCSAG); 929.5875 simply had no decodable FLEX
  traffic in the 4-minute window (a tighter LPF didn't recover it).
- **Both FLEX speeds are live, and flexdec decodes both.** Confirmed from the
  self-identifying sync codeword (independently corroborated by multimon's rate
  tag), not inferred from output: 929.6125 / 929.6625 / 931.2125 are **3200 baud
  4-level (6400 bps)**; 929.9375 / 931.9375 are **1600 baud 4-level (3200 bps)**.
  flexdec picks the mode per frame via sync-template correlation (`corr_acquire`)
  and demuxes 4-level symbols differently per baud (1600: A/B; 3200: A/B/C/D
  interleaved), so clean BCH-validated English came off the half-rate carriers
  too (the trauma alert on 929.9375, the shift-break notice on 931.9375).

**Fair A/B vs channelized multimon** (same English gate on both sides, plus
`difflib` fuzzy clustering at ratio ≥0.80 to collapse garbled near-duplicates):
raw distinct-message counts come out fd 96 / mm 126, but the multimon number is
**inflated by its lack of FEC** — e.g. on 929.9375 its "18 distinct" is really
~7 real messages + ~9 garbled copies of a single trauma alert (so corrupted they
don't even cluster with each other) + 2 noise lines. flexdec's are clean and
deduplicated.

**Verdict:** comparable real-message recall, materially cleaner output. Yield
tracks per-carrier SNR (2–131 trustworthy), but fidelity is uniformly excellent
wherever signal exists — which is what matters for downstream
machine-classification of bodies.

### Would re-admitting `C_suspect` buy more recall? (probe, 2026-05-29)

The open question after Phase 2 was whether the B-only default leaves real
messages on the table in the `C_suspect` tier on weak carriers. Probed it
directly: dumped **every** C-tier alpha page (no readability floor) on all 5
active carriers, clustered (ratio ≥0.80), and compared against the admitted B
set (method: dump every C page with its en/pr, fuzzy-cluster, subtract B).

| carrier | C pages | new clusters vs B | genuinely readable new msgs |
|---|---|---|---|
| 929.6125 | 302 | 212 | 1 (a garbled-header NWS forecast) |
| 929.6625 | 453 | 319 | 0 |
| 929.9375 | 223 | 161 | 0 |
| 931.2125 | 440 | 329 | 0 |
| 931.9375 | 237 | 170 | 0 |

Re-admitting C means accepting **~1,190 new clusters to recover ~1 readable
message.** There *is* real signal buried in C on the weak carriers — fragments of
a trauma alert and a shift-break notice were visible — but they arrive **below the
trust floor** (en 0.5–0.59), indistinguishable from pure garble and encrypted
base64 tokens at the same scores. No threshold pulls them in cleanly.

**Conclusion:** keep the B-only default. The right lever for weak-carrier recall
is better *acquisition* (more capture time / per-carrier gain) so those fragments
arrive at B-tier quality — not relaxing the tier gate.

---

## Usage

```bash
# Recommended alpha-optimized invocation (what the A-B above used):
python3 flexdec.py iq_929612500_250k.cfile --corr --comb --sweep --soft --alpha

# Decode a neighbouring channel by frequency offset:
python3 flexdec.py iq_929612500_250k.cfile --carrier=50000 --corr --comb --sweep --soft --alpha
```

Input is a raw **complex64** (`.cfile`) IQ recording at 250 kS/s.

### Flags

| Flag | Effect |
|---|---|
| `--alpha` | Keep only ALN/SPN pages; apply the `english_score` quality gate. |
| `--corr` | Data-aided correlator acquisition (grid comb). |
| `--comb` | Decode every periodic grid slot, not just detected peaks. |
| `--corr-peaks` | Correlator, but only at detected peaks (no comb). |
| `--corr-off=N` | Peak→frame-start offset (default 1517). |
| `--sweep` | Symbol-timing + parity search per frame. |
| `--frac` | Fractional (sub-sample) timing sweep. |
| `--soft` | Chase-II soft-decision FEC on failed words. |
| `--carrier=HZ` | De-rotate to a channel at this offset before decoding. |
| `--lpf=HZ` | Channel low-pass cutoff (default 12000). |
| `--coh` | Coherent detector (documented dead end; regresses). |
| `--diag`, `--pf` | Histogram / per-frame diagnostics. |

Output: per-frame stats, A/B/C/D confidence summary, by-type/tier breakdown, and
sample message bodies tagged `[tier margin printable% english]`.

---

## Status

**This is an offline research tool, not the live decoder.** The production
7-channel receiver (`flex-receiver.service` on p340) runs `multimon-ng` and is
unaffected by anything here. The core `flexdec.py` is **batch** (reads a whole
`.cfile` start-to-finish) and has been validated on **one carrier / 120 s**,
covering **both live FLEX speeds** (3200 baud/4-level = 6400 bps and 1600
baud/4-level = 3200 bps — see the dual-speed note above). A **streaming wrapper**
(`flexdec_stream.py`, below) now reproduces the batch A/B output exactly, but it
has not yet been wired to a live SDR tap.

A realistic path is a **hybrid**: keep multimon as the live workhorse, run
flexdec as a second-pass refiner on the weak/sparse carriers (929.6625) where it
clearly wins, or as an offline re-decode of logged IQ.

---

## Streaming (`flexdec_stream.py`)

A live decoder built **on top of** the frozen batch core — `flexdec.py` is
imported unchanged (md5 verified), so the streaming path can never drift from
the validated A/B baseline. The only added code is the *scaffolding*: a
complex-baseband ring buffer, an overlapped-window scheduler, and per-frame-slot
dedup. The actual symbol decode replays `flexdec.main()`'s per-frame loop
verbatim (matched-filter bank + per-frame CFO null + Chase-II soft FEC + the
`--alpha` English-likeness tier gate).

**Why overlapped-batch + dedup, not a stateful streaming demod.** FLEX is
strictly periodic (`FRAME_PERIOD = 30000` @ 16 kHz = 1.875 s) on a shared
transmitter clock, so a frame is self-contained. Re-running the validated batch
decode on a sliding window and de-duplicating by *absolute frame slot* is
provably equivalent to batch — there is no resampler state to drift. The window
overlaps by ≥ one frame so every frame is fully inside at least one window;
dedup on `(slot, type, body)` collapses the repeats.

- **Tap point:** the **complex baseband** after channel-select (post freq-xlate,
  pre channel-LPF) — the same samples the matched-filter bank needs. Tapping the
  real FM-discriminator output instead would forfeit the MF-bank quality win
  (A+B = 34 vs 24 on the benchmark).
- **Window / advance:** default **16 frames (30 s) / 8 frames (15 s)**.
  `corr_frames`' `grid_phase` locks the frame grid from the sync anchors *in the
  window*, so a window needs ~16 anchors for its grid to match the whole-capture
  grid to sub-symbol precision. 8/4 dropped one marginal 3-char SPN page; 16/8
  (and larger) reproduce the batch A/B set exactly.

**Validation** (`python3 flexdec_stream.py [cfile]`): replays the frozen 250 k
benchmark in 2 s chunks through `StreamDecoder.feed()` and asserts the streamed
trustworthy-A/B set is a superset of a whole-file batch decode of the same data.
Result on `iq_929612500_250k.cfile`: **batch 32, streamed 32, 0 missing, 0
extra → PASS.** (The whole-file `decode_window` also reproduces `flexdec.py`'s
raw A+B = 34 page-instance count, confirming the lifted loop is byte-faithful;
32 is the deduped unique-message count.)

Library use:

```python
import flexdec_stream as S
sd = S.StreamDecoder(on_page=lambda r: print(r))   # r = (slot, type, body, tier, pr, en)
for chunk in source_of_complex64_baseband_at_250k():
    sd.feed(chunk)
sd.flush()
```

Not yet done: the live SDR tap (feed the GNU Radio `conjugate_cc` output per
carrier into `feed()` instead of replaying a file), and per-carrier resampling
if the tap rate ≠ 250 k.

---

## Provenance / notes

- Built from the TIA-1500 tables as implemented in the `gr-pager` GNU Radio
  module (used only as a protocol reference, not linked).
- Benchmark capture is frozen on purpose — re-capturing new IQ would invalidate
  the multimon A-B baseline.
- The `!T>* -R[5=%"#;~J` prefix on weather pages is a **real transmitted
  header**, not a decode artifact — it appears in multimon's output too.
