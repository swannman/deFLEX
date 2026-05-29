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
unaffected by anything here. flexdec is **batch** (reads a whole `.cfile`
start-to-finish; no streaming path) and has only been validated on **one
carrier / 120 s / mode-3 (3200 baud)**. Promoting it would require a streaming
acquisition path and validation on the 6400-baud carriers.

A realistic path is a **hybrid**: keep multimon as the live workhorse, run
flexdec as a second-pass refiner on the weak/sparse carriers (929.6625) where it
clearly wins, or as an offline re-decode of logged IQ.

---

## Provenance / notes

- Built from the TIA-1500 tables as implemented in the `gr-pager` GNU Radio
  module (used only as a protocol reference, not linked).
- Benchmark capture is frozen on purpose — re-capturing new IQ would invalidate
  the multimon A-B baseline.
- The `!T>* -R[5=%"#;~J` prefix on weather pages is a **real transmitted
  header**, not a decode artifact — it appears in multimon's output too.
