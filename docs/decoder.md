# How the decoder works

deFLEX turns a recording of raw radio samples into validated paging text, in
pure numpy/scipy. The decode core (`core/flex_core.py` for FLEX,
`core/pocsag_core.py` for POCSAG) is run two ways: as a **batch decoder** over a
recorded `.cfile` (the `flex_batch.py` / `pocsag_batch.py` CLIs), and as a **live
streaming decoder** inside the SDR receivers (see [`receiver.md`](receiver.md)).
Both run the *same* decode functions, so the live path is bit-for-bit equivalent
to the batch baseline.

This document describes the FLEX path — the harder of the two protocols. POCSAG
works the same way on a simpler 2-FSK front end and shares the FEC and readability
core; it is summarized at the end.

---

## The signal

FLEX is a one-way paging protocol (TIA-1500). A carrier announces its mode in a
sync word at the start of every frame; the decoder learns the baud/level from the
signal rather than being told.

| Parameter | Value | Notes |
|---|---|---|
| Frame period | 1.875 s = **30000 samples @ 16 kHz** | strictly periodic; 128 frames = 4-min cycle |
| Symbol rate | 1600 or 3200 baud | mode-dependent; ~5 samples/symbol at 16 kHz for 3200 baud |
| Modulation | 2- or 4-FSK, tones at **±4800 / ±1600 Hz** | for 4-level, tone spacing (3200 Hz) = baud rate, so tones are *orthogonal* over a symbol |
| Frame sync | 32-bit marker `0xA6C6AAAA` + mode codeword | packed into a 64-bit sync value |
| Frame data | 11 blocks × 8 words = **88 words** | each word BCH(31,21) + 1 parity bit |
| FEC | BCH(31,21,5) | corrects 1 error, detects 2 per word |

Pages come in types: ALN/SPN (**alpha** — text), NUM/NNM (**numeric** — digits),
and TON (tone-only). deFLEX is optimized for **alpha**, which can be validated
for readability; numeric pages cannot, so they are dropped.

---

## The decode pipeline

```
complex64 IQ (250 kS/s)
  → [optional] de-rotate carrier offset      (select a neighbour channel within the capture)
  → firwin channel LPF (default 12 kHz)
  → 4-FSK noncoherent matched-filter bank     (correlate baseband against the 4 FLEX tones/symbol)
  → per-frame carrier-offset null             (re-centre the reference tones)
  → max-log soft bit metrics
  → frame acquisition                         (matched-filter sync correlator)
  → symbol-timing + parity sweep
  → 8×32 deinterleave
  → BCH(31,21) decode + Chase-II soft FEC
  → frame → pages parse                       (BIW / capcodes / alpha / numeric)
  → per-page confidence tiering (A/B/C/D)
  → English-likeness readability gate (alpha)
```

---

## Front end: 4-FSK noncoherent matched-filter detection

The detector correlates the raw complex baseband against the four FLEX tones over
each symbol and picks the strongest (`mf_bank_mag`). This is the **spec-optimal**
detector for FLEX's 4-level modes: because the mode-3 tone spacing equals the baud
rate, the four tones are exactly orthogonal over a symbol — precisely the regime a
noncoherent matched-filter bank is built for. It substantially out-performs a
plain FM discriminator, especially on weak carriers.

The detector also chooses its own symbol sampling phase via a short timing sweep,
rather than inheriting one from a separate demodulator — the sampling instant that
maximizes matched-filter margin is the one used to decode.

**Per-frame carrier-offset null.** A small residual tuning offset remains after
channel select; `est_cfo` estimates it per frame and shifts the reference tones to
cancel it, recovering pages that would otherwise fail BCH.

(A coherent phase-tracking detector was tried and abandoned: FLEX's integer-h
CPFSK is the regime where coherent's edge over noncoherent is smallest and phase
tracking under fading is hardest, which is exactly why real pagers use
noncoherent FSK.)

---

## Acquisition: matched-filter sync + periodic-grid comb

Frame acquisition uses a **normalized matched-filter correlator** against the
known 64-bit sync template (`corr_acquire`): a normalized cross-correlation,
peak-picked per mode. This integrates the whole sync word, so it survives many bit
errors and locks far more frames on weak carriers than a hard bit-threshold sync.

Because the frame grid is **exactly periodic** on the transmitter's clock, a few
high-confidence sync peaks fix the global phase of the entire capture. deFLEX
then **combs every periodic grid slot**, decoding frames that are too faded to
detect on their own but whose position is known from the grid.

---

## Forward error correction: BCH(31,21) + Chase

FLEX interleaves 8 codewords across each 256-bit block so a noise burst damages
one (correctable) bit in each word instead of destroying one whole word. deFLEX
undoes the interleave, then decodes each 31-bit codeword with **BCH(31,21)** (the
hard-won FEC + Chase soft-decoder lives in `core/paging_core.py`, shared with
POCSAG). Words that fail hard decode get a second pass with **Chase-II
soft-decision** decoding: it flips the least-reliable bits and keeps the
minimum-distance valid codeword. On alpha text this recovers genuinely cleaner
messages.

---

## Confidence tiers (A/B/C/D)

Every page is graded, never hard-dropped, by fusing two independent signals: the
per-word **BCH status** (clean / corrected / chase / failed) and a
**FEC-independent soft margin** (the normalized mean matched-filter decision
margin).

- **A** — all words verified clean.
- **B** — FEC-corrected but healthy (good margin).
- **C** — low margin or low printable ratio.
- **D** — contains an uncorrectable word.

The **A+B** set is the trustworthy output: each such page is BCH/Chase-validated
and reads as real text.

---

## Alpha optimization + readability gate

Numeric pages cannot be validated — NUM bodies have no structure beyond BCH, and
aggressive Chase will manufacture valid-looking digit-soup. deFLEX therefore
discards numeric pages and gates alpha bodies through an `english_score(body)`
function designed to **fail on numeric/garbage by construction**:

- base score = (letters + spaces) / length
- small **capped** credit for digits and limited punctuation, so dates and codes
  survive but a punctuation-soup like `&"Q$2i{Cf` cannot
- heavy penalty for control characters
- penalty for long consonant runs (random text piles 6–10 consonants; English
  rarely exceeds 4)
- × a vowel-band factor (English vowel ratio ≈ 0.38; accept 0.18–0.55 so terse
  codes like "NOC"/"PCB" still pass)

A trustworthy-tier page scoring below the floor (0.60) is demoted to C. This makes
the aggressive full-grid comb **safe**: Chase-forged noise is rejected, real text
is kept, and genuinely-English-but-garbled messages are promoted to trustworthy.

---

## Comparison with `multimon-ng`

On a matched benchmark — the same recorded capture, the same readability gate
(english-score ≥ 0.60), deduplicated on both sides — deFLEX and multimon achieve
**comparable real-message recall**, but deFLEX's output is **categorically
cleaner**: every body it emits is BCH/Chase-validated and garbage-free by
construction, while multimon prints whatever it sliced, errors and all.

| Carrier | Signal | deFLEX (A/B, FEC-clean) | multimon (same gate) |
|---|---|---|---|
| strong, on-center | −16 dB | 24 | 28 |
| weak, off-center | −27.5 dB | **3** | 2 |
| silent | — | 0 | 0 |

multimon's higher count on the strong carrier is inflated by base64/encrypted
blobs and several error-different copies of the same page; deFLEX emits each
message once, clean, or not at all. On the **weak carrier** deFLEX wins outright.

The same page from both decoders (illustrative; identifiers are placeholders):

```
deFLEX : From: alarms@example.net Subject: Alarm Notification - At
          08:29:10, alarm "Alarm" at "default/SiteA/Unit/Controller/Right/
          Alarms/Status is in Alarm" transitioned to Active.

multimon: oO:$alaros@exaople.nef Swbjgat: AlarM ^gthfisAtyol - At 08:29:10,
          alarm$"Alarm" at "default/SiteA/Unit/Condroloer/Right/Alarms/
          [tctuc ic in ELape" transktionmd }o Active.
```

A weak-carrier sensor broadcast that multimon dissolves into noise (placeholders):

```
deFLEX : From: sensors@example.net - LAB ROOM TEMP 11:30:00 PM 19.8 20.0 23.9
multimon: 30:30z00 PM [62]Yof@cVfB]?3fYL}:3x}?~9ma?x{k...   (dissolves into noise)
```

Across a wideband capture of multiple FLEX carriers, **every page deFLEX marks
trustworthy is genuinely clean**, the trustworthy count closely tracks the count
of independently-readable bodies (so the tiering throws away no real text), and
both FLEX baud rates decode (the mode is taken from the self-identifying sync
codeword and the 4-level symbols are de-multiplexed differently per baud).

Where multimon recovers a few extra pages on a strong carrier, it does so by
running continuous bit/frame tracking across the whole transmit cycle. The live
streaming receiver closes most of that gap by decoding continuously over an
unbounded stream rather than a fixed window (see [`receiver.md`](receiver.md)).

---

## POCSAG

POCSAG decoding (`core/pocsag_core.py`, `pocsag_batch.py`) is simpler: a 2-FSK
front end (FM-demod → best-of-8 sampling phase chosen by sync correlation), a
fixed frame structure locked to the 32-bit frame-sync codeword, the **same**
BCH(31,21) + Chase FEC from `paging_core`, and the **same** `english_score`
readability gate. There are no per-frame mode/timing sweeps or confidence tiers to
configure — a page either passes the readability gate or it doesn't.

---

## Usage

The full decode chain — 4-FSK matched-filter bank, correlator acquisition with
grid comb, symbol-timing + parity sweep, and Chase-II soft FEC — is always on.
By default only alpha (ALN/SPN) pages are kept and graded against the
`english_score` gate; `--all` emits every page type without that gate.

```bash
# FLEX, default alpha-optimized decode (raw complex64 .cfile @ 250 kS/s):
python3 flex_batch.py capture.cfile

# Decode a neighbouring channel by frequency offset within the same capture:
python3 flex_batch.py capture.cfile --carrier=50000
```

### FLEX flags

| Flag | Effect |
|---|---|
| `--all` | Emit every page type (NUM/NNM too) and skip the `english_score` gate. |
| `--carrier=HZ` | De-rotate to a channel at this offset before decoding. |
| `--lpf=HZ` | Channel low-pass cutoff (default 12000). |
| `--samp-rate=HZ` | Capture sample rate (default 250000). |
| `--center=MHZ` | Display-only capture center, for labelling carriers. |
| `--corr-peaks` | Acquire only at detected peaks (no grid comb). |
| `--corr-off=N` | Peak→frame-start offset (default 1517). |
| `--frac` | Fractional (sub-sample) timing sweep. |
| `--inv` | Invert the tone→level polarity map. |
| `--diag`, `--pf` | Histogram / per-frame diagnostics. |

Output: per-frame stats, the A/B/C/D confidence summary, a by-type/tier breakdown,
and sample message bodies tagged `[tier margin printable% english]`.

### POCSAG

```bash
python3 pocsag_batch.py capture.cfile [--in-rate HZ] [--min-en F]
# e.g. print only readable pages (english_score >= 0.5):
python3 pocsag_batch.py capture.cfile --min-en 0.5
```

`--in-rate` is the capture sample rate (default 250000; resampled to 9600
internally); `--min-en` is an `english_score` floor for printing a page (0 = print
all). Output: one line per decoded alpha page —
`cap=<RIC> f=<func> en=<score>  <body>`.

---

## Provenance

Built from the TIA-1500 protocol tables as implemented in the `gr-pager` GNU Radio
module (used only as a protocol reference, not linked). The BCH(31,21) FEC + Chase
soft-decoder and the `english_score` gate are protocol-neutral and shared by the
FLEX and POCSAG decoders in `core/paging_core.py`. The FLEX deinterleave/BCH hot
loop has an optional numba (`@njit`) accelerator in `core/flex_numba.py` that is
bit-exact against the pure-Python path.
