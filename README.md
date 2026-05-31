# deFLEX — a modern FLEX + POCSAG paging decoder

A from-scratch numpy/scipy decoder for the **FLEX** (TIA-1500) and **POCSAG**
paging protocols, built around a modern DSP front end and optimized for
**alphanumeric** pages. On alpha traffic it **exceeds `multimon-ng`**: every page
it emits is BCH/Chase-validated and garbage-free, and it recovers weak carriers
that multimon renders as noise.

## What it does

- **FLEX and POCSAG.** FLEX at 1600 and 3200 baud, 2- and 4-level FSK, mode
  auto-detected per frame; POCSAG-1200 alpha.
- **Garbage-free alpha output.** A 4-FSK matched-filter detector feeds BCH(31,21)
  forward error correction with Chase soft-decision decoding, so a decoded body is
  either correct or not emitted — never the half-corrupted text multimon prints.
- **Per-page confidence tiers (A/B/C/D)** fused from BCH status and a
  FEC-independent soft margin, so a consumer can trust the A/B set outright.
- **Strong weak-carrier recovery.** A matched-filter sync correlator plus a
  periodic frame-grid comb pull clean pages off carriers well into the noise.
- **Live multi-carrier reception.** The same validated decoder runs in real time
  over an SDR — one OS process per carrier — and feeds a web viewer.
- **Pure numpy/scipy**, with an optional numba accelerator (bit-exact fallback).

## How it compares to `multimon-ng`

On matched benchmarks (identical readability gate, deduplicated), deFLEX's
real-message recall is comparable, but its **fidelity is categorically better**:
multimon prints whatever it sliced — bit errors, base64 blobs, and several
error-different copies of the same page — while deFLEX emits each message once,
clean, or not at all. On weak / off-center carriers deFLEX **clearly wins**,
recovering pages multimon dissolves into noise. The full comparison is in
[`docs/decoder.md`](docs/decoder.md).

deFLEX is **not** `multimon-ng`-derived: it was built from the TIA-1500 protocol
tables, using `gr-pager` only as a protocol reference (not linked).

## Layout

The four **executables** live at the root; the importable **libraries** live in
`core/`. Naming: `<proto>_core` is the decode library (and the live
`StreamDecoder`/`POCSAGStream` wrapper), `<proto>_batch` is the offline CLI, and
`<proto>_receiver` is the live SDR receiver.

```
flex_receiver.py    live FLEX receiver: SDR -> per-carrier channelize -> StreamDecoder
pocsag_receiver.py  live POCSAG receiver (single carrier)
flex_batch.py       FLEX batch decoder (CLI) over a recorded .cfile
pocsag_batch.py     POCSAG batch decoder (CLI)

core/               importable decode libraries (pure numpy/scipy, no GNU Radio)
  paging_core.py      shared core: BCH(31,21) + Chase soft-decode + english_score gate
  flex_core.py        FLEX decoder + the live StreamDecoder wrapper
  flex_numba.py       optional @njit kernels for the deinterleave/BCH hot loop
  pocsag_core.py      POCSAG decoder + the live POCSAGStream wrapper
viewer/             web feed: tails the decode logs, streams to browsers
docs/               decoder.md (how decoding works), receiver.md (the live receiver)
```

The cores are pure numpy/scipy (no GNU Radio), so the decoders — including the
streaming wrappers — are usable as a plain library; GNU Radio is only needed for
the live SDR receivers.

## Quickstart

```bash
# Decode a recorded FLEX capture (raw complex64 IQ @ 250 kS/s). The full decode
# chain is on by default and only alpha pages are kept; add --all for every type:
python3 flex_batch.py capture.cfile

# Decode a POCSAG capture, printing readable pages only:
python3 pocsag_batch.py capture.cfile --min-en 0.5

# Run the live FLEX receiver on an SDR (carrier frequencies in MHz):
python3 flex_receiver.py --live --carriers 929.6125,929.9375,931.2125

# Run the live POCSAG receiver on a single carrier:
python3 pocsag_receiver.py --live --freq 152.0075
```

## Documentation

- **[`docs/decoder.md`](docs/decoder.md)** — how the decoder works: the DSP
  pipeline, the 4-FSK detector, acquisition, FEC, confidence tiering, the alpha
  readability gate, and the `multimon-ng` comparison. Plus the batch CLI usage.
- **[`docs/receiver.md`](docs/receiver.md)** — the live SDR receiver: the
  overlapped-batch streaming model, the per-carrier multiprocessing design, the
  configuration parameters, and how to run it.
- **[`viewer/README.md`](viewer/README.md)** — the web viewer that renders the
  decoded feed.
