# deFLEX — a modern FLEX + POCSAG paging decoder

A from-scratch decoder for the **FLEX** (TIA-1500) and **POCSAG**
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
- **Live multi-carrier, multi-protocol reception.** A harness tunes one SDR,
  channelizes the captured band, and runs the right decoder per carrier — one OS
  process each. A single frequency can be decoded as **both** FLEX and POCSAG (for
  a channel that carries both). Feeds a web viewer.
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

The **executables** live at the root; the importable **libraries** live in
`core/`. Naming: `<proto>_core` is the decode library (and the live
`FLEXStream`/`POCSAGStream` wrapper), `<proto>_batch` is the offline CLI, and
`<proto>_receiver` is the single-carrier live receiver. `start_receiver.py` is the
multi-carrier, multi-protocol harness.

```
start_receiver.py   multi-channel receiver: one SDR -> channelize -> a decoder per carrier
                    (--flex / --pocsag lists; a shared freq gets both)
flex_receiver.py    single-carrier live FLEX receiver (+ --file replay)
pocsag_receiver.py  single-carrier live POCSAG receiver (+ --file replay)
flex_batch.py       FLEX batch decoder (CLI) over a recorded .cfile
pocsag_batch.py     POCSAG batch decoder (CLI)

core/               importable library modules
  paging_core.py      shared core: BCH(31,21) + Chase soft-decode + english_score gate
  flex_core.py        FLEX decoder + the live FLEXStream wrapper
  flex_numba.py       optional @njit kernels for the deinterleave/BCH hot loop
  pocsag_core.py      POCSAG decoder + the live POCSAGStream wrapper
  receiver_core.py    shared decode-worker layer + log callbacks (no GNU Radio)
  receiver_sdr.py     SoapySDR source + channelization (the one GNU-Radio module)
viewer/             web feed: tails the decode logs, streams to browsers
docs/               decoder.md (how decoding works), receiver.md (the live receiver)
```

The decode cores are pure numpy/scipy, so the decoders — including the streaming
wrappers (`FLEXStream`, `POCSAGStream`) and the offline `--file` paths — are usable
without GNU Radio. Only live SDR capture pulls it in: `core/receiver_sdr.py`, used
by `start_receiver.py` and the receivers' `--live` mode.

## Quickstart

```bash
# Decode a recorded FLEX capture (raw complex64 IQ @ 250 kS/s). The full decode
# chain is on by default and only alpha pages are kept; add --all for every type:
python3 flex_batch.py capture.cfile

# Decode a POCSAG capture, printing readable pages only:
python3 pocsag_batch.py capture.cfile --min-en 0.5

# Live: one SDR, several FLEX carriers + a POCSAG carrier (all in MHz, all within
# one capture window). A frequency in both lists is decoded as FLEX and POCSAG:
python3 start_receiver.py --flex 930.5,931.5 --pocsag 931.5,931.8
python3 start_receiver.py --flex 930.5,931.5 --pocsag 931.8 --dry-run   # preview, no SDR

# Or a single carrier on its own SDR:
python3 flex_receiver.py   --live --freq 931.2125
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
