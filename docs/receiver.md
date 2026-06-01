# The live receiver

`start_receiver.py` is the live harness: it tunes **one** SDR, channelizes the
captured band, and runs the right decoder on each carrier — `FLEXStream` for FLEX,
`POCSAGStream` for POCSAG — writing each clean page to a log the web viewer
(`viewer/`) renders. It feeds the decoders directly from the SDR (no intermediate
audio files), and runs the *same* decode functions as the batch CLIs, so it never
drifts from the validated baseline.

A frequency listed under **both** `--flex` and `--pocsag` is decoded by both — the
only model that handles a channel carrying FLEX and POCSAG on the same frequency.

`flex_receiver.py` and `pocsag_receiver.py` are the **single-carrier** versions
(one carrier on their own SDR), plus a `--file` mode that replays a recorded
capture through the identical streaming decoder for an exact parity check against
the batch decoder.

All carriers must fit inside one SDR capture window (`center ± samp_rate/2`); the
caller picks frequencies that do. FLEX (~900 MHz) and POCSAG VHF (~150 MHz) are
different bands, so they need separate harness instances on separate SDRs.

```
SDR @ 2.5 MS/s (one GNU Radio flowgraph in the parent process)
  → N× freq_xlating_fir_filter (channel-select + decimate to 250 kS/s per carrier)
  → RingSink   (per-carrier; copies to one drop-oldest mp.Queue PER protocol)
  ══ process boundary (complex64 chunks cross the queue) ══
  → worker process  (one per (carrier, protocol), own GIL)
  → FLEXStream  /  POCSAGStream   (overlapped-batch replay of the decode core)
  → /var/log/flex/<carrier>.flexdec.log   (FLEX)   "FLEX|…" lines, ALN only
    /var/log/flex/<carrier>.pocsag.log    (POCSAG)
```

A dual-protocol carrier has two workers (one FLEX, one POCSAG) fed the same
channelized baseband, writing two log files.

---

## Streaming model: overlapped batch + dedup

The live decoders (`FLEXStream` in `core/flex_core.py`, `POCSAGStream` in
`core/pocsag_core.py`) do **not** rewrite the decoder as a stateful streaming
pipeline. Each re-runs the validated batch decode on a sliding window and
de-duplicates the results — provably equivalent to a batch decode because a frame
is self-contained and there is no resampler or PLL state that drifts between
chunks.

- Each window is re-decoded with the full batch pipeline (matched-filter bank +
  per-frame CFO null + Chase-II FEC + the readability tier gate).
- Windows overlap by at least one frame, so every frame is fully inside at least
  one window (never sliced at an edge).
- FLEX frames are tagged with their **absolute frame slot** (FLEX is strictly
  periodic, `FRAME_PERIOD = 30000` @ 16 kHz = 1.875 s); de-dup on `(slot, type,
  body)` collapses the overlap repeats. POCSAG has no global clock, so its stream
  de-dups on the page content directly.

```python
from flex_core import FLEXStream          # pure numpy/scipy, no GNU Radio
sd = FLEXStream(on_page=lambda r: print(r))   # r = (slot, type, body, tier, pr, en)
for chunk in source_of_complex64_baseband_at_250k():
    sd.feed(chunk)
sd.flush()
```

The per-channel baseband rate is **250 kS/s** for both protocols: it equals the
FLEX decoder's internal rate (no resample), and `POCSAGStream` resamples 250k →
9600 internally. So the harness channelizes every carrier identically and only the
worker differs by protocol.

---

## One OS process per (carrier, protocol)

The SDR feeds **one shared GNU Radio flowgraph** in the parent. Each carrier owns a
`freq_xlating_fir_filter` (channel-select + decimate to 250 kS/s) and a `RingSink`
that copies its baseband across a queue to each decode worker.

- **Why processes, not threads.** The decode's hot kernels (the matched-filter
  bank, the `@njit` BCH/deinterleave loop) release the GIL, but the Python glue
  does not — so threads would serialize and only the busiest carrier would keep up.
  One OS process per (carrier, protocol) gives each its own GIL, so decode
  parallelizes across cores for real.
- **Backpressure.** `RingSink` copies each block into a **bounded** `mp.Queue`
  with **drop-oldest on overflow**, so a slow decode can never back-pressure the
  shared flowgraph (and thus the SDR). The queue holds one overlapped-window decode
  burst's worth of input so it drains between bursts.
- Each worker coalesces any queue backlog into a single `feed()` so the
  overlapped-window scheduler always sees large contiguous blocks.

The plumbing (RingSink, channelizer, workers) lives in `core/receiver_sdr.py`
(GNU Radio) and `core/receiver_core.py` (the GR-free workers + log callbacks),
shared by the harness and both single-carrier receivers.

---

## Configuration

Carriers are given on the command line: `--flex` and `--pocsag` take comma-separated
MHz lists; `--center` defaults to the midpoint of all carriers; `--samp-rate`
defaults to 2.5 MS/s. The harness errors out if any carrier falls outside the
capture window. Key fixed parameters (in `core/receiver_core.py`):

| Parameter | Value | Why |
|---|---|---|
| Channel rate | **250 kS/s** | matches the FLEX decoder rate (no resample); POCSAG resamples to 9600 internally — one rate feeds both |
| Sample rate | **2.5 MS/s** (default) | `/10 → 250000`, one integer decimation factor |
| Window / advance | **32 / 28 frames** | ≥16 reproduces the batch result; 4-frame overlap margin |
| Pages emitted | **ALN only** | other page types are dropped before logging |
| Log line | `FLEX|<carrier>|…` | both protocols write the same shape so the viewer is unchanged; POCSAG puts its real RIC in the dedup field, FLEX a fixed `0` |
| Gain | sdrplay: manual RFGR/IFGR; else AGC | the 900 MHz FLEX setup is validated with manual gain |

**Channel-select filter.** The per-carrier filter uses a wide (60 kHz) transition
band on purpose, keeping the tap count low (~137). After decimate-by-10 the first
alias folds in around `250 kHz − 9 kHz ≈ 241 kHz`, leaving an enormous guard band;
a narrow transition would inflate the tap count by an order of magnitude and
overrun the SDR for no benefit.

---

## Running

```bash
# Harness: several FLEX carriers + a POCSAG carrier on one SDR (all MHz, all within
# one window). A frequency in both lists is decoded as FLEX and POCSAG:
python3 start_receiver.py --flex 930.5,931.5 --pocsag 931.5,931.8

# Preview the channelization plan without touching the SDR:
python3 start_receiver.py --flex 930.5,931.5 --pocsag 931.8 --dry-run

# Single carrier on its own SDR:
python3 flex_receiver.py   --live --freq 931.5
python3 pocsag_receiver.py --live --freq 152.0075 [--driver rtlsdr|sdrplay]

# Offline parity check — replay a capture through the streaming decoder and print
# the trustworthy A/B page count (compare against flex_batch / pocsag_batch):
python3 flex_receiver.py   --file capture.cfile --in-rate 250000
python3 pocsag_receiver.py --file capture.cfile --in-rate 250000
```

For a spectrally-mirrored capture, pass `--inv` (to `flex_receiver`,
`start_receiver`, or `flex_batch`) to flip FLEX tone polarity. POCSAG auto-detects
polarity, so it needs no such flag and ignores `--inv`.

The scripts put `core/` and their own directory on `sys.path`, so they run from a
repo checkout **and** from a flat install where everything is co-located. The
`--file` paths use only numpy/scipy; `--live` and the harness need GNU Radio.

For continuous operation, run `start_receiver.py --flex … --pocsag …` as a
long-lived process under your service manager. The per-(carrier, protocol) workers
are child **processes**, so stop them with a process-group kill (e.g. systemd
`KillMode=control-group`) to reap the whole group cleanly.

Logs land in `/var/log/flex/<carrier>.{flexdec,pocsag}.log` as
`<ts> FLEX|<carrier>|…|ALN|<body>` lines, which the web viewer (`viewer/`) tails
and streams to browsers.

---

## Performance characteristics

Each (carrier, protocol) decodes single-threaded within its own process. A carrier
with no sync short-circuits cheaply and stays well under realtime. A carrier
carrying heavy traffic runs the full matched-filter bank + Chase-II FEC every
window and sits near one core; under sustained heavy load it can tip just over
realtime during FEC-heavy windows, at which point the bounded queue drops the
oldest IQ at a slow steady rate — *partial* recall on the busiest carriers, never
a stall.

The lever to lift that ceiling is **intra-carrier parallelism**: decoding several
frames of a window in a thread pool inside the worker. Because the njit/numpy
kernels release the GIL, this can use more than one core per carrier. Shrinking the
decode window reduces per-window CPU but trades against weak-carrier recall.
