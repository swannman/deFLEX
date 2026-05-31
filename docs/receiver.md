# The live receiver

`flex_receiver.py` runs the validated FLEX decoder in real time over an SDR,
decoding many carriers concurrently and writing each clean page to a log that the
web viewer (`viewer/`) renders. It feeds the decoder directly from the SDR — no
intermediate audio files. `pocsag_receiver.py` is the POCSAG counterpart on its
own SDR.

The live path runs the *same* decode functions as the batch CLI, so it never
drifts from the validated baseline; a `--file` mode replays a recorded capture
through the identical ring → process → decoder path for an exact parity check.

```
SDR @ 2.5 MS/s (one shared GNU Radio flowgraph in the parent process)
  → N× freq_xlating_fir_filter (channel-select + decimate to 250 kS/s per carrier)
  → RingSink   (per-carrier mp.Queue, drop-oldest on overflow — never back-pressures the SDR)
  ══ process boundary (complex64 chunks cross the queue) ══
  → worker process  (one OS process per carrier, own GIL)
  → StreamDecoder   (overlapped-batch replay of the decode core + per-frame-slot dedup)
  → /var/log/flex/<carrier>.flexdec.log   (one "FLEX|…" line per page; ALN only)
```

---

## Streaming model: overlapped batch + dedup

The live decoder (`StreamDecoder`, in `core/flex_core.py`) does **not** rewrite
the decoder as a stateful streaming pipeline. Instead it re-runs the validated
batch decode on a sliding window and de-duplicates the results. This is provably
equivalent to a batch decode because FLEX is strictly periodic
(`FRAME_PERIOD = 30000` @ 16 kHz = 1.875 s) on the transmitter's clock — a frame
is self-contained, and there is no resampler or PLL state that could drift between
chunks.

- Each window is re-decoded with the full batch pipeline (matched-filter bank +
  per-frame CFO null + Chase-II FEC + the readability tier gate).
- Windows overlap by at least one frame, so every frame is fully inside at least
  one window (never sliced at an edge).
- Each decoded frame is tagged with its **absolute frame slot** computed from the
  window's stream position; de-duplication on `(slot, type, body)` collapses the
  repeats from the overlap, emitting each page exactly once.

The tap point is the **complex baseband** after channel-select (post freq-xlate,
pre channel-LPF) — the samples the matched-filter bank needs. Default window /
advance is 16 / 8 frames; the live receiver uses 32 / 28, which reproduces the
batch result exactly while leaving a comfortable overlap margin.

```python
from flex_core import StreamDecoder        # pure numpy/scipy, no GNU Radio
sd = StreamDecoder(on_page=lambda r: print(r))   # r = (slot, type, body, tier, pr, en)
for chunk in source_of_complex64_baseband_at_250k():
    sd.feed(chunk)
sd.flush()
```

---

## One OS process per carrier

The SDR feeds **one shared GNU Radio flowgraph** in the parent process. Each
carrier owns a `freq_xlating_fir_filter` (channel-select + decimate to 250 kS/s)
and a `RingSink` block that hands its baseband across a queue to a **dedicated
worker process** running a `StreamDecoder`.

- **Why processes, not threads.** The decode's hot kernels (the matched-filter
  bank, the `@njit` BCH/deinterleave loop) release the GIL, but the Python glue
  around them does not — so threads would serialize and only the busiest carrier
  would keep up. One OS process per carrier gives each its own GIL, and the
  per-carrier decode parallelizes across cores for real.
- **Backpressure.** `RingSink` copies each block into a **bounded** `mp.Queue`
  with **drop-oldest on overflow**, so a slow decode can never back-pressure the
  shared flowgraph (and thus the SDR). The queue is sized to hold one
  overlapped-window decode burst's worth of input so it drains between bursts.
  (The `--file` parity mode uses an unbounded queue so an offline replay never
  drops.)
- Each worker coalesces any queue backlog into a single `feed()` so the
  overlapped-window scheduler always sees large contiguous blocks.

---

## Configuration

The carrier set is supplied on the command line via `--carriers` (a comma-separated
list of channel center frequencies in MHz, all within the SDR's passband); the
tuner center defaults to their midpoint but can be pinned with `--center`. The DSP
parameters below live at the top of `flex_receiver.py`.

| Parameter | Value | Why |
|---|---|---|
| Sample rate | **2.5 MS/s** | `/10 → 250000` exactly matches the decoder's internal rate, so no resampling is needed — the validated pipeline runs untouched |
| Decimation | 10 | one integer factor, no fractional resampling |
| Window / advance | **32 / 28 frames** | ≥16 reproduces the batch result; 4-frame overlap margin |
| Pages emitted | **ALN only** | other page types are dropped before logging |
| Capcode in log | fixed `0` | collapses the viewer's `(capcode, body)` dedup to body-only, killing retransmits and the same page heard on two carriers |
| CPU budget | one core per carrier process + headroom for the parent flowgraph |

**Channel-select filter.** The per-carrier filter uses a wide (60 kHz) transition
band on purpose, which keeps the tap count low (~137 taps). It is still RF-safe:
after decimate-by-10 the first alias folds in around `IN_RATE − 9 kHz ≈ 241 kHz`,
leaving an enormous guard band. A narrow transition would inflate the tap count by
an order of magnitude and overrun the SDR for no benefit.

---

## Running

```bash
# Live receiver (carrier frequencies in MHz; center defaults to their midpoint):
python3 flex_receiver.py --live --carriers 929.6125,929.9375,931.2125

# Offline parity check — replay a recorded capture through the same
# ring → process → StreamDecoder path and print the trustworthy A/B page count:
python3 flex_receiver.py --file capture.cfile --in-rate 250000
```

The scripts insert both `core/` (shared decode core) and their own directory onto
`sys.path`, so they run from a repo checkout **and** from a flat install where
everything is co-located.

For continuous operation, run `flex_receiver.py --live --carriers …` as a long-lived process
under your service manager. One operational note: the per-carrier workers are
child **processes**, so stop them with a process-group kill (e.g. systemd
`KillMode=control-group`) to reap the whole group cleanly.

Logs land in `/var/log/flex/<carrier>.flexdec.log` as
`<ts> FLEX|<carrier>|<slot>|<tier>|0|ALN|<body>` lines, which the web viewer
(`viewer/`) tails and streams to browsers.

### POCSAG receiver

POCSAG dispatch channels sit far from the FLEX band, so `pocsag_receiver.py` runs
on its **own SDR** (default RTL-SDR) as a single-carrier receiver. It reuses the
`POCSAGStream` decoder and writes the **same** `FLEX|…` log line shape, so the web
viewer renders POCSAG pages with no change.

```bash
# Live (carrier frequency in MHz; --driver sdrplay to time-share an SDRplay tuner):
python3 pocsag_receiver.py --live --freq 152.0075 [--driver rtlsdr|sdrplay] [--log /var/log/flex]

# Offline parity check against a recorded capture:
python3 pocsag_receiver.py --file capture.cfile --in-rate 250000
```

---

## Performance characteristics

Each carrier decodes single-threaded within its own process. A carrier with no
sync short-circuits cheaply and stays well under realtime. A carrier carrying
heavy traffic runs the full matched-filter bank + Chase-II FEC every window and
sits near one core; under sustained heavy load it can tip just over realtime
during FEC-heavy windows, at which point the bounded queue drops the oldest IQ at
a slow steady rate — i.e. *partial* recall on the busiest carriers, never a stall.

The lever to lift that ceiling is **intra-carrier parallelism**: decoding several
frames of a window in a thread pool inside the worker. Because the njit/numpy
kernels release the GIL, this can use more than one core per carrier. Shrinking
the decode window reduces per-window CPU but trades against weak-carrier recall.
