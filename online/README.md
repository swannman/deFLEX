# flexdec — online streaming & live receiver

The **online** side of flexdec: the real-time path that turns the frozen batch
decoder (`flexdec.py`, at the repo root) into a continuously-running multi-carrier
receiver fed directly by an SDR — **no intermediate files, no multimon-ng.**

This is the production FLEX receiver on the production host (`flex-receiver.service`). As of
**2026-05-29** it replaced the legacy multimon-ng flowgraph (kept for reference
in `legacy/`). The production path is the **multiprocessing** variant
(`flex_inmem_mp.py`) — one OS process per carrier to beat the GIL (see below);
the threaded `flex_inmem.py` is kept as the GIL-bound reference / `--file` parity
harness.

```
RSPdx @ 930.7625 MHz, 2.5 MS/s         (one shared GNU Radio flowgraph, parent process)
  → 5× freq_xlating_fir_filter_ccc (decim 10 → 250 kS/s baseband per carrier)
  → RingSink  (per-carrier mp.Queue, drop-oldest on overflow — never back-pressures SDR)
  ══ process boundary (complex64 chunks pickled across the queue) ══
  → worker_proc  (one OS PROCESS per carrier, own GIL)
  → StreamDecoder  (overlapped-batch replay of flexdec.py + per-frame-slot dedup)
  → /var/log/flex/<carrier>.flexdec.log   (legacy "FLEX|" line shape; ALN only)
```

| File | Role |
|---|---|
| `flexdec_stream.py` | `StreamDecoder`: overlapped-window scheduler + dedup wrapping the batch core |
| `flex_inmem_mp.py` | **production**: SDR → freq-xlate → ring → **per-carrier process** → StreamDecoder (`--live`); also `--file` parity mode |
| `flex_inmem.py` | threaded variant (one thread per carrier): GIL-bound, superseded by `flex_inmem_mp.py`; still useful for `--file` parity and as the single source of the SDR/carrier constants |
| `flex_stream_live.py` | standalone single-carrier tail of a growing `.cfile` (dev/debug) |
| `pocsagdec_stream.py` | `POCSAGStream`: the POCSAG analogue of `flexdec_stream.py` — overlapped-batch replay of `pocsagdec.py` with emit-region exactly-once emission |
| `pocsag_receiver.py` | live POCSAG receiver (VHF fire/EMS dispatch) feeding `POCSAGStream` |
| `flex-receiver-flexdec.service` | systemd unit (installed as `flex-receiver.service`) |
| `legacy/flex_7ch.py` | **retired** 7-channel multimon flowgraph (2.646 MS/s → FM-discriminator → FIFO) |

---

## Streaming core (`flexdec_stream.py`)

A live decoder built **on top of** the frozen batch core — `flexdec.py` is
imported unchanged, so the streaming path can never drift from the validated A/B
baseline. The only added code is the *scaffolding*: a complex-baseband ring
buffer, an overlapped-window scheduler, and per-frame-slot dedup. The actual
symbol decode replays `flexdec.main()`'s per-frame loop verbatim (matched-filter
bank + per-frame CFO null + Chase-II soft FEC + the `--alpha` English-likeness
tier gate).

**Why overlapped-batch + dedup, not a stateful streaming demod.** FLEX is
strictly periodic (`FRAME_PERIOD = 30000` @ 16 kHz = 1.875 s) on a shared
transmitter clock, so a frame is self-contained. Re-running the validated batch
decode on a sliding window and de-duplicating by *absolute frame slot* is
provably equivalent to batch — there is no resampler state to drift. The window
overlaps by ≥ one frame so every frame is fully inside at least one window;
dedup on `(slot, type, body)` collapses the repeats.

- **Tap point:** the **complex baseband** after channel-select (post freq-xlate,
  pre channel-LPF) — the same samples the matched-filter bank needs. Tapping the
  real FM-discriminator output instead (what multimon consumed) would forfeit the
  MF-bank quality win (A+B = 34 vs 24 on the benchmark).
- **Window / advance:** default **16 frames (30 s) / 8 frames (15 s)**; the live
  receiver uses **32 / 28** (see below). `corr_frames`' `grid_phase` locks the
  frame grid from the sync anchors *in the window*, so a window needs ~16 anchors
  for its grid to match the whole-capture grid to sub-symbol precision. 8/4
  dropped one marginal 3-char SPN page; 16/8 (and larger) reproduce the batch A/B
  set exactly.

**Validation** (`python3 flexdec_stream.py [cfile]`): replays the frozen 250 k
benchmark in 2 s chunks through `StreamDecoder.feed()` and asserts the streamed
trustworthy-A/B set is a superset of a whole-file batch decode of the same data.
Result on `iq_929612500_250k.cfile`: **batch 32, streamed 32, 0 missing, 0
extra → PASS.**

Library use:

```python
import flexdec_stream as S
sd = S.StreamDecoder(on_page=lambda r: print(r))   # r = (slot, type, body, tier, pr, en)
for chunk in source_of_complex64_baseband_at_250k():
    sd.feed(chunk)
sd.flush()
```

---

## Live in-memory path (`flex_inmem_mp.py`)

Replaces the old multimon FIFO. The SDR feeds one shared GNU Radio flowgraph in
the **parent** process; each carrier owns a `freq_xlating_fir_filter`
(channel-select + decimate to 250 kS/s) and a `RingSink` sync block that hands
its baseband to a **dedicated worker PROCESS** (`worker_proc`) running a
`StreamDecoder`.

- **Why processes, not threads.** The threaded `flex_inmem.py` serializes on the
  shared GIL: the njit/numpy hot kernels *release* the GIL, but the Python glue
  around them holds it, so the receiver pinned at **~3 cores regardless of how
  many were free** and only the strongest carrier kept up. One OS process per
  carrier gives each its own GIL → the per-carrier decode parallelizes for real
  (measured ~5 cores vs ~3). `--file` parity is bit-exact (32 pages) vs threaded
  and batch, so the process boundary doesn't change *what* gets decoded, only how
  fast.
- **`RingSink` → `mp.Queue`** hands each `work()` block (a **copied** complex64
  chunk — GNU Radio reuses its buffer and the queue pickles asynchronously) to a
  **bounded queue, drop-oldest on overflow.** A slow decode can *never*
  back-pressure the shared flowgraph (and thus the SDR) — it drops the oldest IQ
  instead. (`--file` parity mode uses an unbounded queue so the offline
  comparison never drops.)
- **Queue depth (`QUEUE_MAXCHUNKS = 1024`).** The worker stops draining its queue
  for the multi-second overlapped-window decode burst; at the original 64 the
  queue overflowed during *every* burst (~30 drops/s per carrier, even on carriers
  with no sync). 1024 (~17 s of 250 kS/s baseband, ~64 MB/carrier worst case) lets
  the queue hold one window's worth of input while the worker decodes, then drain
  between bursts. This eliminates drops on any carrier whose decode stays under
  realtime (see *Known limitation* for the carriers that don't).
- **The worker** coalesces any queue backlog into a single `feed()` so the
  overlapped-window scheduler always sees large contiguous blocks. A `None` on the
  queue is the EOF sentinel (file mode → flush and exit).
- **numba inside each worker:** `mf_bank_mag` (the ~58% hotspot) is vectorized
  numpy and the BCH/deinterleave inner loop is `@njit` (`flexdec_numba.py`). Within
  one worker the decode is single-threaded; the cross-core parallelism comes from
  running five such workers, one per carrier.

### Design decisions baked into the live config

| Decision | Value | Why |
|---|---|---|
| Sample rate | **2.5 MS/s** | `/10 → 250000 == flexdec.SAMP` exactly, so `resample_poly` is **skipped** (resample=OFF) — the validated pipeline runs untouched, no resampler cost |
| Decimation | 10 | one integer factor, no fractional resampling |
| Window / advance | **32 / 28 frames** | 32 ≥ 16 reproduces the batch A/B set; 4-frame overlap margin |
| Pages emitted | **ALN only** | SPN decodes to control-char garbage on these carriers; dropped before logging or feeding the viewer |
| Capcode in log | fixed `0` | collapses the viewer's `(capcode, body)` dedup to body-only → kills retransmits + the same page heard on two carriers |
| Queue depth | **1024 chunks/carrier** | absorbs one window-decode burst so the bounded queue doesn't drop while the worker is busy (see live path above) |
| CPU budget | `CPUQuota=600%` | 6-core cap on the 16-core box: one core per carrier process + headroom for the parent flowgraph. (The old threaded build used 300%, which choked it — MP needs ~5.) |

### The anti-alias filter tap-count gotcha (important)

```python
taps = firdes.low_pass(1.0, SAMP_RATE, 9000, 60000)   # 9 kHz cutoff, 60 kHz transition
```

Tap count ≈ `3.3 × fs / transition_width`. At 2.5 MS/s a **3 kHz** transition →
**~2750 taps per carrier**; ×5 carriers at the full input rate saturated the
3-core budget, the SDR overran continuously (`OsOsOs…`), and **nothing decoded.**
A **60 kHz** transition → **~137 taps**, and it's RF-safe: after decimate-by-10
the first alias folds in at `IN_RATE − 9000 ≈ 241 kHz`, so a 60 kHz transition
still leaves an enormous guard band. **Do not tighten this transition.**

### Carriers

| Carrier (Hz) | Network | Mode | Note |
|---|---|---|---|
| 929 587 500 | Spok | — | silent/unmeasured — **kept by request** |
| 929 612 500 | Spok | 3200 / 4-lvl | strongest |
| 929 937 500 | American Msg | 1600 / 4-lvl | |
| 931 212 500 | Spok | 3200 / 4-lvl | |
| 931 937 500 | SkyTel | 1600 / 4-lvl | |

Dropped vs the 7-channel legacy set: 929.6625 (test) and 931.0625 (POCSAG, not
FLEX).

---

## Running

```bash
# Live receiver (production form):
python3 flex_inmem_mp.py --live

# Offline parity check — feed a recorded capture through the SAME
# ring → process → StreamDecoder path and print the trustworthy A/B page count:
python3 flex_inmem_mp.py --file iq_929612500_250k.cfile --in-rate 250000
```

The scripts insert both the repo root (shared core) and this directory onto
`sys.path`, so they run from a checkout **and** from the flat `/usr/local/bin`
install where everything is co-located.

### systemd

`flex-receiver-flexdec.service` is installed as `flex-receiver.service` on the
production host:

- `User=<user>`, `WorkingDirectory=/usr/local/bin`,
  `ExecStart=/usr/bin/python3 /usr/local/bin/flex_inmem_mp.py --live`
- `CPUQuota=600%`, `KillMode=control-group` (the per-carrier worker processes live
  in the unit cgroup, so `systemctl stop` reaps the whole group)
- `Restart=on-failure`, `RestartSec=5` (also self-heals the SDR-handoff race on
  first start, when the previous owner hasn't released the RSPdx yet)

Logs land in `/var/log/flex/<carrier>.flexdec.log` in the legacy
`<ts> FLEX|<carrier>|<slot>|<tier>|0|ALN|<body>` shape, so the existing web feed
(`flex-tail.service`, FastAPI/uvicorn on `http://<host>:8091/`) parses
them with **no server change**.

---

## Known limitation (open)

Multiprocessing **largely solved** the original "only the strongest carrier keeps
up" problem — the threaded build was GIL-capped at ~3 cores and starved four of
five carriers; MP now runs all five concurrently and the three active carriers
all produce real ALN pages. What remains is a **smaller, per-carrier ceiling**:

**Each carrier's decode is single-threaded, and the active carriers sit right at
~1 core.** A worker that finds no sync short-circuits before Chase-II FEC and
stays comfortably under realtime (the silent carriers hold **0 drops** with the
1024-deep queue). But a worker decoding real frames runs the full
matched-filter-bank + Chase-II FEC and sits at **~92 % of one core**, tipping
just *over* realtime during FEC-heavy windows. Backlog then grows without bound
and the `RingSink` drops at a slow steady rate (~10–20 chunks/s) — a deeper queue
only delays this, it can't fix an unbounded backlog. The result is *partial*
recall on the active carriers, not the threaded build's near-total starvation.

The remaining levers — all deliberately **not** changed unilaterally because they
trade against recall or touch the frozen core:

- **intra-carrier parallelism** — decode multiple frames of a window in a thread
  pool inside the worker (the njit/numpy kernels release the GIL, so this *can*
  use >1 core per carrier). Real code work, but the principled fix;
- **shrink the decode window** (32 → 24/16 frames) — less CPU per window, but
  &lt;16 is unvalidated and risks recall on exactly the weak carriers we want;
- **cap Chase-II** on weak/expensive frames — directly cuts the FEC cost, but
  touches the validated `flexdec.py` core (would need an A/B re-validation);
- **raise `CPUQuota`** — does *not* help here: the cap isn't the limiter (workers
  total ~480 % under the 600 % cap), the single-core-per-process ceiling is.

This is the unresolved part of the live cutover and needs an explicit call on the
recall-vs-effort tradeoff.

---

## Legacy (`legacy/flex_7ch.py`)

The retired production flowgraph: RSPdx @ 2.646 MS/s → 7 channels →
FM-discriminator → FIFO → `multimon-ng`. Kept for reference and for the A/B
provenance (the multimon side of every comparison in `../offline/`). Superseded
by the in-memory flexdec path on 2026-05-29.
