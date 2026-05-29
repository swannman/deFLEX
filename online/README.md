# flexdec — online streaming & live receiver

The **online** side of flexdec: the real-time path that turns the frozen batch
decoder (`flexdec.py`, at the repo root) into a continuously-running multi-carrier
receiver fed directly by an SDR — **no intermediate files, no multimon-ng.**

This is the production FLEX receiver on **p340** (`flex-receiver.service`). As of
**2026-05-29** it replaced the legacy multimon-ng flowgraph (kept for reference
in `legacy/`).

```
RSPdx @ 930.7625 MHz, 2.5 MS/s
  → 5× freq_xlating_fir_filter_ccc (decim 10 → 250 kS/s baseband per carrier)
  → RingSink  (bounded queue, drop-oldest on overflow — never back-pressures SDR)
  → CarrierWorker thread  (one per carrier)
  → StreamDecoder  (overlapped-batch replay of flexdec.py + per-frame-slot dedup)
  → /var/log/flex/<carrier>.flexdec.log   (legacy "FLEX|" line shape; ALN only)
```

| File | Role |
|---|---|
| `flexdec_stream.py` | `StreamDecoder`: overlapped-window scheduler + dedup wrapping the batch core |
| `flex_inmem.py` | **production**: SDR → freq-xlate → ring → worker → StreamDecoder (`--live`); also `--file` parity mode |
| `flex_stream_live.py` | standalone single-carrier tail of a growing `.cfile` (dev/debug) |
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

## Live in-memory path (`flex_inmem.py`)

Replaces the old multimon FIFO. The SDR feeds one shared GNU Radio flowgraph;
each carrier owns a `freq_xlating_fir_filter` (channel-select + decimate to
250 kS/s), a `RingSink` sync block, and a `CarrierWorker` thread driving a
`StreamDecoder`.

- **`RingSink`** hands each `work()` block to a **bounded queue, drop-oldest on
  overflow.** A slow decode can *never* back-pressure the shared flowgraph (and
  thus the SDR) — it drops the oldest IQ instead. (`--file` parity mode uses an
  unbounded queue so the offline comparison never drops.)
- **`CarrierWorker`** coalesces any queue backlog into a single `feed()` so the
  overlapped-window scheduler always sees large contiguous blocks.
- **numba parallelism:** `mf_bank_mag` (the ~58% hotspot) is vectorized numpy
  and the BCH/deinterleave inner loop is `@njit` (`flexdec_numba.py`), both
  releasing the GIL — so the per-carrier worker threads run their matched-filter
  banks concurrently across cores. No multiprocessing, no 8-core blowup.

### Design decisions baked into the live config

| Decision | Value | Why |
|---|---|---|
| Sample rate | **2.5 MS/s** | `/10 → 250000 == flexdec.SAMP` exactly, so `resample_poly` is **skipped** (resample=OFF) — the validated pipeline runs untouched, no resampler cost |
| Decimation | 10 | one integer factor, no fractional resampling |
| Window / advance | **32 / 28 frames** | 32 ≥ 16 reproduces the batch A/B set; 4-frame overlap margin; ~0.54× realtime per carrier |
| Pages emitted | **ALN only** | SPN decodes to control-char garbage on these carriers; dropped before logging or feeding the viewer |
| Capcode in log | fixed `0` | collapses the viewer's `(capcode, body)` dedup to body-only → kills retransmits + the same page heard on two carriers |
| CPU budget | `CPUQuota=300%` | self-imposed 3-core cap so the receiver can never eat the whole box (p340 has 16 cores) |

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
python3 flex_inmem.py --live

# Offline parity check — feed a recorded capture through the SAME
# ring → thread → StreamDecoder path and print the trustworthy A/B page count:
python3 flex_inmem.py --file iq_929612500_250k.cfile --in-rate 250000
```

The scripts insert both the repo root (shared core) and this directory onto
`sys.path`, so they run from a checkout **and** from the flat `/usr/local/bin`
install where everything is co-located.

### systemd

`flex-receiver-flexdec.service` is installed as `flex-receiver.service` on p340:

- `User=swannman`, `ExecStart=/usr/bin/python3 /usr/local/bin/flex_inmem.py --live`
- `CPUQuota=300%`, `KillMode=control-group`
- `Restart=on-failure`, `RestartSec=5` (also self-heals the SDR-handoff race on
  first start, when the previous owner hasn't released the RSPdx yet)

Logs land in `/var/log/flex/<carrier>.flexdec.log` in the legacy
`<ts> FLEX|<carrier>|<slot>|<tier>|0|ALN|<body>` shape, so the existing web feed
(`flex-tail.service`, FastAPI/uvicorn on `http://192.168.5.176:8091/`) parses
them with **no server change**.

---

## Known limitation (open)

**Five concurrent decoders do not fit in the 300% (3-core) CPU budget.** After
the filter-tap fix eliminated the front-end overruns, the *decode stage* itself
becomes the bottleneck: in practice only the strongest carrier (929.6125)
keeps up and produces pages, while the other four drop most of their IQ at the
`RingSink` (dropped-chunk counter climbing steadily) and emit ~zero pages.

The levers — all deliberately **not** changed unilaterally because the current
values were set on purpose:

- **raise `CPUQuota`** (e.g. 400–500%) — gives the decoders more cores, at the
  cost of the "never eat the box" guarantee;
- **drop the silent 929.5875 carrier** — frees one decoder, but it's "kept by
  request";
- **shrink the decode window** (32 → 16 frames) — less CPU per window, at some
  recall risk on weak carriers.

This is the unresolved part of the live cutover and needs an explicit call on the
CPU-budget-vs-carrier-count tradeoff.

---

## Legacy (`legacy/flex_7ch.py`)

The retired production flowgraph: RSPdx @ 2.646 MS/s → 7 channels →
FM-discriminator → FIFO → `multimon-ng`. Kept for reference and for the A/B
provenance (the multimon side of every comparison in `../offline/`). Superseded
by the in-memory flexdec path on 2026-05-29.
