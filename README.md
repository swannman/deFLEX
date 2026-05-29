# flexdec — a from-scratch FLEX paging decoder

A single-file numpy/scipy decoder for the FLEX paging protocol (TIA-1500), built
from the spec to see whether a modern DSP front end could match or beat
`multimon-ng` on real off-air traffic — and then grown into the live multi-carrier
receiver that replaced multimon in production.

The headline result: on a frozen 120 s benchmark capture, flexdec produces
**FEC-validated, garbage-free** alpha decodes that are cleaner than multimon's,
and it **decisively wins on weak carriers** — recovering a lab-temperature
broadcast at −27.5 dB that multimon renders as noise. That same validated decode
core now runs **live and continuously** on all five active carriers.

---

## The signal

FLEX is a one-way paging protocol. The benchmark carrier (929.6125 MHz, Spok
network, Seattle area) is **mode 3: 3200 baud, 4-level FSK, inverted polarity**.

| Parameter | Value | Notes |
|---|---|---|
| Frame period | 1.875 s = **30000 samples @ 16 kHz** | strictly periodic; 128 frames = 4-min cycle |
| Symbol rate | 3200 / 1600 baud | mode 3 / mode 1; only ~5 samples/symbol at 16 kHz |
| Modulation | 4-FSK (`±4800 / ±1600 Hz`) | tone spacing = baud → tones orthogonal over a symbol |
| Frame sync | `0xA6C6AAAA` + mode codeword | packed into a 64-bit sync value |
| Frame data | 11 blocks × 8 words = **88 words** | each word BCH(31,21) + parity |
| FEC | BCH(31,21,5) | corrects 1, detects 2 errors per word |

Optimized for **alpha** (ALN) pages — text that can be validated; numeric pages
can't be, so they're dropped.

---

## Repository layout

Two approaches, deliberately kept separate, over one shared decode core:

```
flexdec.py          ← shared core: the validated batch FLEX decoder
flexdec_numba.py    ← shared core: @njit kernels (GIL-releasing) for the hot loop

offline/            ← batch research / A-B tool  (see offline/README.md)
  compare_ab.py       fair flexdec-vs-multimon comparison harness

online/             ← live real-time receiver    (see online/README.md)
  flexdec_stream.py   StreamDecoder: overlapped-batch replay + dedup
  flex_inmem.py       production: SDR → freq-xlate → ring → worker → decoder
  flex_stream_live.py standalone single-carrier .cfile tail (dev)
  flex-receiver-flexdec.service
  legacy/
    flex_7ch.py       retired multimon-ng flowgraph (kept for A/B provenance)
```

- **`flexdec.py` / `flexdec_numba.py` (root)** — the shared core, imported
  unchanged by both sides. `flexdec.py` was frozen as the known-good baseline,
  then unfrozen 2026-05-29 to add numba acceleration; the njit path is **bit-exact**
  vs pure Python (validated IDENTICAL A/B set, 32 pages) and its real win is
  releasing the GIL so per-carrier threads parallelize. Pure-Python fallback
  behind a `_HAVE_NUMBA` guard.
- **[`offline/`](offline/README.md)** — run `flexdec.py` over a frozen `.cfile`:
  the algorithm narrative, confidence tiering, the `--alpha` English gate, the
  full A/B-vs-multimon results, and the Phase 2 wideband (all-7-carrier) study.
- **[`online/`](online/README.md)** — the live receiver: the streaming wrapper
  (`StreamDecoder`), the in-memory SDR path, the 2.5 MS/s / numba / ALN-only /
  `CPUQuota` design, the systemd unit, and the deploy notes.

---

## Status

The live in-memory flexdec receiver **replaced multimon-ng in production on
2026-05-29** (`flex-receiver.service` on p340): RSPdx @ 2.5 MS/s → 5 carriers →
`StreamDecoder` threads → `/var/log/flex`, feeding the existing web viewer with
no server change. See [`online/README.md`](online/README.md) for the full design
and one **open limitation** (5 concurrent decoders don't yet fit the 3-core CPU
budget — only the strongest carrier currently keeps up).

The decoder is **not** GPL `multimon-ng`-derived: it was built from the TIA-1500
tables (using `gr-pager` only as a protocol reference, not linked).
