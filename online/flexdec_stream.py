#!/usr/bin/env python3
"""Streaming FLEX decoder built ON TOP of the validated batch core (flexdec.py).

flexdec.py is the frozen A/B baseline; this module imports it and reuses every
leaf DSP/FEC function unchanged. The only thing it adds is the *streaming
scaffolding*: a complex-baseband ring buffer, an overlapped-window scheduler,
and per-frame-slot dedup. The actual symbol decode replays flexdec's main()
per-frame loop verbatim (matched-filter bank + per-frame CFO null + Chase-II
soft FEC + the alpha English-likeness tier gate), so a streamed run produces
exactly the same trustworthy A/B messages a batch run would.

Why "overlapped batch + dedup" instead of a stateful streaming demod:
  * FLEX is strictly periodic (FRAME_PERIOD = 30000 @ 16 kHz = 1.875 s) on a
    shared transmitter clock. A frame is self-contained, so re-running the
    validated batch decode on a sliding window and de-duplicating by absolute
    frame slot is provably equivalent to batch -- no resampler state to drift.
  * The window overlaps by >= one frame so every frame is fully contained in at
    least one window; dedup (slot, type, body) collapses the repeats.

Validation target: replay iq_929612500_250k.cfile (already complex @ 250k, no
resampling) in chunks; the streamed A/B set must be a superset of the batch set.
"""
import os
import re
import sys
import numpy as np
from scipy import signal

_HERE = os.path.dirname(os.path.abspath(__file__))
# Shared core (flexdec.py / flexdec_numba.py) lives at the repo root; in the
# flat /usr/local/bin install it sits beside this file -- cover both layouts.
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import flexdec as F

# 250k samples per FLEX frame period (468750) -- the streaming quantum.
FRAME_250 = int(round(F.FRAME_PERIOD * F.SAMP / F.FS))

# Dedup-key normalization. Overlapping windows re-decode the same frame slot;
# the low-reliability tail of a short/empty message gets Chase-corrected into
# slightly different junk each pass -- varying whitespace and a stray trailing
# "[nn]" -- so an exact (slot,typ,body) key emits the same logical page several
# times. Collapse whitespace and strip trailing "[nn]" tokens for the KEY ONLY;
# the full original body is still logged. A/B on the frozen strong+weak captures:
# every collapse was a genuine same-message duplicate (e.g. an "HMC Rapid
# Response" page recurring with [], [84], [94]); no distinct real page merged.
_WS_RUN = re.compile(rb"\s+")
_TRAIL_BRACKET = re.compile(rb"(\s*\[\d+\])+\s*$")

def dedup_body(body):
    return _TRAIL_BRACKET.sub(b"", _WS_RUN.sub(b" ", body).strip()).strip()

# Mirror of the validated `--mfbank --corr --comb --sweep --soft --alpha` run.
DEFAULT_CFG = dict(
    mfbank=True, nocfo=False, coh=False, coh_alpha=0.05, inv=False,
    soft=True, slicer="perframe", sweep=True, sweep_half=1, frac=False,
    alpha_only=True, comb=True, mflen=F.SPB, corr_off=1517.0,
    lpf=12000.0, in_rate=F.SAMP, MARGIN_OK=0.5, ALPHA_EN_OK=0.60,
)


def _classify(typ, body, pc, cfg):
    """Replica of flexdec.main()'s tier_of + the pr/en computation around it."""
    pr = sum(1 for c in body if 32 <= c < 127) / len(body)
    en = F.english_score(body) if cfg["alpha_only"] else 1.0
    if pc is None:
        t = "C"
    elif pc["failed"] > 0:
        t = "D"
    elif pc["margin"] == pc["margin"] and pc["margin"] < cfg["MARGIN_OK"]:
        t = "C"
    elif pc["corrected"] or pc["chase"]:
        t = "B"
    else:
        t = "A"
    if t in ("A", "B") and pr < 0.85:
        t = "C"
    if cfg["alpha_only"] and en < cfg["ALPHA_EN_OK"] and t in ("A", "B"):
        t = "C"
    return t, pr, en


def decode_window(xb, s0, cfg):
    """Decode one complex-baseband window. `s0` = absolute index (250k samples)
    of xb[0] in the stream, used to assign each frame its global slot number.
    Returns list of (slot, typ, body_bytes, pc). This is flexdec.main()'s frame
    loop (lines ~998-1055) lifted verbatim, parameterized by cfg/xb/demod.

    `xb` is the channel-selected complex baseband (post freq-xlate). If its rate
    `cfg["in_rate"]` differs from the internal SAMP (250k) -- e.g. a live tap at
    22050 Hz -- we polyphase-resample the window to 250k first, exactly as
    load_baseband() does, then apply the same 127-tap channel LPF, so the
    matched-filter bank and FM demod see identical samples to a batch run. The
    resample+filter run per-window; their edge transient is far shorter than a
    frame, and window overlap keeps every frame away from an edge in at least one
    window. `s0` is in INPUT-rate samples (mapped to the 16k grid for slots)."""
    in_rate = cfg["in_rate"]
    if in_rate != F.SAMP:
        from math import gcd
        g = gcd(int(round(in_rate)), F.SAMP)
        xb = signal.resample_poly(xb, F.SAMP // g, int(round(in_rate)) // g)
    taps = signal.firwin(127, cfg["lpf"] / (F.SAMP / 2))
    xb = signal.lfilter(taps, 1.0, xb)
    demod = F.demod_from_baseband(xb, mf=False, mflen=cfg["mflen"])
    demod = demod - np.median(demod)                 # global DC/CFO removal
    corr_thr = -1.0 if cfg["alpha_only"] else 0.35
    frames = F.corr_frames(demod, p0_offset=cfg["corr_off"], grid=True,
                           include_thr=corr_thr)
    if cfg["comb"]:
        frames = F.add_comb_frames(frames, len(demod), verbose=False)

    xp = np.arange(len(demod), dtype=float)

    def sample_at(positions):
        return np.interp(positions, xp, demod)

    def decode_syms(syms, baud, levels, chase=False):
        if cfg["slicer"] == "fixed":
            thr, dc = 2.0, 0.0
        else:
            dc = np.median(syms)
            lo, hi = F.two_means(np.abs(syms - dc))
            thr = (lo + hi) / 2
        phases, mags = F.slice_soft(syms, baud, levels, thr, dc)
        return F.decode_phases(phases, mags, chase=chase)

    def decode_mf(pos16, baud, levels, chase=False, use_coh=False):
        cfo = 0.0 if cfg["nocfo"] else F.est_cfo(
            xb, pos16[0] * F.SAMP / F.FS, pos16[-1] * F.SAMP / F.FS)
        if use_coh:
            C = F.mf_bank_mag(xb, pos16, baud, cfo=cfo, inv=cfg["inv"],
                              complex_out=True)
            R = F.coherent_metric(C, baud, inv=cfg["inv"], alpha=cfg["coh_alpha"])
            best = None
            for sgn in (1.0, -1.0):
                ba, bb, ma, mb = F.mfbank_softbits(sgn * R)
                ph, mg = F.demux_phases(ba, bb, ma, mb, baud)
                res = F.decode_phases(ph, mg, chase=chase)
                if best is None or res[2] < best[2]:
                    best = res
            return best
        metric = F.mf_bank_mag(xb, pos16, baud, cfo=cfo, inv=cfg["inv"])
        bit_a, bit_b, mag_a, mag_b = F.mfbank_softbits(metric)
        phases, mags = F.demux_phases(bit_a, bit_b, mag_a, mag_b, baud)
        return F.decode_phases(phases, mags, chase=chase)

    results = []
    for f in frames:
        spb = f["spb"]; p0 = f["p0"]; nsyms = f["nsyms"]
        if not cfg["sweep"]:
            offs = [0.0]
        elif cfg["frac"]:
            offs = list(np.arange(-spb / 2, spb / 2, 0.5))
        else:
            # Timing-phase search around the sync-locked grid. mf_bank_mag is
            # ~68% of decode CPU and the sweep multiplies it by len(offs)*len(pars)
            # per frame; sweep_half bounds the integer offset window. ±1 (3 offsets)
            # vs the full ±spb//2: A/B on the frozen strong+weak captures dropped
            # only one garbage SPN that had squeaked past the English gate and
            # gained one real page on the weak carrier, while cutting mf_bank_mag
            # calls ~36%. None=full width.
            _h = cfg.get("sweep_half")
            if _h is not None:
                offs = [float(o) for o in range(-_h, _h + 1)]
            else:
                offs = [float(o) for o in range(-(spb // 2), spb - spb // 2)]
        pars = [0, 1] if f["baud"] == 3200 else [0]
        best = None
        for par in pars:
            for off in offs:
                base = p0 + off + par * spb
                n = nsyms - par
                pos = base + np.arange(n) * spb
                if pos[0] < 0 or pos[-1] >= len(demod) - 1:
                    continue
                if cfg["mfbank"]:
                    wordsets, confsets, nf, nc, _ = decode_mf(pos, f["baud"], f["levels"])
                else:
                    syms = sample_at(pos)
                    wordsets, confsets, nf, nc, _ = decode_syms(syms, f["baud"], f["levels"])
                if best is None or nf < best[1]:
                    best = (off, nf, nc, wordsets, confsets, par, pos)
        if best is None:
            continue
        off, nf, nc, wordsets, confsets, par, bpos = best
        if cfg["mfbank"] and (cfg["coh"] or cfg["soft"]):
            wordsets, confsets, nf, nc, nch = decode_mf(
                bpos, f["baud"], f["levels"], chase=cfg["soft"], use_coh=cfg["coh"])
        elif cfg["soft"]:
            bsyms = sample_at(bpos)
            wordsets, confsets, nf, nc, nch = decode_syms(
                bsyms, f["baud"], f["levels"], chase=True)
        abs16 = s0 * F.FS / cfg["in_rate"] + p0  # this frame's absolute 16k position
        slot = int(round(abs16 / F.FRAME_PERIOD))
        for words, cf in zip(wordsets, confsets):
            if words is not None:
                for cap, typ, body, pc in F.parse_frame(words, cf):
                    results.append((slot, typ, body, pc))
    return results


class StreamDecoder:
    """Feed complex baseband (already channelized to one carrier at cfg['in_rate'],
    default 250k) in arbitrary-sized chunks; emits trustworthy A/B alpha pages
    exactly once each. Use `on_page` callback or read `self.pages` after `flush()`."""

    def __init__(self, cfg=None, window_frames=16, advance_frames=8, on_page=None):
        # 16-frame (30 s) window / 8-frame (15 s) advance: corr_frames' grid_phase
        # locks the frame grid from the sync anchors in the window, so a window
        # needs enough anchors (~16) for its grid to match the whole-capture grid
        # to sub-symbol precision -- otherwise a borderline page can slip just
        # under the tier-B margin. 8/4 lost one marginal 3-char SPN; 16/8 (and
        # larger) reproduce the batch A/B set exactly.
        self.cfg = dict(DEFAULT_CFG, **(cfg or {}))
        # window/advance counted in INPUT-rate samples (one frame period at in_rate)
        self.frame_samp = int(round(F.FRAME_PERIOD * self.cfg["in_rate"] / F.FS))
        self.window = window_frames * self.frame_samp
        self.advance = advance_frames * self.frame_samp
        self.on_page = on_page
        self.buf = np.empty(0, dtype=np.complex64)
        self._pending = []            # chunks fed but not yet merged into buf
        self._pending_len = 0
        self.base = 0                 # absolute input-rate index of buf[0]
        self.seen = set()             # (slot, typ, dedup_body(body)) already emitted
        self.pages = []               # accepted A/B pages (slot, typ, body, tier, pr, en)

    def _emit(self, results):
        for slot, typ, body, pc in results:
            if self.cfg["alpha_only"] and typ not in ("ALN", "SPN"):
                continue
            if len(body) == 0:
                continue
            t, pr, en = _classify(typ, body, pc, self.cfg)
            if t not in ("A", "B"):
                continue
            key = (slot, typ, dedup_body(body))
            if key in self.seen:
                continue
            self.seen.add(key)
            rec = (slot, typ, body, t, pr, en)
            self.pages.append(rec)
            if self.on_page:
                self.on_page(rec)

    def _drain(self, final=False):
        # Slide a fixed window across the buffer, advancing by `advance` (< window
        # for overlap). Non-final: only fire on full windows, leaving the overlap
        # tail buffered for the next feed. Final: also process the short tail
        # (needs >= ~1.5 frames to possibly contain a decodable frame).
        min_final = int(1.5 * self.frame_samp)
        while True:
            have = len(self.buf)
            if not final:
                if have < self.window:
                    return
                self._emit(decode_window(self.buf[:self.window], self.base, self.cfg))
                self.buf = self.buf[self.advance:]
                self.base += self.advance
            else:
                if have < min_final:
                    return
                end = min(self.window, have)
                self._emit(decode_window(self.buf[:end], self.base, self.cfg))
                if end >= have:                  # tail fully covered; done
                    self.buf = self.buf[end:]
                    self.base += end
                    return
                self.buf = self.buf[self.advance:]
                self.base += self.advance

    def _merge_pending(self):
        # Concatenate the buffer ONCE per window-fill instead of once per feed.
        # buf can hold a full 32-frame window (~15M complex64 ≈ 120 MB); the old
        # per-feed np.concatenate([buf, chunk]) recopied all of it for every small
        # SDR chunk (~120 MB memcpy ×30/s at 8k-sample chunks), which dominated
        # live CPU (py-spy: 99% in feed) and scaled inversely with chunk size.
        if not self._pending:
            return
        parts = ([self.buf] if len(self.buf) else []) + self._pending
        self.buf = parts[0] if len(parts) == 1 else np.concatenate(parts)
        self._pending = []
        self._pending_len = 0

    def feed(self, samples):
        s = np.asarray(samples, dtype=np.complex64)
        self._pending.append(s)
        self._pending_len += len(s)
        # Only merge+drain once enough is buffered to form at least one window;
        # _drain processes only full windows, so batching can't drop a window.
        if len(self.buf) + self._pending_len >= self.window:
            self._merge_pending()
            self._drain(final=False)

    def flush(self):
        self._merge_pending()
        self._drain(final=True)
        return self.pages


def _ab_set(records):
    return {(slot, typ, body) for (slot, typ, body, *_rest) in records}


def validate(cfile="/tmp/flex_ab/iq_929612500_250k.cfile", chunk_s=2.0, cfg=None):
    """Batch (whole-file single window) vs streamed (chunked) on the same data."""
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    x = np.fromfile(cfile, dtype=np.complex64)
    print(f"loaded {len(x)} complex samples ({len(x)/F.SAMP:.1f}s @ {F.SAMP/1e3:.0f}k)")

    # --- batch reference: one window over the entire capture ---
    batch_raw = decode_window(x, 0, cfg)
    batch = []
    seen = set()
    for slot, typ, body, pc in batch_raw:
        if typ not in ("ALN", "SPN") or len(body) == 0:
            continue
        t, pr, en = _classify(typ, body, pc, cfg)
        if t in ("A", "B"):
            k = (slot, typ, body)
            if k not in seen:
                seen.add(k)
                batch.append((slot, typ, body, t, pr, en))
    bset = _ab_set(batch)
    print(f"batch trustworthy A/B alpha pages: {len(bset)}")

    # --- streamed: feed the same samples in chunks ---
    sd = StreamDecoder(cfg=cfg)
    chunk = int(chunk_s * F.SAMP)
    for i in range(0, len(x), chunk):
        sd.feed(x[i:i + chunk])
    sd.flush()
    sset = _ab_set(sd.pages)
    print(f"streamed trustworthy A/B alpha pages: {len(sset)}")

    missing = bset - sset          # batch found, stream missed -> FAILURE
    extra = sset - bset            # stream found, batch missed (overlap can recover edges)
    print(f"overlap: {len(bset & sset)}  missing(batch-only): {len(missing)}  "
          f"extra(stream-only): {len(extra)}")
    if missing:
        print("--- MISSING (batch decoded these, stream did not) ---")
        for slot, typ, body in sorted(missing):
            print(f"  slot={slot} {typ}: {body.decode('ascii','replace')!r}")
    if extra:
        print("--- EXTRA (stream-only, from window overlap) ---")
        for slot, typ, body in sorted(extra):
            print(f"  slot={slot} {typ}: {body.decode('ascii','replace')!r}")
    ok = not missing
    print(f"RESULT: {'PASS' if ok else 'FAIL'} (stream is a {'superset' if ok else 'NON-superset'} of batch)")
    return ok


if __name__ == "__main__":
    cf = sys.argv[1] if len(sys.argv) > 1 else "/tmp/flex_ab/iq_929612500_250k.cfile"
    sys.exit(0 if validate(cf) else 1)
