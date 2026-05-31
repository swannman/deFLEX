#!/usr/bin/env python3
"""FLEX batch decoder (CLI).

Runs the FLEX decoder (flex_core) over a recorded .cfile capture and grades every
page by confidence tier (A=verified, B=FEC-corrected, C=suspect, D=failed) from
how hard the FEC worked and whether the body reads as English. The offline entry
point; the live receiver imports flex_core directly. See docs/decoder.md for the
flags and the algorithm.

The full-strength decode chain is always on: 4-FSK matched-filter bank, correlator
acquisition, comb-synthesized frames, timing/parity sweep, and Chase-II soft FEC.
By default only alpha pages (ALN/SPN) are kept and graded; pass --all to emit every
page type without the English-readability gate.

Usage: flex_batch.py <cfile> [--all] [--carrier=HZ] [--lpf=HZ]
                    [--samp-rate=HZ] [--center=MHZ] [--inv] [--diag]
"""
import os
import sys
import numpy as np

# flex_core (+ flex_numba, paging_core) live in core/ at the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                    # flat-install fallback
sys.path.insert(0, os.path.join(_HERE, "core"))
from flex_core import *          # noqa: F401,F403 -- decode functions + constants

# The offline entry point: read a capture, run the full decode chain (matched-filter
# bank + correlator acquisition + comb + sweep + Chase soft FEC), and grade every
# page A/B/C/D by how hard the FEC worked and whether its body reads as English. It
# never hard-drops a page -- it labels confidence and lets the caller choose. The
# live receivers import the same building blocks from flex_core directly.
def main():
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        sys.exit("usage: flex_batch.py <cfile> [--all] [options]")
    cfile = sys.argv[1]
    diag = "--diag" in sys.argv

    mflen = SPB
    lpf = 12000.0
    carrier = 0.0          # channel-select offset (Hz) within the +-fs/2 capture band
    in_rate = None         # input file sample rate (None -> assume SAMP=250k)
    center_mhz = 929.6125  # display-only: capture center freq for labelling carriers
    for a in sys.argv:
        if a.startswith("--mflen="):
            mflen = int(a.split("=", 1)[1])
        if a.startswith("--lpf="):
            lpf = float(a.split("=", 1)[1])
        if a.startswith("--carrier="):
            carrier = float(a.split("=", 1)[1])
        if a.startswith("--samp-rate="):
            in_rate = float(a.split("=", 1)[1])
        if a.startswith("--center="):
            center_mhz = float(a.split("=", 1)[1])
    nocfo = "--nocfo" in sys.argv               # disable Tier 2 per-frame carrier null
    inv = "--inv" in sys.argv                   # reverse tone->level map (polarity)
    # --carrier shifts a neighbouring FLEX channel down to baseband (load_baseband
    # de-rotates by `carrier` Hz; the existing LPF then isolates it, pushing the
    # original centre carrier out of band). Lets us decode every channel that falls
    # within the +-125 kHz of the SAME capture IQ -- a true multi-carrier decode
    # with no new capture. The sync state machine auto-detects the channel's mode.
    demod, xb = front_end(cfile, cfo=carrier, mf=("--mf" in sys.argv), mflen=mflen,
                          lpf=lpf, return_baseband=True, in_rate=in_rate)
    demod = demod - np.median(demod)            # global DC/CFO removal
    corr_grid = "--corr-peaks" not in sys.argv   # comb all grid slots (vs only detected peaks)
    corr_off = 1517.0
    alpha_only = "--all" not in sys.argv        # optimize for ALN/SPN: drop NUM/NNM (--all keeps every type)
    # alpha-only lets us comb aggressively: garbage numeric pages (which always
    # look 'printable') would otherwise flood the tiers, but we discard them, and
    # garbage alpha is self-evidently non-English (caught by the printable gate).
    corr_thr = -1.0 if alpha_only else 0.35
    for a in sys.argv:
        if a.startswith("--corr-off="):
            corr_off = float(a.split("=", 1)[1])
        if a.startswith("--corr-thr="):
            corr_thr = float(a.split("=", 1)[1])
    frames = corr_frames(demod, p0_offset=corr_off, grid=corr_grid,
                         include_thr=corr_thr)
    n_anchor = len(frames)
    frames = add_comb_frames(frames, len(demod), verbose=True)

    _syms_list = [f["syms"] for f in frames if f.get("syms") is not None]
    all_syms = np.concatenate(_syms_list) if _syms_list else np.array([])
    if diag and len(all_syms):
        v = all_syms - np.median(all_syms)
        lo, hi = two_means(np.abs(v))
        print(f"[diag] frames={len(frames)} datasyms={len(all_syms)}")
        print(f"[diag] |v| inner~{lo:.3f} outer~{hi:.3f} -> adaptive thr~{(lo+hi)/2:.3f} (fixed=2.0)")
        for q in (1, 5, 25, 50, 75, 95, 99):
            print(f"[diag]   pctl {q:2d}: v={np.percentile(v,q):+.3f}")
        hist, edges = np.histogram(v, bins=40, range=(-6, 6))
        peak = hist.max()
        for h, e in zip(hist, edges):
            bar = "#" * int(60 * h / peak)
            print(f"[hist] {e:+5.1f} | {bar}")

    xp = np.arange(len(demod), dtype=float)

    def sample_at(positions):
        return np.interp(positions, xp, demod)

    def decode_mf(pos16, baud, levels, chase=False):
        # Tier 1/2 path: 4-FSK matched-filter bank on the complex baseband at the
        # given symbol centers (16 kHz units), with per-frame carrier null (Tier 2).
        cfo = 0.0 if nocfo else est_cfo(xb, pos16[0] * SAMP / FS,
                                        pos16[-1] * SAMP / FS)
        metric = mf_bank_mag(xb, pos16, baud, cfo=cfo, inv=inv)
        bit_a, bit_b, mag_a, mag_b = mfbank_softbits(metric)
        phases, mags = demux_phases(bit_a, bit_b, mag_a, mag_b, baud)
        return decode_phases(phases, mags, chase=chase)

    sweep = True
    frac = "--frac" in sys.argv          # sub-sample (fractional) phase search
    soft = True                          # Chase-II soft-decision FEC on hard fails
    pages = []
    bch_fail = bch_corr = bch_chase = 0
    off_hist = {}
    for f in frames:
        spb = f["spb"]; p0 = f["p0"]; nsyms = f["nsyms"]
        if not sweep:
            offs = [0.0]
        elif frac:
            offs = list(np.arange(-spb / 2, spb / 2, 0.5))
        else:
            offs = [float(o) for o in range(-(spb // 2), spb - spb // 2)]
        # symbol-parity search: 3200-baud A/C de-mux can land on the wrong
        # symbol parity (whole frame fails). Try starting on symbol 0 or 1.
        pars = [0, 1] if f["baud"] == 3200 else [0]
        best = None
        # sweep uses fast HARD decode to pick timing/parity; soft Chase is run
        # once on the winner (it only helps words, never changes which sync is best)
        for par in pars:
            for off in offs:
                base = p0 + off + par * spb
                n = nsyms - par
                pos = base + np.arange(n) * spb
                if pos[0] < 0 or pos[-1] >= len(demod) - 1:
                    continue
                wordsets, confsets, nf, nc, _ = decode_mf(pos, f["baud"], f["levels"])
                if best is None or nf < best[1]:
                    best = (off, nf, nc, wordsets, confsets, par, pos)
        if best is None:
            continue
        off, nf, nc, wordsets, confsets, par, bpos = best
        # Sweep above used the fast hard decode to pick timing/parity. Refine the
        # winner once with Chase soft-decision FEC.
        wordsets, confsets, nf, nc, nch = decode_mf(
            bpos, f["baud"], f["levels"], chase=soft)
        bch_chase += nch
        key = (round(off, 1), par)
        off_hist[key] = off_hist.get(key, 0) + 1
        bch_fail += nf; bch_corr += nc
        if "--pf" in sys.argv:
            bsyms = sample_at(bpos)
            c = kmeans4(bsyms)
            gaps = np.diff(c)
            # within-level spread proxy: residual after assigning to nearest center
            d = np.abs(bsyms[:, None] - c[None, :]); lab = d.argmin(1)
            resid = np.sqrt(np.mean((bsyms - c[lab]) ** 2))
            merit = float(np.min(gaps) / (resid + 1e-9))
            print(f"[pf] off={off:+.1f} par={par} fail={nf:3d} "
                  f"c=[{c[0]:+.1f},{c[1]:+.1f},{c[2]:+.1f},{c[3]:+.1f}] "
                  f"resid={resid:.2f} merit={merit:.2f}")
        for words, cf in zip(wordsets, confsets):
            if words is not None:
                pages.extend(parse_frame(words, cf))
    if sweep:
        print(f"[sweep] chosen (offset,parity) histogram: {dict(sorted(off_hist.items()))}")

    # confidence-graded classification (NO hard BCH gate): emit every page, score it.
    # Tier A verified : all words clean syndrome (status 0) AND min-margin healthy.
    # Tier B fec      : BCH-corrected or Chase-recovered words, no uncorrectable word.
    # Tier C suspect  : passed BCH but min-margin < 0.5x local median (likely miscorrect),
    #                   or no soft info available.
    # Tier D failed   : >=1 uncorrectable word in the page (corrupted body).
    MARGIN_OK = 0.5
    ALPHA_EN_OK = 0.60   # english_score floor for a trustworthy --alpha page
    empty = 0
    tiers = {"A": 0, "B": 0, "C": 0, "D": 0}
    by_tier_type = {"A": {}, "B": {}, "C": {}, "D": {}}
    samples = []

    def tier_of(pc, pr, en):
        # RF/FEC confidence first, then a body-text sanity check: a page the RF
        # layer is sure of but whose body isn't mostly printable is NOT a
        # trustworthy *message* (binary/junk or a likely miscorrection), so it
        # can't sit in A/B regardless of margin.
        if pc is None:
            t = "C"
        elif pc["failed"] > 0:
            t = "D"
        elif pc["margin"] == pc["margin"] and pc["margin"] < MARGIN_OK:
            t = "C"
        elif pc["corrected"] or pc["chase"]:
            t = "B"
        else:
            t = "A"
        if t in ("A", "B") and pr < 0.85:
            t = "C"
        # --alpha: the comb is aggressive, so demand the body actually reads as
        # English. This rejects Chase-manufactured garbage that slips past the
        # printable gate, and conversely lets a genuinely-English-but-garbled
        # message keep its FEC tier.
        if alpha_only and en < ALPHA_EN_OK and t in ("A", "B"):
            t = "C"
        return t

    for cap, typ, body, pc in pages:
        if alpha_only and typ not in ("ALN", "SPN"):   # keep only alpha pages
            continue
        if len(body) == 0:
            empty += 1; continue
        pr = sum(1 for c in body if 32 <= c < 127) / len(body)
        en = english_score(body) if alpha_only else 1.0
        t = tier_of(pc, pr, en)
        tiers[t] += 1
        by_tier_type[t][typ] = by_tier_type[t].get(typ, 0) + 1
        if t in ("A", "B") or len(samples) < 36:
            mg = pc["margin"] if pc else float("nan")
            samples.append((t, typ, mg, pr, en, body.decode("ascii", "replace")))

    fe = "mfbank" + ("" if nocfo else "+cfo") + ("+inv" if inv else "")
    acq = "corr-grid" if corr_grid else "corr-peaks"
    fe = f"{fe}/{acq}"
    ctxt = f" carrier={center_mhz + carrier/1e6:.4f}MHz (offset{carrier/1e3:+.0f}kHz)" if carrier else ""
    print(f"=== front-end={fe} soft={soft} comb=True{ctxt} ===")
    print(f"frames        : {len(frames)} (anchors synced={n_anchor}, comb-synth={len(frames)-n_anchor})")
    print(f"BCH corrected : {bch_corr}   Chase-recovered: {bch_chase}   BCH FAILED: {bch_fail}")
    nonempty = sum(tiers.values())
    print(f"pages emitted : {len(pages)}  (empty={empty}, nonempty={nonempty})")
    print(f"confidence    : A_verified={tiers['A']}  B_fec={tiers['B']}  "
          f"C_suspect={tiers['C']}  D_failed={tiers['D']}   (trustworthy A+B={tiers['A']+tiers['B']})")
    print(f"by type/tier  : A={by_tier_type['A']} B={by_tier_type['B']} "
          f"C={by_tier_type['C']} D={by_tier_type['D']}")
    print("--- samples [tier margin printable% english] ---")
    for t, typ, mg, pr, en, s in sorted(samples, key=lambda x: x[0]):
        mtxt = f"{mg:4.2f}" if mg == mg else " nan"
        entxt = f" en={en:.2f}" if alpha_only else ""
        print(f"  [{t} m={mtxt} pr={pr:.2f}{entxt}] {typ}: {s!r}")

if __name__ == "__main__":
    main()
