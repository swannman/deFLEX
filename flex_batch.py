#!/usr/bin/env python3
"""FLEX batch decoder (CLI).

Thin wrapper over flex_core.decode_baseband(): runs the FLEX decoder over a
recorded .cfile capture and grades every page by confidence tier (A=verified,
B=FEC-corrected, C=suspect, D=failed). The offline counterpart to the live
receiver, which imports flex_core directly. See docs/decoder.md for the algorithm.

The full-strength decode chain is always on: 4-FSK matched-filter bank, correlator
acquisition, comb-synthesized frames, timing/parity sweep, and Chase-II soft FEC.
By default only alpha pages (ALN/SPN) are kept and graded; pass --all to emit every
page type without the English-readability gate.

Each trustworthy (A/B) page prints one per line on stdout; the FEC/tier report goes
to stderr (so `flex_batch.py … | grep` sees just the pages, like pocsag_batch).

Usage: flex_batch.py <cfile> [--in-rate HZ] [--carrier HZ] [--all] [--inv]
"""
import os
import sys
import numpy as np

# flex_core (+ flex_numba, paging_core) live in core/ at the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                    # flat-install fallback
sys.path.insert(0, os.path.join(_HERE, "core"))
from flex_core import decode_baseband


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="FLEX batch decoder: decode a recorded .cfile capture and grade "
                    "every page by confidence tier (A/B/C/D). The full-strength chain "
                    "is always on; by default only alpha (ALN/SPN) pages are kept and "
                    "graded against the english_score gate. Trustworthy (A/B) pages "
                    "print to stdout; the tier/FEC report goes to stderr.")
    ap.add_argument("cfile", help="raw complex64 IQ capture")
    ap.add_argument("--in-rate", type=float, default=None,
                    help="capture sample rate in Hz (default: 250000 = decoder SAMP)")
    ap.add_argument("--all", action="store_true",
                    help="emit every page type (NUM/NNM too) and skip the english_score gate")
    ap.add_argument("--carrier", type=float, default=0.0,
                    help="channel-select offset in Hz within the +-fs/2 capture band")
    ap.add_argument("--inv", action="store_true",
                    help="invert the tone->level polarity map (spectrally-mirrored capture)")
    args = ap.parse_args()

    cfg = dict(carrier=args.carrier, inv=args.inv, alpha_only=not args.all,
               sweep_half=None)        # None -> full timing sweep (the batch default)
    if args.in_rate is not None:
        cfg["in_rate"] = args.in_rate
    x = np.fromfile(args.cfile, dtype=np.complex64)
    pages, st = decode_baseband(x, cfg)

    fe = "mfbank+cfo" + ("+inv" if args.inv else "")
    ctxt = f" carrier offset {args.carrier/1e3:+.0f} kHz" if args.carrier else ""
    t = st["tiers"]
    print(f"=== front-end={fe}/corr-grid soft=True comb=True{ctxt} ===\n"
          f"frames        : {st['n_frames']} (anchors synced={st['n_anchor']}, "
          f"comb-synth={st['n_frames']-st['n_anchor']})\n"
          f"BCH corrected : {st['bch_corr']}   Chase-recovered: {st['bch_chase']}   "
          f"BCH FAILED: {st['bch_fail']}\n"
          f"[sweep] (offset,parity) histogram: {dict(sorted(st['off_hist'].items()))}\n"
          f"pages emitted : {st['raw_pages']}  (empty={st['empty']}, "
          f"nonempty={sum(t.values())})\n"
          f"confidence    : A_verified={t['A']}  B_fec={t['B']}  C_suspect={t['C']}  "
          f"D_failed={t['D']}   (trustworthy A+B={t['A']+t['B']})\n"
          f"by type/tier  : A={st['by_tier_type']['A']} B={st['by_tier_type']['B']} "
          f"C={st['by_tier_type']['C']} D={st['by_tier_type']['D']}", file=sys.stderr)

    for cap, typ, body, tier, pr, en in pages:
        if tier in ("A", "B"):
            entxt = f" en={en:.2f}" if not args.all else ""
            print(f"cap={cap:>9} {typ} {tier}{entxt}  {body.decode('ascii', 'replace')!r}")


if __name__ == "__main__":
    main()
