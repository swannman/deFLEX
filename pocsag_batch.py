#!/usr/bin/env python3
"""POCSAG batch decoder (CLI).

Runs the POCSAG decoder (pocsag_core) over a recorded .cfile capture and prints
decoded alpha pages with their english_score. The offline counterpart to the live
pocsag_receiver. See docs/decoder.md for details.

Usage: pocsag_batch.py <cfile> [--in-rate HZ] [--min-en F]
"""
import os
import sys
import numpy as np

# pocsag_core + paging_core live in core/ at the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                    # flat-install fallback
sys.path.insert(0, os.path.join(_HERE, "core"))
from pocsag_core import *        # noqa: F401,F403 -- decode functions + constants
import paging_core as PC

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cfile")
    ap.add_argument("--in-rate", type=float, default=IN_RATE_DEFAULT)
    ap.add_argument("--min-en", type=float, default=0.0,
                    help="english_score floor to print a page (0 = print all)")
    args = ap.parse_args()
    x = np.fromfile(args.cfile, dtype=np.complex64)
    pages = decode_baseband(x, in_rate=args.in_rate)
    shown = 0
    for addr, func, text, _pos in pages:
        en = PC.english_score(text)
        if en < args.min_en:
            continue
        shown += 1
        body = text.decode("ascii", "replace")
        print(f"cap={addr:>8} f={func} en={en:.2f}  {body!r}")
    print(f"\n{len(pages)} alpha pages decoded, {shown} shown "
          f"(en>={args.min_en})", file=sys.stderr)


if __name__ == "__main__":
    main()
