#!/usr/bin/env python3
"""Live paging receiver harness: one SDR, many FLEX and POCSAG carriers.

Tunes a single SoapySDR device to a center frequency, channelizes the captured
band into the requested carriers, and runs the appropriate decoder on each --
FLEXStream for FLEX carriers, POCSAGStream for POCSAG carriers. A carrier given
in BOTH lists (a channel that carries FLEX and POCSAG) is decoded by both.

All carriers must fit inside one SDR capture window (center +- samp_rate/2); it is
up to the caller to pick frequencies that do. FLEX (~900 MHz) and POCSAG VHF
(~152 MHz) are different bands, so they need separate harness instances/SDRs.

All frequencies are in MHz. A frequency listed under BOTH --flex and --pocsag is
the dual-protocol case (decoded by both). In the example below the two lists share
one frequency, which therefore gets both decoders:
  start_receiver.py --flex 930.5,931.5 --pocsag 931.5,931.8 \\
                    [--center MHZ] [--samp-rate HZ] [--driver sdrplay] [--log DIR]
  start_receiver.py --flex ... --pocsag ... --dry-run   # print the plan, no SDR
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "core"))
sys.path.insert(0, _HERE)
import receiver_core as R


def _mhz_list(spec):
    """'930.5,931.5' -> [930500000, 931500000] (Hz)."""
    out = []
    for tok in (spec or "").split(","):
        tok = tok.strip()
        if tok:
            out.append(int(round(float(tok) * 1e6)))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Live paging receiver: one SDR, many FLEX and POCSAG carriers. "
                    "Channelizes the captured band and runs FLEXStream / POCSAGStream "
                    "per carrier (both, for a dual-protocol channel).")
    ap.add_argument("--flex", default="",
                    help="comma-separated FLEX carrier frequencies in MHz")
    ap.add_argument("--pocsag", default="",
                    help="comma-separated POCSAG carrier frequencies in MHz")
    ap.add_argument("--center", type=float,
                    help="SDR tuner center frequency in MHz "
                         "(default: midpoint of all carriers)")
    ap.add_argument("--samp-rate", type=float, default=R.SAMP_RATE,
                    help=f"SDR capture sample rate in Hz (default {R.SAMP_RATE})")
    ap.add_argument("--driver", default="sdrplay",
                    help="SoapySDR driver (sdrplay|rtlsdr|airspy)")
    ap.add_argument("--inv", action="store_true",
                    help="invert FLEX tone polarity for a spectrally-mirrored capture "
                         "(POCSAG auto-detects polarity, so this only affects FLEX)")
    ap.add_argument("--log", help=f"log directory (default {R.LOG_DIR})")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the channelization plan and exit (no SDR access)")
    args = ap.parse_args()

    flex = _mhz_list(args.flex)
    pocsag = _mhz_list(args.pocsag)
    carriers = sorted(set(flex) | set(pocsag))
    if not carriers:
        ap.error("specify at least one carrier via --flex and/or --pocsag")
    center = (int(round(args.center * 1e6)) if args.center is not None
              else (min(carriers) + max(carriers)) // 2)

    if args.dry_run:
        print(R.plan_text(flex, pocsag, center, args.samp_rate, args.driver))
        return
    import receiver_sdr           # GNU Radio; imported only for the live path
    receiver_sdr.run_live(flex, pocsag, center, samp_rate=args.samp_rate,
                          driver=args.driver, log_dir=args.log or R.LOG_DIR,
                          inv=args.inv)


if __name__ == "__main__":
    main()
