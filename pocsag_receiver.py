#!/usr/bin/env python3
"""Live POCSAG receiver (single carrier).

Decodes ONE POCSAG carrier: its own SDR tuned to the carrier, channelized to
250 kS/s, fed to a POCSAGStream decoder. For multiple carriers -- or mixed
FLEX/POCSAG on one SDR, including a channel that carries both -- use
start_receiver.py instead.

It emits the SAME log-line shape the FLEX path writes, so the web viewer renders
POCSAG pages with no change:
    <ts> FLEX|<carrier>|0|A|<capcode>|ALN|<body>
The capcode (real RIC) sits in the field the viewer dedups on, so retransmits of
the same page collapse in the viewer's (capcode, body) de-dup.

Modes:
  --live --freq MHZ [--driver rtlsdr|sdrplay] [--log DIR]   own SDR, one carrier
  --file CFILE [--freq MHZ] [--in-rate HZ]                  replay a capture

The --file path uses only numpy/scipy (no GNU Radio); --live needs GNU Radio.
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "core"))
sys.path.insert(0, _HERE)
import pocsag_core as P
import receiver_core as RC


def run_file(cfile, in_rate, carrier, log_path=None):
    """Feed a capture through POCSAGStream exactly as the live path would."""
    lf = open(log_path, "a", buffering=1) if log_path else None
    ps = P.POCSAGStream(in_rate=in_rate, on_page=RC.make_pocsag_on_page(carrier, lf))
    x = np.fromfile(cfile, dtype=np.complex64)
    chunk = int(2.0 * in_rate)
    for i in range(0, len(x), chunk):
        ps.feed(x[i:i + chunk])
    ps.flush()
    print(f"pocsag_receiver file: {len(ps.pages)} pages "
          f"({len(x)/in_rate:.1f}s @ {in_rate/1e3:.0f}k, en>={RC.EN_FLOOR} logged)",
          file=sys.stderr)
    return len(ps.pages)


def main():
    ap = argparse.ArgumentParser(
        description="Single-carrier live POCSAG receiver. For multi-carrier or mixed "
                    "FLEX/POCSAG, use start_receiver.py.")
    ap.add_argument("--file")
    ap.add_argument("--in-rate", type=float, default=float(RC.CHAN_RATE))
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--freq", type=float,
                    help="POCSAG carrier frequency in MHz. Required for --live; "
                         "also tags --file log lines.")
    ap.add_argument("--driver", default="rtlsdr",
                    help="SoapySDR driver for --live (rtlsdr|sdrplay)")
    ap.add_argument("--log", help=f"log directory for --live (default {RC.LOG_DIR})")
    args = ap.parse_args()
    if args.file:
        carrier = int(round(args.freq * 1e6)) if args.freq is not None else 0
        run_file(args.file, args.in_rate, carrier, log_path=args.log)
    elif args.live:
        if args.freq is None:
            ap.error("--live requires --freq (carrier frequency in MHz)")
        freq = int(round(args.freq * 1e6))
        import receiver_sdr       # GNU Radio; imported only for the live path
        receiver_sdr.run_live([], [freq], freq, driver=args.driver,
                              log_dir=args.log or RC.LOG_DIR)
    else:
        ap.error("specify --file CFILE or --live")


if __name__ == "__main__":
    main()
