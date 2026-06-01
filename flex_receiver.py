#!/usr/bin/env python3
"""Live FLEX receiver (single carrier).

Decodes ONE FLEX carrier: its own SDR tuned to the carrier, channelized to 250 kS/s,
fed to a FLEXStream decoder. For multiple carriers -- or mixed FLEX/POCSAG on one
SDR, including a channel that carries both -- use start_receiver.py instead.

Modes:
  --live --freq MHZ [--driver ...] [--log DIR]   own SDR, one carrier
  --file CFILE [--in-rate HZ]                     replay a recorded capture through
                                                  FLEXStream (parity vs flex_batch)

The --file path uses only numpy/scipy (no GNU Radio); --live needs GNU Radio.
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "core"))
sys.path.insert(0, _HERE)
import flex_core as F
import receiver_core as RC


def run_file(cfile, in_rate, log_path=None, inv=False):
    """Replay a capture through FLEXStream exactly as the live path would, then
    report the trustworthy A/B page count (parity check vs flex_batch)."""
    lf = open(log_path, "a", buffering=1) if log_path else None
    carrier = 0
    sd = F.FLEXStream(cfg=dict(in_rate=in_rate, inv=inv),
                      window_frames=RC.WINDOW_FR, advance_frames=RC.ADVANCE_FR,
                      on_page=RC.make_flex_on_page(carrier, lf))
    x = np.fromfile(cfile, dtype=np.complex64)
    chunk = int(2.0 * in_rate)
    for i in range(0, len(x), chunk):
        sd.feed(x[i:i + chunk])
    sd.flush()
    print(f"flex_receiver file: {len(sd.pages)} trustworthy A/B pages "
          f"({len(x)/in_rate:.1f}s @ {in_rate/1e3:.0f}k)", file=sys.stderr)
    return len(sd.pages)


def main():
    ap = argparse.ArgumentParser(
        description="Single-carrier live FLEX receiver. For multi-carrier or mixed "
                    "FLEX/POCSAG, use start_receiver.py.")
    ap.add_argument("--file")
    ap.add_argument("--in-rate", type=float, default=float(F.SAMP))
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--freq", type=float,
                    help="FLEX carrier frequency in MHz (required for --live)")
    ap.add_argument("--driver", default="sdrplay",
                    help="SoapySDR driver for --live (sdrplay|rtlsdr|airspy)")
    ap.add_argument("--gain",
                    help="SDR gain for --live: 'agc', overall dB, or ELEM=VAL pairs "
                         "(sdrplay: RFGR,IFGR; airspy: LNA,MIX,VGA). Default: per-driver.")
    ap.add_argument("--antenna", help="SDR antenna port for --live")
    ap.add_argument("--ppm", type=float, help="frequency correction in ppm")
    ap.add_argument("--bandwidth", type=float, help="analog filter bandwidth in Hz")
    ap.add_argument("--inv", action="store_true",
                    help="invert tone->level polarity (spectrally-mirrored capture)")
    ap.add_argument("--log", help=f"log directory for --live (default {RC.LOG_DIR})")
    args = ap.parse_args()
    if args.file:
        run_file(args.file, args.in_rate, inv=args.inv)
    elif args.live:
        if args.freq is None:
            ap.error("--live requires --freq (carrier frequency in MHz)")
        freq = int(round(args.freq * 1e6))
        import receiver_sdr       # GNU Radio; imported only for the live path
        receiver_sdr.run_live([freq], [], freq, driver=args.driver,
                              log_dir=args.log or RC.LOG_DIR, inv=args.inv,
                              gain=RC.parse_gain(args.gain), antenna=args.antenna,
                              ppm=args.ppm, bandwidth=args.bandwidth)
    else:
        ap.error("specify --file CFILE or --live")


if __name__ == "__main__":
    main()
