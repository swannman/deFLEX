#!/usr/bin/env python3
"""Dedicated POCSAG alpha-paging receiver for 152.0075 MHz (SNO911 fire/EMS).

Sibling of flex_inmem_mp.py, but POCSAG on VHF lives ~780 MHz from the 929-932
FLEX band and so CANNOT share the RSPdx tuner pass -- it needs its OWN SDR. The
signal is strong (decodes even on the 915 MHz antenna), so a cheap RTL-SDR is
plenty. One carrier means one decode thread behind a drop-oldest ring is enough
(no GIL pressure to design around, unlike the 5-carrier FLEX path).

It reuses the validated pocsagdec core (POCSAGStream) and emits the SAME log line
the FLEX path writes, so the existing flex-tail web viewer renders POCSAG pages
with no client change:

    <ts> FLEX|<carrier>|0|A|<capcode>|ALN|<body>

The capcode goes in the field the viewer dedups on (FLEX logs hardcode "0"
there; POCSAG has a real RIC, so we use it -- retransmits of the same page on
the same capcode collapse in the viewer's (capcode, body) de-dup).

Modes:
  --file CFILE [--in-rate HZ]   offline parity check against a capture
  --live [--log DIR]            live RTL-SDR @ 152.0075 MHz
"""
import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# pocsagdec (batch core) + paging_core sit at the repo root; pocsagdec_stream
# sits beside this file in online/. The flat /usr/local/bin install co-locates
# everything -- cover both layouts.
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import pocsagdec as P
import pocsagdec_stream as PS   # POCSAGStream (streaming wrapper)

FREQ = 152_007_500          # SNO911 fire/EMS alphanumeric dispatch
CARRIER = 152007500         # log carrier id (also the log filename stem)
RX_RATE = 250000            # RTL-SDR low-rate regime; POCSAGStream resamples to 9600
EN_FLOOR = 0.45             # english_score floor inside P.is_clean_alpha. 0.45
                            # keeps every ground-truth dispatch (lowest real page
                            # = 0.49 in the 600s A/B); the gate also drops sub-12
                            # char and control-char FEC fragments.
LOG_DIR = "/var/log/flex"
QUEUE_MAXCHUNKS = 256       # ~one window-decode of input @ 250k; drop-oldest on full


def make_on_page(log_path):
    """Return an on_page(rec) callback that gates by english_score and appends
    the viewer-compatible FLEX| log line (or prints it if log_path is None)."""
    lf = open(log_path, "a", buffering=1) if log_path else None

    def on_page(rec):
        addr, func, text = rec
        if not P.is_clean_alpha(text, EN_FLOOR):
            return
        body = (text.decode("ascii", "replace")
                .replace("\\", "\\\\").replace("\n", "\\n")
                .replace("\r", "\\r").replace("\t", "\\t"))
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        line = "%s FLEX|%d|0|A|%d|ALN|%s" % (ts, CARRIER, addr, body)
        if lf is not None:
            lf.write(line + "\n")
        else:
            print(line)

    return on_page


def run_file(cfile, in_rate, log_path=None):
    """Feed a capture through POCSAGStream exactly as the live path would."""
    on_page = make_on_page(log_path)
    x = np.fromfile(cfile, dtype=np.complex64)
    ps = PS.POCSAGStream(in_rate=in_rate, on_page=on_page)
    chunk = int(2.0 * in_rate)
    for i in range(0, len(x), chunk):
        ps.feed(x[i:i + chunk])
    ps.flush()
    print(f"pocsag_receiver file: {len(ps.pages)} pages "
          f"({len(x)/in_rate:.1f}s @ {in_rate/1e3:.0f}k, en>={EN_FLOOR} logged)",
          file=sys.stderr)
    return len(ps.pages)


# --- live path (RTL-SDR via SoapySDR + GNU Radio) ---------------------------
# Imported lazily so file mode runs on a box without GNU Radio installed.

def run_live(log_dir, driver="rtlsdr"):
    import queue
    import threading
    from gnuradio import gr, blocks, soapy

    os.makedirs(log_dir, exist_ok=True)
    log_path = f"{log_dir}/{CARRIER}.pocsag.log"
    q = queue.Queue(maxsize=QUEUE_MAXCHUNKS)
    dropped = {"n": 0}

    class PocsagSink(gr.sync_block):
        """complex64 -> queue (drop-oldest on overflow), same contract as the
        FLEX RingSink: a slow decode never back-pressures the SDR."""

        def __init__(self):
            gr.sync_block.__init__(self, name="pocsag_sink",
                                   in_sig=[np.complex64], out_sig=None)

        def work(self, input_items, output_items):
            x = input_items[0]
            chunk = np.asarray(x, dtype=np.complex64).copy()
            try:
                q.put_nowait(chunk)
            except queue.Full:
                try:
                    q.get_nowait()
                    dropped["n"] += 1
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(chunk)
                except queue.Full:
                    pass
            return len(x)

    ps = PS.POCSAGStream(in_rate=RX_RATE, on_page=make_on_page(log_path))

    def worker():
        while True:
            try:
                chunk = q.get(timeout=0.5)
            except queue.Empty:
                continue
            if chunk is None:
                ps.flush()
                return
            chunks = [chunk]
            try:
                while True:
                    c = q.get_nowait()
                    if c is None:
                        ps.feed(np.concatenate(chunks))
                        ps.flush()
                        return
                    chunks.append(c)
            except queue.Empty:
                pass
            ps.feed(np.concatenate(chunks))

    class LiveGraph(gr.top_block):
        def __init__(self):
            gr.top_block.__init__(self, "pocsag_receiver_live")
            self.src = soapy.source(f"driver={driver}", "fc32", 1, "", "", [""], [""])
            self.src.set_sample_rate(0, RX_RATE)   # 250k: native on RTL & RSPdx
            self.src.set_frequency(0, FREQ)
            self.src.set_gain_mode(0, True)        # AGC; signal is strong
            self.sink = PocsagSink()
            self.connect(self.src, self.sink)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    tb = LiveGraph()
    print(f"pocsag_receiver LIVE: {driver} @ {FREQ/1e6:.4f} MHz, {RX_RATE} S/s -> "
          f"POCSAGStream -> {log_path}", file=sys.stderr, flush=True)
    tb.start()
    try:
        while True:
            time.sleep(30)
            print(f"[pocsag] pages={len(ps.pages)} dropped={dropped['n']}",
                  file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        tb.stop(); tb.wait()
        q.put(None)
        t.join(timeout=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--in-rate", type=float, default=float(RX_RATE))
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--driver", default="rtlsdr",
                    help="SoapySDR driver for --live (rtlsdr|sdrplay). "
                         "Use sdrplay to time-share the FLEX RSPdx.")
    ap.add_argument("--log", help="log dir (live) or file (file mode); default stdout")
    args = ap.parse_args()
    if args.file:
        run_file(args.file, args.in_rate, log_path=args.log)
    elif args.live:
        run_live(args.log or LOG_DIR, driver=args.driver)
    else:
        ap.error("specify --file CFILE or --live")


if __name__ == "__main__":
    main()
