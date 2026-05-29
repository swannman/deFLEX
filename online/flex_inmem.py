#!/usr/bin/env python3
"""In-memory FLEX decoder: SDR -> per-carrier freq-xlate -> ring buffer ->
StreamDecoder worker thread. No intermediate file between the SDR and decode.

This is the replacement for the multimon FIFO path. Each carrier owns:
  * a freq_xlating_fir_filter (channel-select + decimate to IN_RATE),
  * a RingSink gr.sync_block that hands samples to a bounded queue,
  * a CarrierWorker thread that pops the queue and drives a StreamDecoder.

The queue is drop-oldest on overflow, so a slow decode can NEVER back-pressure
the shared flowgraph (and thus the SDR). mf_bank_mag (the 58% hotspot) is
vectorized numpy that releases the GIL, so the per-carrier worker threads run
their matched-filter banks concurrently across cores under the service
CPUQuota -- no multiprocessing, no 8-core blowup.

Modes:
  --file CFILE [--in-rate HZ]   offline parity check: feed a recorded complex64
                                capture through the SAME ring->thread->decoder
                                path; print the trustworthy A/B page count so it
                                can be compared to the batch baseline.
  --live [FIFO_DIR]             live SoapySDR RSPdx source, all active carriers.
"""
import argparse
import os
import queue
import threading
import time
import sys

import numpy as np
from gnuradio import gr, blocks, filter, soapy
from gnuradio.filter import firdes

_HERE = os.path.dirname(os.path.abspath(__file__))
# flexdec_stream sits beside this file (online/); the shared core
# (flexdec.py / flexdec_numba.py) is at the repo root. In the flat
# /usr/local/bin install everything is co-located -- cover both layouts.
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import flexdec_stream as S
import flexdec as F

CENTER     = 930_762_500
SAMP_RATE  = 2_500_000                    # /10 -> 250000 == flexdec.SAMP: no resample_poly
DECIM      = 10
IN_RATE    = SAMP_RATE // DECIM          # 250000 Hz complex baseband (== F.SAMP)
WINDOW_FR  = 32                          # see flexdec_stream: >=16 reproduces batch
ADVANCE_FR = 28                          # 4-frame edge margin, ~0.54x realtime/carrier

# Active carriers: drop 929.6625 (test) + 931.0625 (POCSAG, non-FLEX).
# 929.5875 kept by request even though silent.
CARRIERS = [
    929_587_500,   # Spok          (silent/unmeasured) -- kept by request
    929_612_500,   # Spok          3200 baud / 4-lvl
    929_937_500,   # American Msg  1600 baud / 4-lvl
    931_212_500,   # Spok          3200 baud / 4-lvl
    931_937_500,   # SkyTel        1600 baud / 4-lvl
]

LOG_DIR = "/var/log/flex"


class RingSink(gr.sync_block):
    """Complex64 sink feeding a bounded queue (drop-oldest on overflow).
    work() always consumes its whole input and never blocks, so the upstream
    flowgraph is decoupled from decode latency."""

    def __init__(self, carrier, maxchunks=64):
        # maxchunks=0 -> unbounded (file parity mode: never drop, decode at its
        # own pace). Live mode keeps a finite bound so a slow decode drops the
        # oldest IQ instead of back-pressuring the SDR.
        gr.sync_block.__init__(self, name=f"ring_{carrier}",
                               in_sig=[np.complex64], out_sig=None)
        self.q = queue.Queue(maxsize=maxchunks)
        self.dropped_chunks = 0

    def work(self, input_items, output_items):
        x = input_items[0]
        chunk = np.asarray(x, dtype=np.complex64).copy()
        try:
            self.q.put_nowait(chunk)
        except queue.Full:
            try:
                self.q.get_nowait()          # drop oldest
                self.dropped_chunks += 1
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(chunk)
            except queue.Full:
                pass
        return len(x)


class CarrierWorker(threading.Thread):
    """Drains one RingSink's queue and feeds a StreamDecoder. Coalesces any
    backlog into a single feed() so the overlapped-window scheduler sees large
    contiguous blocks."""

    def __init__(self, carrier, ring, in_rate, on_page):
        super().__init__(daemon=True, name=f"worker_{carrier}")
        self.carrier = carrier
        self.ring = ring
        self.sd = S.StreamDecoder(
            cfg=dict(in_rate=in_rate),
            window_frames=WINDOW_FR, advance_frames=ADVANCE_FR,
            on_page=on_page)
        self._draining = False
        self.done = threading.Event()

    def run(self):
        while True:
            try:
                chunk = self.ring.q.get(timeout=0.5)
            except queue.Empty:
                if self._draining:
                    self.sd.flush()
                    self.done.set()
                    return
                continue
            chunks = [chunk]
            try:
                while True:
                    chunks.append(self.ring.q.get_nowait())
            except queue.Empty:
                pass
            self.sd.feed(np.concatenate(chunks))

    def finish(self):
        """Signal end-of-stream; worker drains remaining queue then flushes."""
        self._draining = True


# ---------------------------------------------------------------------------

class FileGraph(gr.top_block):
    """file_source -> RingSink for the single-carrier parity check."""

    def __init__(self, cfile):
        gr.top_block.__init__(self, "flex_inmem_file")
        self.src = blocks.file_source(gr.sizeof_gr_complex, cfile, False)
        self.ring = RingSink("file", maxchunks=0)   # unbounded: no drops in parity test
        self.connect(self.src, self.ring)


class LiveGraph(gr.top_block):
    """SoapySDR RSPdx -> per-carrier freq_xlate -> RingSink."""

    def __init__(self):
        gr.top_block.__init__(self, "flex_inmem_live")
        self.src = soapy.source("driver=sdrplay", "fc32", 1, "", "", [""], [""])
        self.src.set_sample_rate(0, SAMP_RATE)
        self.src.set_frequency(0, CENTER)
        self.src.set_gain_mode(0, False)
        self.src.set_gain(0, "RFGR", 4)
        self.src.set_gain(0, "IFGR", 25)
        self.src.set_antenna(0, "Antenna A")

        # Wide transition band on purpose: after decimate-by-DECIM the first
        # alias folds in at IN_RATE-9000 ~= 241 kHz, so a 60 kHz transition still
        # leaves huge guard. A 3 kHz transition forced ~2750 taps/carrier at the
        # full 2.5 MS/s input rate -> 5 carriers saturated the 3-core budget and
        # the SDR overran (continuous "O"), decoding nothing. ~137 taps fixes it.
        taps = firdes.low_pass(1.0, SAMP_RATE, 9000, 60000)
        self.rings = {}
        self._blocks = []
        for carrier in CARRIERS:
            offset = carrier - CENTER
            xlate = filter.freq_xlating_fir_filter_ccc(DECIM, taps, offset, SAMP_RATE)
            ring = RingSink(carrier)
            self.connect(self.src, xlate, ring)
            self.rings[carrier] = ring
            self._blocks += [xlate, ring]


# ---------------------------------------------------------------------------

def run_file(cfile, in_rate):
    pages = {"n": 0}
    seen_records = []

    def on_page(rec):
        pages["n"] += 1
        seen_records.append(rec)

    tb = FileGraph(cfile)
    worker = CarrierWorker("file", tb.ring, in_rate, on_page)
    worker.start()
    t0 = time.time()
    tb.start()
    tb.wait()                       # file_source EOF
    worker.finish()
    worker.done.wait()
    dt = time.time() - t0
    n_samp = os.path.getsize(cfile) // 8
    secs = n_samp / in_rate
    print(f"file={cfile}")
    print(f"in_rate={in_rate} samples={n_samp} ({secs:.1f}s of IQ)")
    print(f"window={WINDOW_FR}fr advance={ADVANCE_FR}fr")
    print(f"trustworthy A/B alpha pages (via in-memory ring->thread): {len(worker.sd.pages)}")
    print(f"dropped_chunks={tb.ring.dropped_chunks}")
    print(f"wall={dt:.1f}s  ({dt/secs:.2f}x realtime, single carrier)")
    return worker.sd.pages


def run_live(fifo_dir):
    os.makedirs(LOG_DIR, exist_ok=True)
    logs = {c: open(f"{LOG_DIR}/{c}.flexdec.log", "a", buffering=1) for c in CARRIERS}

    def make_on_page(carrier):
        lf = logs[carrier]

        def on_page(rec):
            slot, typ, body, tier, pr, en = rec
            # ALN only -- SPN decodes to garbage on these carriers (control chars,
            # no words), so it is dropped here and never written or fed to the viewer.
            if typ != "ALN":
                return
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            # Legacy multimon "FLEX|" line shape so flex_tail_server parses it with
            # NO server change: FLEX|<f1>|<f2>|<f3>|<capcode>|ALN|<body>, body = p[6:],
            # flag always K. capcode is fixed at 0 so the viewer's (capcode, body)
            # dedup collapses to body-only -> kills retransmits + same page on two
            # carriers. Escape CR/LF/TAB to literal so one page stays one log line.
            text = (body.decode("ascii", "replace")
                    .replace("\\", "\\\\").replace("\n", "\\n")
                    .replace("\r", "\\r").replace("\t", "\\t"))
            line = "%s FLEX|%d|%d|%s|0|ALN|%s" % (ts, carrier, slot, tier, text)
            lf.write(line + "\n")
        return on_page

    tb = LiveGraph()
    workers = {}
    for carrier in CARRIERS:
        w = CarrierWorker(carrier, tb.rings[carrier], IN_RATE, make_on_page(carrier))
        w.start()
        workers[carrier] = w

    print(f"flex_inmem LIVE: RSPdx @ {CENTER/1e6:.4f} MHz, {SAMP_RATE} S/s -> "
          f"{len(CARRIERS)} carriers @ {IN_RATE} Hz -> StreamDecoder threads "
          f"(window={WINDOW_FR} advance={ADVANCE_FR}) numba={F._HAVE_NUMBA} "
          f"resample={'OFF' if IN_RATE == F.SAMP else 'ON'}",
          file=sys.stderr, flush=True)
    tb.start()
    try:
        while True:
            time.sleep(30)
            drops = {c: tb.rings[c].dropped_chunks for c in CARRIERS}
            pgs = {c: len(workers[c].sd.pages) for c in CARRIERS}
            print(f"[inmem] pages={pgs} dropped={drops}", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        tb.stop(); tb.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--in-rate", type=float, default=float(F.SAMP))
    ap.add_argument("--live", action="store_true")
    ap.add_argument("fifo_dir", nargs="?", default="/tmp/flex")
    args = ap.parse_args()
    if args.file:
        run_file(args.file, args.in_rate)
    elif args.live:
        run_live(args.fifo_dir)
    else:
        ap.error("specify --file CFILE or --live")


if __name__ == "__main__":
    main()
