#!/usr/bin/env python3
"""Multiprocessing variant of flex_inmem: one OS PROCESS per carrier instead of
one thread, so each carrier's StreamDecoder runs under its own GIL and the
per-carrier decode work parallelizes across cores for real.

Why: the threaded flex_inmem.py serializes on the shared GIL. Five carriers'
worth of decode is only ~2.7 cores of actual work (profiled ~0.54x realtime per
carrier), but the Python glue around the njit/numpy hot kernels holds the GIL,
so the worker threads can't use >1 core concurrently and 4/5 carriers starve.
Giving each carrier its own process sidesteps the GIL entirely. GNU Radio's
flowgraph stays in the parent (native C++); complex64 baseband chunks cross to
the worker processes via an mp.Queue (drop-oldest on overflow -- same contract
as the threaded ring, so a slow decode never back-pressures the SDR).

Modes mirror flex_inmem.py:
  --file CFILE [--in-rate HZ]   single-carrier parity check (unbounded queue)
  --live [FIFO_DIR]             live RSPdx, all active carriers, one proc each
"""
import argparse
import os
import queue
import sys
import time
import multiprocessing as mp

import numpy as np
from gnuradio import gr, blocks, filter, soapy
from gnuradio.filter import firdes

_HERE = os.path.dirname(os.path.abspath(__file__))
# flexdec_stream sits beside this file (online/); shared core at the repo root.
# In the flat /usr/local/bin install everything is co-located -- cover both.
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import flexdec_stream as S
import flexdec as F
# Single source of truth for the SDR/carrier constants -- reuse the threaded
# module's values so the two paths can never drift in a benchmark.
from flex_inmem import (CENTER, SAMP_RATE, DECIM, IN_RATE,
                        WINDOW_FR, ADVANCE_FR, CARRIERS, LOG_DIR)


class RingSink(gr.sync_block):
    """complex64 sink -> mp.Queue (drop-oldest on overflow). Lives in the parent
    (flowgraph) process; the queue carries copied chunks to a worker PROCESS.
    The queue's bound is set by the caller via maxsize on the mp.Queue."""

    def __init__(self, carrier, q):
        gr.sync_block.__init__(self, name=f"ring_{carrier}",
                               in_sig=[np.complex64], out_sig=None)
        self.q = q
        self.dropped_chunks = 0

    def work(self, input_items, output_items):
        x = input_items[0]
        # Must copy: GNU Radio reuses the input buffer, but mp.Queue pickles
        # asynchronously in a feeder thread after put() returns.
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


def worker_proc(carrier, q, in_rate, window, advance, log_path, pages_val):
    """Runs in its own process: drains the mp.Queue, feeds a StreamDecoder.
    Coalesces any backlog into one feed() so the scheduler sees big blocks.
    `None` on the queue is the EOF sentinel (file mode) -> flush and exit.
    pages_val (mp.Value 'i') mirrors len(sd.pages) for the parent's stats."""
    lf = open(log_path, "a", buffering=1) if log_path else None

    def on_page(rec):
        if lf is None:
            return
        slot, typ, body, tier, pr, en = rec
        # ALN only -- SPN decodes to control-char garbage on these carriers.
        if typ != "ALN":
            return
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        text = (body.decode("ascii", "replace")
                .replace("\\", "\\\\").replace("\n", "\\n")
                .replace("\r", "\\r").replace("\t", "\\t"))
        lf.write("%s FLEX|%d|%d|%s|0|ALN|%s\n" % (ts, carrier, slot, tier, text))

    sd = S.StreamDecoder(cfg=dict(in_rate=in_rate),
                         window_frames=window, advance_frames=advance,
                         on_page=on_page)

    def finish():
        sd.flush()
        pages_val.value = len(sd.pages)

    while True:
        try:
            chunk = q.get(timeout=0.5)
        except queue.Empty:
            continue
        if chunk is None:
            finish()
            return
        chunks = [chunk]
        eof = False
        try:
            while True:
                c = q.get_nowait()
                if c is None:
                    eof = True
                    break
                chunks.append(c)
        except queue.Empty:
            pass
        sd.feed(np.concatenate(chunks))
        pages_val.value = len(sd.pages)
        if eof:
            finish()
            return


# ---------------------------------------------------------------------------

class FileGraph(gr.top_block):
    """file_source -> RingSink for the single-carrier parity check."""

    def __init__(self, cfile, q):
        gr.top_block.__init__(self, "flex_inmem_mp_file")
        self.src = blocks.file_source(gr.sizeof_gr_complex, cfile, False)
        self.ring = RingSink("file", q)
        self.connect(self.src, self.ring)


class LiveGraph(gr.top_block):
    """SoapySDR RSPdx -> per-carrier freq_xlate -> RingSink (one mp.Queue each)."""

    def __init__(self, queues):
        gr.top_block.__init__(self, "flex_inmem_mp_live")
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
        # full 2.5 MS/s input rate -> the SDR overran. ~137 taps fixes it.
        taps = firdes.low_pass(1.0, SAMP_RATE, 9000, 60000)
        self.rings = {}
        self._blocks = []
        for carrier in CARRIERS:
            offset = carrier - CENTER
            xlate = filter.freq_xlating_fir_filter_ccc(DECIM, taps, offset, SAMP_RATE)
            ring = RingSink(carrier, queues[carrier])
            self.connect(self.src, xlate, ring)
            self.rings[carrier] = ring
            self._blocks += [xlate, ring]


# ---------------------------------------------------------------------------

def run_file(cfile, in_rate):
    q = mp.Queue()                       # unbounded: no drops in parity test
    pages_val = mp.Value('i', 0)
    p = mp.Process(target=worker_proc,
                   args=("file", q, in_rate, WINDOW_FR, ADVANCE_FR, None, pages_val),
                   daemon=True)
    p.start()

    tb = FileGraph(cfile, q)
    t0 = time.time()
    tb.start()
    tb.wait()                            # file_source EOF
    q.put(None)                          # sentinel -> worker flush + exit
    p.join()
    dt = time.time() - t0

    n_samp = os.path.getsize(cfile) // 8
    secs = n_samp / in_rate
    print(f"file={cfile}")
    print(f"in_rate={in_rate} samples={n_samp} ({secs:.1f}s of IQ)")
    print(f"window={WINDOW_FR}fr advance={ADVANCE_FR}fr")
    print(f"trustworthy A/B alpha pages (via per-process ring->proc): {pages_val.value}")
    print(f"wall={dt:.1f}s  ({dt/secs:.2f}x realtime, single carrier)")
    return pages_val.value


def run_live(fifo_dir):
    os.makedirs(LOG_DIR, exist_ok=True)
    queues = {c: mp.Queue(maxsize=64) for c in CARRIERS}
    pages = {c: mp.Value('i', 0) for c in CARRIERS}

    procs = {}
    for carrier in CARRIERS:
        p = mp.Process(target=worker_proc,
                       args=(carrier, queues[carrier], IN_RATE, WINDOW_FR,
                             ADVANCE_FR, f"{LOG_DIR}/{carrier}.flexdec.log",
                             pages[carrier]),
                       daemon=True)
        p.start()
        procs[carrier] = p

    tb = LiveGraph(queues)
    print(f"flex_inmem_mp LIVE: RSPdx @ {CENTER/1e6:.4f} MHz, {SAMP_RATE} S/s -> "
          f"{len(CARRIERS)} carriers @ {IN_RATE} Hz -> StreamDecoder PROCESSES "
          f"(window={WINDOW_FR} advance={ADVANCE_FR}) numba={F._HAVE_NUMBA} "
          f"resample={'OFF' if IN_RATE == F.SAMP else 'ON'}",
          file=sys.stderr, flush=True)
    tb.start()
    try:
        while True:
            time.sleep(30)
            drops = {c: tb.rings[c].dropped_chunks for c in CARRIERS}
            pgs = {c: pages[c].value for c in CARRIERS}
            print(f"[inmem-mp] pages={pgs} dropped={drops}", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        tb.stop(); tb.wait()
        for q in queues.values():
            q.put(None)                  # flush each worker
        for p in procs.values():
            p.join(timeout=5)


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
