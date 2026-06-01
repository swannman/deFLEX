#!/usr/bin/env python3
"""SDR + channelization layer (GNU Radio) for the live receivers.

One SoapySDR source is tuned to a center frequency and split into N carriers --
one freq_xlate each, decimated to receiver_core.CHAN_RATE -- with a drop-oldest
RingSink delivering each carrier's baseband to its decode worker process(es).
run_live() wires this up and spawns one receiver_core worker per (carrier,
protocol). Importing this module requires GNU Radio; the --file replay paths in
the receivers avoid it by using receiver_core / the decode cores directly.
"""
import os
import sys
import time
import queue
import multiprocessing as mp

import numpy as np
from gnuradio import gr, filter, soapy
from gnuradio.filter import firdes

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import receiver_core as RC


class RingSink(gr.sync_block):
    """complex64 sink -> one or more mp.Queues (drop-oldest per queue). Runs in
    the parent flowgraph process; the queues carry copied chunks to worker
    PROCESSES. Multiple queues = one carrier fanned to multiple protocol workers."""

    def __init__(self, tag, queues):
        gr.sync_block.__init__(self, name=f"ring_{tag}",
                               in_sig=[np.complex64], out_sig=None)
        self.queues = queues
        self.dropped_chunks = 0

    def work(self, input_items, output_items):
        x = np.asarray(input_items[0], dtype=np.complex64).copy()
        for q in self.queues:
            try:
                q.put_nowait(x)
            except queue.Full:
                try:
                    q.get_nowait()
                    self.dropped_chunks += 1
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(x)
                except queue.Full:
                    pass
        return len(input_items[0])


def build_source(driver, center, samp_rate):
    """SoapySDR source tuned to `center`. sdrplay uses the validated manual gain
    (matches the 900 MHz FLEX setup); other drivers use AGC."""
    src = soapy.source(f"driver={driver}", "fc32", 1, "", "", [""], [""])
    src.set_sample_rate(0, samp_rate)
    src.set_frequency(0, center)
    if driver == "sdrplay":
        src.set_gain_mode(0, False)
        src.set_gain(0, "RFGR", 4)
        src.set_gain(0, "IFGR", 25)
        src.set_antenna(0, "Antenna A")
    else:
        src.set_gain_mode(0, True)          # AGC (rtlsdr/airspy)
    return src


def build_channelizer(carrier_queues, center, samp_rate, driver):
    """source -> per-carrier freq_xlate(decimate to CHAN_RATE) -> RingSink.
    carrier_queues: {carrier_hz: [queue, ...]}. Returns a startable top_block
    with .rings[carrier] for drop-stats."""
    tb = gr.top_block("receiver")
    src = build_source(driver, center, samp_rate)
    decim = int(round(samp_rate / RC.CHAN_RATE))
    # Wide 60 kHz transition keeps the tap count low (~137); after decimation the
    # first alias folds in near CHAN_RATE-9000 ~= 241 kHz, leaving huge guard.
    taps = firdes.low_pass(1.0, samp_rate, 9000, 60000)
    tb.rings = {}
    tb._blocks = []
    for carrier, queues in carrier_queues.items():
        xlate = filter.freq_xlating_fir_filter_ccc(decim, taps, carrier - center, samp_rate)
        ring = RingSink(carrier, queues)
        tb.connect(src, xlate, ring)
        tb.rings[carrier] = ring
        tb._blocks += [xlate, ring]
    return tb


def run_live(flex_carriers, pocsag_carriers, center, samp_rate=RC.SAMP_RATE,
             driver="sdrplay", log_dir=RC.LOG_DIR):
    """Tune one SDR to `center`, channelize, and spawn one worker process per
    (carrier, protocol). Blocks until KeyboardInterrupt."""
    a = RC.assign(flex_carriers, pocsag_carriers)
    oob = RC.check_in_band(list(a), center, samp_rate)
    if oob:
        raise SystemExit(f"carriers outside the {samp_rate/1e6:.1f} MHz window around "
                         f"{center/1e6:.4f} MHz: {[round(c/1e6, 4) for c in oob]} MHz")
    os.makedirs(log_dir, exist_ok=True)

    carrier_queues = {}
    procs = []
    pages = {}                      # (carrier, proto) -> mp.Value
    for carrier, protos in a.items():
        qs = []
        for proto in protos:
            q = mp.Queue(maxsize=RC.QUEUE_MAXCHUNKS)
            pv = mp.Value('i', 0)
            target = RC.flex_worker if proto == "flex" else RC.pocsag_worker
            suffix = "flexdec" if proto == "flex" else "pocsag"
            p = mp.Process(target=target,
                           args=(carrier, q, f"{log_dir}/{carrier}.{suffix}.log", pv),
                           daemon=True)
            p.start()
            procs.append(p)
            pages[(carrier, proto)] = pv
            qs.append(q)
        carrier_queues[carrier] = qs

    tb = build_channelizer(carrier_queues, center, samp_rate, driver)
    nflex = sum(1 for v in a.values() if "flex" in v)
    npoc = sum(1 for v in a.values() if "pocsag" in v)
    print(f"receiver LIVE: {driver} @ {center/1e6:.4f} MHz, {samp_rate} S/s -> "
          f"{len(a)} carriers ({nflex} FLEX, {npoc} POCSAG) @ {RC.CHAN_RATE} Hz -> "
          f"worker PROCESSES numba={RC.F._HAVE_NUMBA}", file=sys.stderr, flush=True)
    tb.start()
    try:
        while True:
            time.sleep(30)
            pgs = {f"{c/1e6:.4f}:{pr}": v.value for (c, pr), v in pages.items()}
            drops = {round(c/1e6, 4): tb.rings[c].dropped_chunks for c in carrier_queues}
            print(f"[recv] pages={pgs} dropped={drops}", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        tb.stop(); tb.wait()
        for qs in carrier_queues.values():
            for q in qs:
                q.put(None)
        for p in procs:
            p.join(timeout=5)
