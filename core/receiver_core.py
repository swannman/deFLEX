#!/usr/bin/env python3
"""Shared decode-worker layer for the live receivers (GNU-Radio-free).

Holds the per-channel decode workers and their viewer-compatible log callbacks,
plus carrier planning helpers. A worker drains an mp.Queue of complex64 baseband
into a streaming decode core -- FLEXStream for FLEX, POCSAGStream for POCSAG --
and appends log lines. The SDR source + channelization that feed those queues
live in receiver_sdr.py (which imports GNU Radio); keeping them separate lets the
receivers' --file replay paths run without GNU Radio installed.
"""
import os
import sys
import time
import queue

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # sibling cores
import flex_core as F
import pocsag_core as P

# Per-channel baseband rate. == flex_core.SAMP, so FLEXStream needs no resample;
# POCSAGStream resamples 250k -> 9600 internally. One rate feeds both protocols.
CHAN_RATE  = 250_000
SAMP_RATE  = 2_500_000          # default SDR capture rate (/10 -> CHAN_RATE)
WINDOW_FR  = 32                 # FLEXStream window (>=16 reproduces batch)
ADVANCE_FR = 28                 # 4-frame overlap margin
LOG_DIR    = "/var/log/flex"
EN_FLOOR   = 0.45               # POCSAG english_score gate (is_clean_alpha)
# Per-carrier queue depth: ~one window-decode of input (~17 s @ 250 kS/s) so the
# multi-second decode burst never overflows. Drop-oldest on full.
QUEUE_MAXCHUNKS = 1024


def _escape(body):
    return (body.decode("ascii", "replace")
            .replace("\\", "\\\\").replace("\n", "\\n")
            .replace("\r", "\\r").replace("\t", "\\t"))


def make_flex_on_page(carrier, lf):
    """FLEX log line: <ts> FLEX|<carrier>|<slot>|<tier>|0|ALN|<body>. ALN only."""
    def on_page(rec):
        slot, typ, body, tier, pr, en = rec
        if typ != "ALN":
            return
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        line = "%s FLEX|%d|%d|%s|0|ALN|%s" % (ts, carrier, slot, tier, _escape(body))
        lf.write(line + "\n") if lf else print(line)
    return on_page


def make_pocsag_on_page(carrier, lf):
    """POCSAG log line: <ts> FLEX|<carrier>|0|A|<capcode>|ALN|<body>. The real RIC
    goes in the viewer's dedup field; gated by english_score (is_clean_alpha)."""
    def on_page(rec):
        addr, func, text = rec
        if not P.is_clean_alpha(text, EN_FLOOR):
            return
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        line = "%s FLEX|%d|0|A|%d|ALN|%s" % (ts, carrier, addr, _escape(text))
        lf.write(line + "\n") if lf else print(line)
    return on_page


def _pump(q, decoder, pages_val):
    """Drain the mp.Queue into a streaming decoder, coalescing any backlog into
    one feed() so the scheduler sees big blocks. None on the queue = EOF."""
    while True:
        try:
            chunk = q.get(timeout=0.5)
        except queue.Empty:
            continue
        if chunk is None:
            decoder.flush()
            if pages_val is not None:
                pages_val.value = len(decoder.pages)
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
        decoder.feed(np.concatenate(chunks))
        if pages_val is not None:
            pages_val.value = len(decoder.pages)
        if eof:
            decoder.flush()
            if pages_val is not None:
                pages_val.value = len(decoder.pages)
            return


def flex_worker(carrier, q, log_path, pages_val=None, inv=False):
    lf = open(log_path, "a", buffering=1) if log_path else None
    sd = F.FLEXStream(cfg=dict(in_rate=CHAN_RATE, inv=inv),
                      window_frames=WINDOW_FR, advance_frames=ADVANCE_FR,
                      on_page=make_flex_on_page(carrier, lf))
    _pump(q, sd, pages_val)


def pocsag_worker(carrier, q, log_path, pages_val=None):
    lf = open(log_path, "a", buffering=1) if log_path else None
    ps = P.POCSAGStream(in_rate=CHAN_RATE, on_page=make_pocsag_on_page(carrier, lf))
    _pump(q, ps, pages_val)


# --- carrier planning -------------------------------------------------------

def assign(flex_carriers, pocsag_carriers):
    """-> {carrier_hz: ['flex'|'pocsag', ...]} over the union of both lists."""
    flex_set, pocsag_set = set(flex_carriers), set(pocsag_carriers)
    out = {}
    for c in sorted(flex_set | pocsag_set):
        protos = []
        if c in flex_set:
            protos.append("flex")
        if c in pocsag_set:
            protos.append("pocsag")
        out[c] = protos
    return out


def check_in_band(carriers, center, samp_rate):
    """Carriers must fall within the SDR capture window. Returns the offenders."""
    half = samp_rate / 2
    return [c for c in carriers if abs(c - center) > half]


def parse_gain(spec):
    """Parse a --gain string into the form build_source expects (kept here so the
    GR-free dry-run path can parse without importing GNU Radio):
      None         -> None   (use the per-driver default in build_source)
      'agc'        -> 'agc'   (hardware AGC)
      '37'         -> 37.0    (overall gain in dB; Soapy distributes across stages)
      'LNA=14,MIX=12,VGA=11' -> {'LNA':14.0,'MIX':12.0,'VGA':11.0}  (per element)
    """
    if spec is None:
        return None
    spec = spec.strip()
    if spec.lower() == "agc":
        return "agc"
    if "=" in spec:
        out = {}
        for tok in spec.split(","):
            if tok.strip():
                k, v = tok.split("=", 1)
                out[k.strip()] = float(v)
        return out
    return float(spec)


def plan_text(flex_carriers, pocsag_carriers, center, samp_rate, driver):
    """Human-readable channelization plan for --dry-run (no SDR access)."""
    a = assign(flex_carriers, pocsag_carriers)
    lines = [f"driver={driver} center={center/1e6:.4f} MHz "
             f"samp_rate={samp_rate/1e6:.2f} MS/s -> {len(a)} carriers @ "
             f"{CHAN_RATE/1e3:.0f} kHz"]
    for c, protos in a.items():
        lines.append(f"  {c/1e6:9.4f} MHz  offset {(c-center)/1e3:+8.1f} kHz  "
                     f"-> {'+'.join(protos)}")
    oob = check_in_band(list(a), center, samp_rate)
    if oob:
        lines.append(f"  !! OUT OF BAND (>{samp_rate/2/1e6:.2f} MHz from center): "
                     f"{[round(c/1e6, 4) for c in oob]}")
    return "\n".join(lines)
