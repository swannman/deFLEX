#!/usr/bin/env python3
"""Streaming POCSAG decoder built ON TOP of the validated batch core (pocsagdec.py).

pocsagdec.decode_baseband decodes a whole recorded FILE at once. This file makes
the same decoder run forever on a live SDR feed, where samples arrive in small
chunks and each page must be emitted EXACTLY ONCE the moment we have enough of
it. It is the POCSAG analogue of flexdec_stream.StreamDecoder, and uses the same
overlap-save trick (read that class's comment for the full rationale).

The classic hazard: if we decoded back-to-back fixed blocks, a page that
straddled a block boundary would be cut in half and lost. The fix is the
"overlap-save" pattern, borrowed from block convolution:
  * keep a sliding WINDOW of samples and decode it whole;
  * after each decode, advance by less than a window (ADVANCE < WINDOW), so
    consecutive windows OVERLAP by (window - advance) seconds;
  * only EMIT pages whose address codeword began within the first `advance`
    seconds of the window. A page emitted there still has the full overlap
    region after it to hold its (bounded-length) message, so it is never
    truncated; and because every page is emitted by exactly one window's
    emit-region, we get each page once with no de-dup bookkeeping.
The overlap just needs to exceed the longest possible transmission.

Unlike FLEX (frame-periodic on a shared clock) POCSAG has no global slot, so the
exactly-once emission comes entirely from this emit-region rule rather than an
absolute-slot dedup.

Validation (`python3 pocsagdec_stream.py CFILE`): batch (whole-file) vs streamed
(chunked) on the same data; the streamed readable set must cover the batch set.
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# Batch core (pocsagdec.py / paging_core.py) lives at the repo root; in the flat
# /usr/local/bin install it sits beside this file -- cover both layouts.
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import pocsagdec as P


class POCSAGStream:
    """Feed complex baseband (one carrier @ in_rate, default 250k) in arbitrary
    chunks; emits alpha pages once each via on_page(addr, func, text) or read
    self.pages after flush(). Sibling of flexdec_stream.StreamDecoder.

    Unlike FLEX (frame-periodic on a shared clock) POCSAG has no global slot, so
    we slide an overlapped window over the buffer and re-run the validated batch
    decode per window. Exactly-once emission comes from overlap-save: a window
    only emits pages whose address codeword starts within its first `advance`
    symbols, so every emitted page has the full (window - advance) overlap left
    to contain its message. With overlap >= the longest transmission, no page is
    truncated at an edge and none lands in two windows' emit regions -- so no
    de-dup is needed. The final flush emits its whole tail."""

    def __init__(self, in_rate=P.IN_RATE_DEFAULT, window_s=30.0, advance_s=25.0,
                 on_page=None):
        self.in_rate = in_rate
        self.window = int(window_s * in_rate)
        self.advance = int(advance_s * in_rate)
        self.advance_syms = int(advance_s * P.BAUD)   # emit cutoff (symbol units)
        self.on_page = on_page
        self.buf = np.empty(0, dtype=np.complex64)
        self._pending = []
        self._pending_len = 0
        self.base = 0                  # absolute input-sample index of buf[0]
        self.pages = []                # (addr, func, text)

    def _emit(self, win_pages, emit_all):
        for addr, func, text, pos in win_pages:
            if not emit_all and pos >= self.advance_syms:
                continue               # overlap region: the next window emits it
            rec = (addr, func, text)
            self.pages.append(rec)
            if self.on_page:
                self.on_page(rec)

    def _merge_pending(self):
        if not self._pending:
            return
        parts = ([self.buf] if len(self.buf) else []) + self._pending
        self.buf = parts[0] if len(parts) == 1 else np.concatenate(parts)
        self._pending = []
        self._pending_len = 0

    def _drain(self, final=False):
        while True:
            have = len(self.buf)
            if not final:
                if have < self.window:
                    return
                self._emit(P.decode_baseband(self.buf[:self.window], self.in_rate),
                           emit_all=False)
                self.buf = self.buf[self.advance:]
                self.base += self.advance
            else:
                if have <= self.window:    # tail fits one window -> emit all of it
                    if have:
                        self._emit(P.decode_baseband(self.buf, self.in_rate),
                                   emit_all=True)
                    self.buf = self.buf[have:]
                    self.base += have
                    return
                self._emit(P.decode_baseband(self.buf[:self.window], self.in_rate),
                           emit_all=False)
                self.buf = self.buf[self.advance:]
                self.base += self.advance

    def feed(self, samples):
        s = np.asarray(samples, dtype=np.complex64)
        self._pending.append(s)
        self._pending_len += len(s)
        if len(self.buf) + self._pending_len >= self.window:
            self._merge_pending()
            self._drain(final=False)

    def flush(self):
        self._merge_pending()
        self._drain(final=True)
        return self.pages


def validate(cfile, in_rate=P.IN_RATE_DEFAULT, chunk_s=2.0,
             en_floor=P.EN_FLOOR_DEFAULT):
    """Batch (whole-file) vs streamed (chunked) on the same data. Compares only
    pages that pass is_clean_alpha -- the SAME gate the live receiver logs
    behind. The batch picks one global sampling phase over the whole file while
    the stream picks the best phase per window, so sub-readable garbage gets
    different FEC bit-errors in each; the gate screens that out so what remains
    is real pages, and the streamed set must cover the batch set."""
    def readable(pages):
        return {p[2] for p in pages if P.is_clean_alpha(p[2], en_floor)}

    x = np.fromfile(cfile, dtype=np.complex64)
    print(f"loaded {len(x)} samples ({len(x)/in_rate:.1f}s @ {in_rate/1e3:.0f}k)")
    batch = readable(P.decode_baseband(x, in_rate))
    print(f"batch readable pages: {len(batch)}")
    ps = POCSAGStream(in_rate=in_rate)
    chunk = int(chunk_s * in_rate)
    for i in range(0, len(x), chunk):
        ps.feed(x[i:i + chunk])
    ps.flush()
    stream = readable([(a, f, t) for a, f, t in ps.pages])
    print(f"stream readable pages: {len(stream)}")
    missing = batch - stream
    extra = stream - batch
    print(f"overlap={len(batch & stream)} missing={len(missing)} extra={len(extra)}")
    for t in sorted(missing):
        print(f"  MISSING {t.decode('ascii','replace')!r}")
    for t in sorted(extra):
        print(f"  EXTRA   {t.decode('ascii','replace')!r}")
    ok = not missing
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: pocsagdec_stream.py CFILE [in_rate]", file=sys.stderr)
        sys.exit(2)
    cf = sys.argv[1]
    ir = float(sys.argv[2]) if len(sys.argv) > 2 else P.IN_RATE_DEFAULT
    sys.exit(0 if validate(cf, in_rate=ir) else 1)
