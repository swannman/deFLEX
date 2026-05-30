"""POCSAG (CCIR Radio Paging Code No. 1) alpha decoder.

Sibling to flexdec.py for the p340 paging rig. Focus: alphanumeric pages only
(the 7-bit-ASCII message type) -- numeric/tone pages are ignored.

Why this can reuse flexdec: POCSAG and FLEX share the *identical* BCH(31,21)
codeword (generator 0x769). The POCSAG frame-sync codeword 0x7CD215D8 and idle
codeword 0x7A89C197 both verify with zero syndrome under flexdec.bch3121, and
flexdec.bch3121_chase soft-decodes POCSAG words unchanged. english_score is
reused for the alpha-readability gate.

Signal model (2-FSK NRZ):
  preamble (>=576 alternating bits) -> [ FSC + 8 frames x 2 codewords ] repeated.
  A codeword is 32 bits, transmitted MSB first:
    bit31 (first sent) = flag : 0 = address codeword, 1 = message codeword
    address cw : bits30..13 = 18-bit address, bits12..11 = 2 function bits
    message cw : bits30..11 = 20 payload bits
    bits10..1  = 10 BCH check bits, bit0 = even parity
  Alpha text = payload bits of consecutive message codewords concatenated in
  transmission order, sliced into 7-bit chars LSB-first (first bit = char LSB).

Pipeline: complex baseband -> FM demod -> decimate to OSR samples/symbol ->
DC-block -> Mueller&Muller timing recovery (1 soft sample/symbol) -> FSC-locked
batch slicer -> per-codeword BCH (+Chase soft) -> alpha assembly -> english gate.
"""
import sys
import numpy as np
from scipy import signal

import flexdec as F   # bch3121, bch3121_chase, english_score

FSC  = 0x7CD215D8     # frame sync codeword
IDLE = 0x7A89C197     # idle codeword
FSC_INV = FSC ^ 0xFFFFFFFF

IN_RATE_DEFAULT = 250000
BAUD = 1200
OSR  = 8              # samples/symbol after decimation (M&M interpolates within)
DEVIATION = 4500.0    # POCSAG nominal +/-4.5 kHz

CW_BITS = 32
BATCH_CWS = 16        # 8 frames x 2 codewords
PREAMBLE_MIN = 0      # we lock on FSC directly, preamble only aids the DPLL

EN_FLOOR_DEFAULT = 0.45   # english_score gate; lowest real SNO911 page = 0.49
ALPHA_MIN_LEN = 12        # real dispatches run 34+ chars. Sub-12 readable
                          # strings ('er St', 'Qf') are FEC fragments, never
                          # pages -- english_score alone rates a 2-char run 1.0.


def is_clean_alpha(text, en_floor=EN_FLOOR_DEFAULT):
    """Reliability gate shared by the receiver's log path and the stream
    validator: a real alpha page is long, all-printable, and English-like.
    Two failure modes slip past english_score on its own -- tiny strings
    (ratio-based score saturates) and printable-but-garbled runs carrying
    control bytes (e.g. '...\\x0f...') -- so screen length and control chars
    first. Embedded controls (other than TAB/CR/LF) only appear via FEC
    corruption, so any such byte fails the page."""
    if len(text) < ALPHA_MIN_LEN:
        return False
    if any((c < 32 and c not in (9, 10, 13)) or c == 127 for c in text):
        return False
    return F.english_score(text) >= en_floor


def fm_demod(x):
    """Instantaneous frequency (rad/sample) of complex baseband."""
    return np.angle(x[1:] * np.conj(x[:-1])).astype(np.float64)


def _decimate_to_osr(d, in_rate):
    """Resample the demod to exactly OSR*BAUD samples/s (integer OSR/symbol)."""
    target = OSR * BAUD                         # 9600 for OSR=8, BAUD=1200
    g = np.gcd(int(round(in_rate)), target)
    up, down = target // g, int(round(in_rate)) // g
    return signal.resample_poly(d, up, down)


def _dc_block(y, span_syms=32):
    """Remove slow frequency offset: subtract a moving average ~span_syms long.
    POCSAG preamble/data is DC-balanced over a few symbols, so this centres the
    eye without touching symbol transitions."""
    n = int(span_syms * OSR)
    if n < 3 or n >= len(y):
        return y - np.mean(y)
    k = np.ones(n) / n
    base = signal.fftconvolve(y, k, mode="same")
    return y - base


def _word_at(bits, n):
    """Pack 32 bits (MSB first) starting at index n into an int."""
    w = 0
    for i in range(CW_BITS):
        w = (w << 1) | int(bits[n + i])
    return w


def _hamming32(a, b):
    return bin((a ^ b) & 0xFFFFFFFF).count("1")


def _decode_word(w, mag32, chase=True):
    """Return (corrected_word, ok). Uses hard BCH then Chase-II soft fallback."""
    cw, nerr = F.bch3121(w)
    if nerr >= 0:
        return cw, True
    if chase and mag32 is not None:
        cw, metric = F.bch3121_chase(w, mag32, L=4)
        if metric >= 0:
            return cw, True
    return w, False


def _alpha_from_payloads(payload_bits):
    """payload_bits: list of message-bit ints (0/1) in transmission order.
    7-bit chars, LSB first (first bit = char LSB)."""
    chars = bytearray()
    acc = 0
    nb = 0
    for bit in payload_bits:
        acc |= (bit & 1) << nb
        nb += 1
        if nb == 7:
            c = acc & 0x7F
            if c not in (0x00, 0x04):        # NUL / EOT padding
                chars.append(c)
            acc = 0
            nb = 0
    return bytes(chars)


_FSC_PM = np.array([1.0 if (FSC >> (CW_BITS - 1 - i)) & 1 else -1.0
                    for i in range(CW_BITS)], dtype=np.float32)


def _fsc_score(bits):
    """Count near-exact FSC codeword matches in a bit stream (sampling-quality
    proxy). The phase whose sample instant sits in the centre of the eye yields
    the most low-error FSC words, so this picks the optimal timing without an
    M&M loop. Vectorized as a +/-1 cross-correlation with the FSC pattern: a
    32-bit window scores CW_BITS minus twice its Hamming distance to FSC, so
    >= CW_BITS-4 counts every <=2-bit match (the same threshold _scan_bits uses
    to lock)."""
    if len(bits) < CW_BITS:
        return 0
    pm = bits.astype(np.float32) * 2.0 - 1.0
    corr = np.correlate(pm, _FSC_PM, mode="valid")
    return int(np.count_nonzero(corr >= CW_BITS - 4))


def decode_demod(y, on_page=None):
    """Decode an oversampled (OSR samples/symbol) DC-blocked demod into alpha
    pages. The resampler already locks to exactly OSR samples/symbol and the FSC
    re-syncs every batch, so instead of an M&M loop we score every sampling phase
    and both FSK polarities by exact-FSC count, then decode only the best one
    (the centred eye gives the fewest bit errors). Returns list of
    (address, function, text_bytes)."""
    best_soft = None
    best_hits = -1
    for phase in range(OSR):
        sym = y[phase::OSR]
        for soft in (sym, -sym):           # both FSK polarities
            hits = _fsc_score((soft > 0).astype(np.uint8))
            if hits > best_hits:
                best_hits = hits
                best_soft = soft
    pages = []
    if best_soft is not None and best_hits > 0:
        bits = (best_soft > 0).astype(np.uint8)
        pages = _scan_bits(bits, best_soft)
    if on_page:
        for p in pages:
            on_page(p)
    return pages


def _scan_bits(bits, soft):
    """Find FSC-locked batches, decode codewords, assemble alpha messages.
    Returns (address, function, text, sym_pos) where sym_pos is the symbol index
    of the page's address codeword -- used by POCSAGStream to dedup the same
    physical page seen in two overlapping windows."""
    pages = []
    N = len(bits)
    n = 0
    # message-assembly state (an alpha page can span frames/batches)
    cur_addr = None
    cur_func = 0
    cur_payload = []
    cur_pos = 0

    def flush():
        nonlocal cur_addr, cur_payload
        if cur_addr is not None and cur_payload:
            text = _alpha_from_payloads(cur_payload)
            if text:
                pages.append((cur_addr, cur_func, text, cur_pos))
        cur_addr = None
        cur_payload = []

    while n + CW_BITS <= N:
        w = _word_at(bits, n)
        if _hamming32(w, FSC) <= 2:
            # batch: 16 codewords immediately follow the FSC
            base = n + CW_BITS
            for cw_idx in range(BATCH_CWS):
                p = base + cw_idx * CW_BITS
                if p + CW_BITS > N:
                    break
                raw = _word_at(bits, p)
                mag = np.abs(soft[p:p + CW_BITS]) if p + CW_BITS <= len(soft) else None
                # mag32[b] = reliability of integer bit b (LSB..MSB).
                # transmit bit i sits at integer bit (31-i); reliability = |soft[p+i]|.
                mag32 = mag[::-1] if mag is not None and len(mag) == CW_BITS else None
                word, ok = _decode_word(raw, mag32)
                if not ok:
                    continue
                if word == IDLE:
                    flush()
                    continue
                flag = (word >> 31) & 1
                if flag == 0:                       # address codeword
                    flush()
                    addr18 = (word >> 13) & 0x3FFFF
                    func = (word >> 11) & 0x3
                    frame = cw_idx // 2             # 0..7
                    cur_addr = (addr18 << 3) | frame
                    cur_func = func
                    cur_payload = []
                    cur_pos = p
                else:                               # message codeword
                    if cur_addr is not None:
                        for b in range(30, 10, -1):  # bits 30..11, MSB first
                            cur_payload.append((word >> b) & 1)
            n = base + BATCH_CWS * CW_BITS
        else:
            n += 1
    flush()
    return pages


def decode_baseband(x, in_rate=IN_RATE_DEFAULT, on_page=None):
    """Full pipeline from complex baseband to alpha pages.
    Returns list of (address, function, text, sym_pos)."""
    d = fm_demod(x)
    y = _decimate_to_osr(d, in_rate)
    y = _dc_block(y)
    return decode_demod(y, on_page=on_page)


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

    def __init__(self, in_rate=IN_RATE_DEFAULT, window_s=30.0, advance_s=25.0,
                 on_page=None):
        self.in_rate = in_rate
        self.window = int(window_s * in_rate)
        self.advance = int(advance_s * in_rate)
        self.advance_syms = int(advance_s * BAUD)   # emit cutoff (symbol units)
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
                self._emit(decode_baseband(self.buf[:self.window], self.in_rate),
                           emit_all=False)
                self.buf = self.buf[self.advance:]
                self.base += self.advance
            else:
                if have <= self.window:    # tail fits one window -> emit all of it
                    if have:
                        self._emit(decode_baseband(self.buf, self.in_rate),
                                   emit_all=True)
                    self.buf = self.buf[have:]
                    self.base += have
                    return
                self._emit(decode_baseband(self.buf[:self.window], self.in_rate),
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


def _validate_stream(cfile, in_rate=IN_RATE_DEFAULT, chunk_s=2.0,
                     en_floor=EN_FLOOR_DEFAULT):
    """Batch (whole-file) vs streamed (chunked) on the same data. Compares only
    pages that pass is_clean_alpha -- the SAME gate the live receiver logs
    behind. The batch picks one global sampling phase over the whole file while
    the stream picks the best phase per window, so sub-readable garbage gets
    different FEC bit-errors in each; the gate screens that out so what remains
    is real pages, and the streamed set must cover the batch set."""
    def readable(pages):
        return {p[2] for p in pages if is_clean_alpha(p[2], en_floor)}

    x = np.fromfile(cfile, dtype=np.complex64)
    print(f"loaded {len(x)} samples ({len(x)/in_rate:.1f}s @ {in_rate/1e3:.0f}k)")
    batch = readable(decode_baseband(x, in_rate))
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


def _main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cfile")
    ap.add_argument("--in-rate", type=float, default=IN_RATE_DEFAULT)
    ap.add_argument("--min-en", type=float, default=0.0,
                    help="english_score floor to print a page (0 = print all)")
    ap.add_argument("--validate-stream", action="store_true",
                    help="A/B the streaming wrapper against the batch decode")
    args = ap.parse_args()
    if args.validate_stream:
        sys.exit(0 if _validate_stream(args.cfile, in_rate=args.in_rate) else 1)
    x = np.fromfile(args.cfile, dtype=np.complex64)
    pages = decode_baseband(x, in_rate=args.in_rate)
    shown = 0
    for addr, func, text, _pos in pages:
        en = F.english_score(text)
        if en < args.min_en:
            continue
        shown += 1
        body = text.decode("ascii", "replace")
        print(f"cap={addr:>8} f={func} en={en:.2f}  {body!r}")
    print(f"\n{len(pages)} alpha pages decoded, {shown} shown "
          f"(en>={args.min_en})", file=sys.stderr)


if __name__ == "__main__":
    _main()
