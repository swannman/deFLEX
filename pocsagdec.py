"""POCSAG (CCIR Radio Paging Code No. 1) alpha decoder.

Sibling to flexdec.py for the p340 paging rig. Focus: alphanumeric pages only
(the 7-bit-ASCII message type) -- numeric/tone pages are ignored.

=============================================================================
A PRIMER FOR THE READER WHO IS A SOFTWARE ENGINEER BUT NOT AN RF ENGINEER
=============================================================================
If you have never touched a radio, here is the whole journey a pager message
takes and exactly where this file joins it. Everything below is "just signal
processing on arrays of numbers" -- there is no analog magic hiding in here.

1. ENCODING THE BITS (done by the transmitter, far away).
   To send bits over the air a transmitter uses FSK -- frequency-shift keying.
   To send a 0 it nudges the carrier frequency one way; to send a 1 it nudges
   it the other. POCSAG uses TWO frequencies (2-FSK), about +/-4.5 kHz around
   the channel center (the DEVIATION constant). So "the data" is literally
   encoded as "is the instantaneous frequency high or low right now?".

2. WHAT THE RADIO HANDS US.
   The SDR (software-defined radio) gives us a stream of complex numbers --
   "I/Q samples" -- that represent the received signal as a rotating phasor
   (a unit-ish vector spinning in the complex plane). The *speed* at which that
   phasor rotates IS the instantaneous frequency. Our sample rate is 250 kHz,
   so we get 250000 of these complex numbers per second.

3. FM DEMODULATION (`fm_demod`): radio -> a 1-D wiggle.
   To recover "high tone vs low tone" we measure how far the phasor rotated
   between each adjacent pair of samples: angle(x[n] * conj(x[n-1])). A positive
   angle means it spun one way (one tone), negative the other. This collapses
   the 2-D complex stream into a 1-D real-valued signal -- the "demod" -- whose
   sign carries the bit. (This is the same math an analog FM radio's
   discriminator does; here it is two lines of numpy.)

4. SYMBOL TIMING (`_decimate_to_osr` + the phase search in `decode_demod`).
   The transmitter sends 1200 bits/second (the BAUD). At 250 kHz that is ~208
   samples per bit -- far more than we need. We resample so each bit ("symbol")
   spans exactly OSR=8 samples. But WHICH of those 8 samples best represents the
   bit? The middle of the bit is cleanest (the open centre of the "eye
   diagram"); the edges are where the signal is sliding between tones. Many
   receivers run a feedback loop (Mueller & Muller, Gardner, ...) to lock onto
   that centre. We do something simpler and just as good here because the frame
   sync re-appears constantly: we try ALL 8 sampling phases (and both tone
   polarities) and keep whichever one makes the known sync word appear most
   often. See `decode_demod`. (Note: an earlier draft of this docstring claimed
   Mueller&Muller recovery -- there is no M&M loop; the brute phase search
   replaced it.)

5. ERROR CORRECTION (`_decode_word` -> flexdec BCH + Chase).
   The air is noisy, so POCSAG wraps every 32-bit codeword in a BCH(31,21)
   error-correcting code (+1 parity bit) able to FIX up to 2 flipped bits per
   word with no retransmission. Words that still fail get a second pass with
   "Chase" soft-decision decoding, which uses our per-bit confidence to guess
   which bits to flip. This is the single most important reason a weak page
   still decodes cleanly.

6. REASSEMBLY + SANITY GATE (`_scan_bits`, `_alpha_from_payloads`,
   `is_clean_alpha`). The corrected codewords are stitched back into the
   original 7-bit ASCII text, then screened so that FEC-manufactured nonsense
   never reaches the log.

=============================================================================
WHY THIS FILE CAN REUSE flexdec
=============================================================================
POCSAG and FLEX share the *identical* BCH(31,21) codeword (generator 0x769).
The POCSAG frame-sync codeword 0x7CD215D8 and idle codeword 0x7A89C197 both
verify with zero syndrome under flexdec.bch3121, and flexdec.bch3121_chase
soft-decodes POCSAG words unchanged. english_score is reused for the
alpha-readability gate. That hard-won FEC + language code lives in one place;
this file only adds the POCSAG-specific framing on top.

=============================================================================
THE WIRE FORMAT (2-FSK NRZ)
=============================================================================
  preamble (>=576 alternating bits) -> [ FSC + 8 frames x 2 codewords ] repeated.
  A codeword is 32 bits, transmitted MSB first:
    bit31 (first sent) = flag : 0 = address codeword, 1 = message codeword
    address cw : bits30..13 = 18-bit address, bits12..11 = 2 function bits
    message cw : bits30..11 = 20 payload bits
    bits10..1  = 10 BCH check bits, bit0 = even parity
  Alpha text = payload bits of consecutive message codewords concatenated in
  transmission order, sliced into 7-bit chars LSB-first (first bit = char LSB).

  The gotcha that bites everyone: POCSAG splits each batch into 8 numbered
  "frames", and a given pager only listens in the one frame its address hashes
  to (a power-saving trick -- the receiver can sleep the other 7/8 of the time).
  The 18-bit address carried in the codeword therefore DROPS its low 3 bits;
  those 3 bits are implied by WHICH frame slot the word arrived in. `_scan_bits`
  rebuilds the full capcode as (addr18 << 3) | frame_index.

Pipeline: complex baseband -> FM demod -> decimate to OSR samples/symbol ->
DC-block -> best-of-8 sampling phase (chosen by FSC correlation, not an M&M
loop) -> FSC-locked batch slicer -> per-codeword BCH (+Chase soft) ->
alpha assembly -> english gate.
"""
import sys
import numpy as np
from scipy import signal

import flexdec as F   # bch3121, bch3121_chase, english_score

# ---------------------------------------------------------------------------
# PROTOCOL CONSTANTS
# ---------------------------------------------------------------------------
# The Frame Sync Codeword (FSC) is a fixed 32-bit pattern the transmitter sends
# at the start of every batch. It is the lighthouse we steer by: finding it in
# the bit stream tells us both WHERE codewords begin and that our sampling/
# polarity choices are correct. IDLE is the filler word sent in unused slots --
# spotting it lets us close out a finished message. Both happen to be valid
# BCH(31,21) codewords (zero syndrome), which is how flexdec's BCH validates them.
FSC  = 0x7CD215D8     # frame sync codeword
IDLE = 0x7A89C197     # idle codeword
FSC_INV = FSC ^ 0xFFFFFFFF   # FSC with every bit flipped (the opposite polarity)

IN_RATE_DEFAULT = 250000   # SDR sample rate we expect at the input (Hz)
BAUD = 1200                # POCSAG-1200: 1200 bits (symbols) per second
# OSR = "oversampling ratio" = samples per symbol AFTER `_decimate_to_osr`.
# We deliberately keep 8 samples per bit (not 1) so the phase search in
# `decode_demod` has 8 candidate sampling instants to choose the eye-centre from.
OSR  = 8
DEVIATION = 4500.0    # FSK tone offset from center: POCSAG nominal +/-4.5 kHz

CW_BITS = 32          # every POCSAG codeword is 32 bits
BATCH_CWS = 16        # a batch = 8 frames x 2 codewords = 16 codewords after the FSC
PREAMBLE_MIN = 0      # we lock on the FSC directly; the preamble only helps timing

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


# ---------------------------------------------------------------------------
# DSP FRONT-END: complex radio samples -> a clean 1-D symbol stream
# ---------------------------------------------------------------------------
# These three functions are the entire "analog" half of the decoder. After them
# we are working with a plain real-valued array where the SIGN of each sample is
# the received bit. Everything downstream is bit-twiddling and error correction.

def fm_demod(x):
    """FM-demodulate complex baseband -> instantaneous frequency (rad/sample).

    The product x[n] * conj(x[n-1]) is a complex number whose ANGLE is exactly
    how far the received phasor rotated in one sample step. Rotation rate is
    frequency, and POCSAG put the data in the frequency (2-FSK), so this angle
    sequence IS the demodulated signal: positive = one tone, negative = the
    other. Two lines of numpy replace what an analog radio does with a
    discriminator circuit. Output is one shorter than the input (we need pairs)."""
    return np.angle(x[1:] * np.conj(x[:-1])).astype(np.float64)


def _decimate_to_osr(d, in_rate):
    """Resample the demod so each symbol is EXACTLY OSR samples wide.

    The input arrives at `in_rate` (e.g. 250 kHz) but a symbol is 1/BAUD seconds
    long, so symbols start at non-integer sample positions -- awkward to slice.
    We rationally resample to OSR*BAUD = 9600 samples/s, after which every symbol
    is precisely OSR=8 samples and symbol k starts at sample 8*k. resample_poly
    needs an integer up/down ratio, so we divide both by their GCD."""
    target = OSR * BAUD                         # 9600 for OSR=8, BAUD=1200
    g = np.gcd(int(round(in_rate)), target)
    up, down = target // g, int(round(in_rate)) // g
    return signal.resample_poly(d, up, down)


def _dc_block(y, span_syms=32):
    """Remove slow frequency offset so the decision boundary sits at zero.

    Problem: the transmitter's centre frequency and our SDR's tuner never match
    perfectly, so the whole demod sits shifted up or down by a slowly-drifting
    offset (a DC bias). If we sliced bits at exactly 0 with a nonzero bias, every
    bit could read the same. Fix: estimate that bias as a moving average ~32
    symbols long and subtract it, re-centring the signal on zero. POCSAG data is
    DC-balanced over a few symbols (roughly as many 0s as 1s), so the moving
    average tracks the offset, not the data, and symbol transitions survive."""
    n = int(span_syms * OSR)
    if n < 3 or n >= len(y):
        return y - np.mean(y)
    k = np.ones(n) / n
    base = signal.fftconvolve(y, k, mode="same")
    return y - base


# ---------------------------------------------------------------------------
# BIT / CODEWORD UTILITIES + ERROR CORRECTION
# ---------------------------------------------------------------------------

def _word_at(bits, n):
    """Pack the 32 bits starting at index n into one int, MSB first.

    POCSAG sends each codeword most-significant-bit first, so bits[n] is bit 31
    of the resulting word and bits[n+31] is bit 0. We just shift-and-or them in
    order, which naturally lands the first-received bit in the top position."""
    w = 0
    for i in range(CW_BITS):
        w = (w << 1) | int(bits[n + i])
    return w


def _hamming32(a, b):
    """Hamming distance: how many of the 32 bit positions differ between a and b.
    XOR sets a 1 wherever they disagree; popcount counts those 1s. Used to ask
    'is this received word close enough to the FSC to call it a sync?'."""
    return bin((a ^ b) & 0xFFFFFFFF).count("1")


def _decode_word(w, mag32, chase=True):
    """Error-correct one received codeword. Returns (corrected_word, ok).

    Two-stage, cheapest-first: try HARD BCH decoding (fixes <=2 bit errors from
    the bit pattern alone, very fast). If that fails and we have per-bit
    reliabilities (mag32), fall back to CHASE soft-decision decoding, which is
    allowed to fix more errors by trusting that the lowest-confidence bits are
    the likely culprits. `ok` is False only if both give up -- a genuinely
    corrupted word we must drop. (Both routines are flexdec's, reused verbatim.)"""
    cw, nerr = F.bch3121(w)
    if nerr >= 0:
        return cw, True
    if chase and mag32 is not None:
        cw, metric = F.bch3121_chase(w, mag32, L=4)
        if metric >= 0:
            return cw, True
    return w, False


def _alpha_from_payloads(payload_bits):
    """Reassemble the 7-bit ASCII message text from a run of payload bits.

    Each message codeword contributed 20 payload bits (added MSB-first by
    _scan_bits). Concatenated across codewords, those bits form a continuous
    stream that POCSAG packs as 7-bit characters, LEAST-significant bit first.
    So we fill an accumulator low-bit-first, emit a char every 7 bits, and skip
    the NUL/EOT codes used purely as end-of-message padding.

    payload_bits: list of message-bit ints (0/1) in transmission order.
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


# ---------------------------------------------------------------------------
# SYNCHRONISATION + DECODE
# ---------------------------------------------------------------------------
# FSC as a +/-1 vector (bit 1 -> +1, bit 0 -> -1), precomputed once. Correlating
# a slice of the bit stream against this is just a dot product, and that dot
# product peaks when the slice equals the FSC -- the basis of both timing
# selection (_fsc_score) and batch locking (_scan_bits).
_FSC_PM = np.array([1.0 if (FSC >> (CW_BITS - 1 - i)) & 1 else -1.0
                    for i in range(CW_BITS)], dtype=np.float32)


def _fsc_score(bits):
    """Count near-exact FSC matches in a bit stream -- our timing-quality score.

    The idea: of the OSR=8 candidate sampling phases, the one that lands in the
    open centre of the eye produces the fewest bit errors, so the known sync word
    shows up most often. We therefore RANK phases by 'how many clean FSCs does
    this phase produce', and `decode_demod` keeps the winner. No feedback loop
    needed because the sync word recurs every batch.

    Vectorised trick: map bits to +/-1 and cross-correlate with the FSC pattern.
    A 32-bit window's correlation equals CW_BITS minus twice its Hamming distance
    to the FSC, so a correlation >= CW_BITS-4 is exactly a <=2-bit match -- the
    same 2-error tolerance _scan_bits uses when it locks onto a real batch."""
    if len(bits) < CW_BITS:
        return 0
    pm = bits.astype(np.float32) * 2.0 - 1.0
    corr = np.correlate(pm, _FSC_PM, mode="valid")
    return int(np.count_nonzero(corr >= CW_BITS - 4))


def decode_demod(y, on_page=None):
    """Decode an oversampled, DC-blocked demod into alpha pages.

    This is where 'symbol timing recovery' happens, the cheap way. Two unknowns:
    (a) which of the OSR sampling phases is the eye centre, and (b) the FSK
    POLARITY -- whether a positive demod means bit 1 or bit 0, which depends on
    whether the SDR tuned above or below the carrier and can flip per capture.
    We brute-force both: for each phase take every OSR-th sample, try it and its
    negation, score each with _fsc_score, and keep the (phase, polarity) pair
    that yields the most clean sync words. Then decode that one stream. Cheaper
    and more robust than a tracking loop because the FSC re-syncs every batch.
    Returns list of (address, function, text_bytes, sym_pos)."""
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
    """Walk the bit stream, lock onto each batch via the FSC, and parse pages.

    This is the protocol state machine. We slide one bit at a time looking for a
    word within 2 errors of the FSC; once found, the 16 codewords of that batch
    sit immediately after it at fixed 32-bit spacing, so we decode them in place
    (no further searching) and jump past the whole batch. Within a batch:
      * an ADDRESS codeword (flag bit = 0) starts a new page -- we flush whatever
        message was in progress, then rebuild the full capcode by appending the
        frame number (the codeword's slot, 0..7) to the 18-bit address, because
        POCSAG drops those 3 low bits (see the wire-format note up top);
      * a MESSAGE codeword (flag bit = 1) contributes 20 payload bits to the
        current page;
      * an IDLE word flushes the current page (end of message).

    `soft` (the signed pre-slice samples) gives Chase its per-bit confidence:
    |soft| is large where the symbol sat far from the decision boundary.

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


# ---------------------------------------------------------------------------
# TOP-LEVEL ENTRY POINTS
# ---------------------------------------------------------------------------

def decode_baseband(x, in_rate=IN_RATE_DEFAULT, on_page=None):
    """Full pipeline from complex baseband to alpha pages -- the four front-end
    steps from the primer, in order: FM demod -> resample to OSR/symbol ->
    DC-block -> phase-search + frame parse.
    Returns list of (address, function, text, sym_pos)."""
    d = fm_demod(x)
    y = _decimate_to_osr(d, in_rate)
    y = _dc_block(y)
    return decode_demod(y, on_page=on_page)


# ---------------------------------------------------------------------------
# STREAMING WRAPPER (for the live receiver)
# ---------------------------------------------------------------------------
# Offline we decode a whole file at once. Live, samples arrive forever in small
# chunks and we must emit each page EXACTLY ONCE the moment we have enough of it.
# The classic hazard: if we just decoded back-to-back fixed blocks, a page that
# straddled a block boundary would be cut in half and lost. The fix is the
# "overlap-save" pattern, borrowed from block convolution:
#   * keep a sliding WINDOW of samples and decode it whole;
#   * after each decode, advance by less than a window (ADVANCE < WINDOW), so
#     consecutive windows OVERLAP by (window - advance) seconds;
#   * only EMIT pages whose address codeword began within the first `advance`
#     seconds of the window. A page emitted there still has the full overlap
#     region after it to hold its (bounded-length) message, so it is never
#     truncated; and because every page is emitted by exactly one window's
#     emit-region, we get each page once with no de-dup bookkeeping.
# The overlap just needs to exceed the longest possible transmission. This is
# the POCSAG analogue of flexdec_stream.StreamDecoder.

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
