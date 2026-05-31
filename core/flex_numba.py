#!/usr/bin/env python3
"""Numba @njit kernels for the GIL-bound integer hot path of flex_core.py.

deFLEX's matched-filter bank (the 58% hotspot) is already vectorized numpy and
releases the GIL, so it parallelizes across decoder threads as-is. What does NOT
parallelize is the per-frame BCH/deinterleave inner loop: pure-Python scalar bit
twiddling (the 32x8 phase transpose, BCH syndrome reduction, Chase-II soft FEC)
that holds the GIL the whole time. Those loops are what these kernels replace.

The kernels are bit-exact reimplementations of paging_core's _syndrome / bch3121 /
bch3121_chase / the deinterleave_decode block loop. flex_core.py imports them with
a pure-Python fallback and is validated to produce the identical trustworthy A/B
page set (32 on iq_929612500_250k.cfile) with or without numba.

Design notes vs the pure-Python originals:
  * The dict syndrome->error table (_BCH_TBL) becomes a dense int64[2048] array
    (BCH syndrome is 11 bits) so it is njit-indexable; 0 = "no correctable error"
    (a valid 1/2-bit error pattern is always nonzero, and a nonzero syndrome is
    never produced by the no-error case, so 0 is an unambiguous miss sentinel).
  * reverse_bits32 + the FLEX 21-bit unmask are folded into the kernel via a
    passed-in rev8 table, so a decoded word comes out ready to parse.
  * The per-call median margin normalization stays in Python (it needs np.median
    over the whole call and is cheap), so the kernel returns raw margins.
"""
import numpy as np
from numba import njit

BCH_N, BCH_K, BCH_POLY = 31, 21, 0x769


@njit(inline="always")
def _popcount32(x):
    x &= 0xFFFFFFFF
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


@njit(inline="always")
def _syndrome_nb(data):
    syn = data >> 1
    mask = 1 << (BCH_N - 1)
    coeff = BCH_POLY << (BCH_K - 1)
    n = BCH_K
    while n > 0:
        if syn & mask:
            syn ^= coeff
        mask >>= 1
        coeff >>= 1
        n -= 1
    if _popcount32(data) & 1:
        syn |= (1 << (BCH_N - BCH_K))
    return syn


@njit(inline="always")
def _bch_nb(data, bch_arr):
    """returns (corrected_word, nerr); nerr in {0,1,2,-1}"""
    s = _syndrome_nb(data)
    if s == 0:
        return data, 0
    e = bch_arr[s]
    if e == 0:
        return data, -1
    return data ^ e, _popcount32(e)


@njit(inline="always")
def _chase_nb(word, mag32, L, bch_arr):
    """Chase-II soft decode. Returns (codeword, metric); metric < 0 on failure."""
    order = np.argsort(mag32)[:L]
    best_cw = word
    best_d = -1.0
    found = False
    for combo in range(1 << L):
        t = word
        for bi in range(L):
            if combo & (1 << bi):
                t ^= (1 << order[bi])
        cw, nerr = _bch_nb(t, bch_arr)
        if nerr < 0:
            continue
        diff = cw ^ word
        d = 0.0
        b = 0
        while diff:
            if diff & 1:
                d += mag32[b]
            diff >>= 1
            b += 1
        if (not found) or d < best_d:
            best_d = d
            best_cw = cw
            found = True
    if not found:
        return word, -1.0
    return best_cw, best_d


@njit(cache=True)
def deint_decode_core(bits, mag, chase, L, bch_arr, rev8):
    """Bit-exact njit core of flex_core.deinterleave_decode (have_mag branch).

    bits : uint8[nblocks*256]    deinterleaved phase bits
    mag  : float64[nblocks*256]  per-bit reliability (same layout as bits)
    returns (out_words int64[nblocks*8], statuses int64[..], margins float64[..],
             nfail, ncorr, nchase). out_words are already reverse_bits32'd and
             FLEX 21-bit unmasked, ready for parse_frame.
    """
    nblocks = bits.shape[0] // 256
    nwords = nblocks * 8
    out_words = np.empty(nwords, dtype=np.int64)
    statuses = np.empty(nwords, dtype=np.int64)
    margins = np.empty(nwords, dtype=np.float64)
    nfail = 0
    ncorr = 0
    nchase = 0
    cw = np.empty(8, dtype=np.int64)
    mw = np.empty((8, 32), dtype=np.float64)
    w = 0
    for blk in range(nblocks):
        base = blk * 256
        for j in range(8):
            cw[j] = 0
        for i in range(32):
            for j in range(8):
                idx = base + i * 8 + j
                cw[j] = (cw[j] << 1) | bits[idx]
                mw[j, 31 - i] = mag[idx]
        for j in range(8):
            word, nerr = _bch_nb(cw[j], bch_arr)
            status = nerr
            if nerr < 0 and chase:
                w2, m = _chase_nb(cw[j], mw[j], L, bch_arr)
                if m >= 0.0:
                    word = w2
                    status = 3
                    nchase += 1
                else:
                    nfail += 1
            elif nerr < 0:
                nfail += 1
            elif nerr > 0:
                ncorr += 1
            s = 0.0
            for k in range(32):
                s += mw[j, k]
            margins[w] = s / 32.0
            rw = ((rev8[word & 0xFF] << 24) |
                  (rev8[(word >> 8) & 0xFF] << 16) |
                  (rev8[(word >> 16) & 0xFF] << 8) |
                  (rev8[(word >> 24) & 0xFF]))
            rw = (rw & 0x001FFFFF) ^ 0x001FFFFF
            out_words[w] = rw
            statuses[w] = status
            w += 1
    return out_words, statuses, margins, nfail, ncorr, nchase
