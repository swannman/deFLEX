#!/usr/bin/env python3
"""Shared, protocol-NEUTRAL decode core for the paging receivers.

Both the FLEX decoder (flex_core.py) and the POCSAG decoder (pocsag_core.py)
are different radio protocols, but they lean on the exact same two pieces of math:

  1. BCH(31,21) forward error correction. Both protocols protect their codewords
     with the identical BCH(31,21) code (generator polynomial 0x769). 21 data
     bits are guarded by 10 parity bits to form a 31-bit codeword (FLEX adds a
     32nd even-parity bit; POCSAG's 32nd bit is also an even-parity bit). ANY
     received word maps -- via its "syndrome" -- to the nearest valid codeword,
     correcting up to 2 flipped bits with no retransmission. The syndrome is the
     remainder of polynomial division by the generator; it is LINEAR over XOR,
     which lets _build_bch_table precompute every syndrome->error mapping once
     instead of searching at decode time. bch3121_chase goes further: it trades
     CPU for reach using soft (confidence) information to correct >2 hard errors
     when the extra errors land on low-confidence bits.

  2. english_score: a 0..1 readability heuristic used as the trust gate on a
     decoded alphanumeric body. It is protocol-independent -- it only looks at
     the recovered ASCII text -- so the FLEX and POCSAG paths apply the same gate
     to decide whether a page is real or FEC-laundered noise.

Nothing in here knows about FLEX framing or POCSAG batches; it is pure
numpy/Python with no numba dependency. Protocol-specific acceleration (e.g. the
FLEX deinterleave njit kernels in flex_numba.py) lives with its protocol.
"""
import numpy as np


# ---- bit utilities --------------------------------------------------------
def popcount(x):
    return bin(x & 0xFFFFFFFF).count("1")


# ---- BCH(31,21) -----------------------------------------------------------
# Forward error correction. BCH(31,21) means: 21 data bits are protected by 10
# parity bits to make a 31-bit codeword (the protocols add a 32nd even-parity
# bit). The magic is that ANY received 31-bit word maps -- via its "syndrome" --
# to the nearest valid codeword, automatically correcting up to 2 flipped bits
# with no retransmission. The syndrome is just the remainder of polynomial
# division by the generator BCH_POLY (0x769); crucially it is LINEAR over XOR,
# which is what lets _build_bch_table precompute every syndrome->error mapping
# once instead of searching at decode time. This identical code protects both
# FLEX and POCSAG. For words still broken after 2-bit correction, bch3121_chase
# trades CPU for reach using soft (confidence) information.
BCH_N, BCH_K, BCH_POLY = 31, 21, 0x769
def _even_parity(x):
    return popcount(x) & 1
def _syndrome(data):
    syn = data >> 1
    mask = 1 << (BCH_N - 1)
    coeff = BCH_POLY << (BCH_K - 1)
    n = BCH_K
    while n > 0:
        if syn & mask:
            syn ^= coeff
        mask >>= 1; coeff >>= 1; n -= 1
    if _even_parity(data):
        syn |= (1 << (BCH_N - BCH_K))
    return syn
# Precompute syndrome -> error-pattern table. The BCH syndrome is linear over
# XOR (polynomial remainder + parity bit are both linear), so for any received
# word w with S(w)=s, the weight<=2 error e satisfies S(e)=s. One table lookup
# replaces the O(32^2) brute-force search -> ~1000x faster, enabling cheap sweeps.
def _build_bch_table():
    tbl = {}
    for i in range(32):
        e = 1 << i
        tbl.setdefault(_syndrome(e), e)
        for j in range(i + 1, 32):
            e2 = e | (1 << j)
            tbl.setdefault(_syndrome(e2), e2)
    return tbl
_BCH_TBL = _build_bch_table()

def bch3121(data):
    """returns (corrected_word, nerr) ; nerr in {0,1,2,-1(fail)}"""
    s = _syndrome(data)
    if not s:
        return data, 0
    e = _BCH_TBL.get(s)
    if e is None:
        return data, -1
    return data ^ e, popcount(e)

def bch3121_chase(word, mag32, L=4):
    """Chase-II soft-decision decode. word=32-bit hard-sliced received word;
    mag32[b]=reliability (|soft|) of integer-bit b (b=0..31). Flip every subset
    of the L least-reliable bits, hard-decode each test pattern, and keep the
    valid codeword with the smallest reliability-weighted distance to `word`
    (sum of mag over differing bits) -- the analog-weight / max-correlation
    metric. Corrects >2 hard errors when the extra errors fall on low-confidence
    bits, which is exactly what kills long ALN messages under hard BCH.
    Returns (codeword, metric) or (word, -1) on failure."""
    order = np.argsort(mag32)[:L]            # least-reliable integer-bit indices
    best_cw = None; best_d = None
    for combo in range(1 << L):
        t = word
        for bi in range(L):
            if combo & (1 << bi):
                t ^= (1 << int(order[bi]))
        cw, nerr = bch3121(t)
        if nerr < 0:
            continue
        diff = cw ^ word
        d = 0.0
        b = 0
        while diff:
            if diff & 1:
                d += mag32[b]
            diff >>= 1; b += 1
        if best_d is None or d < best_d:
            best_d = d; best_cw = cw
    if best_cw is None:
        return word, -1
    return best_cw, best_d


# ---- readability gate -----------------------------------------------------
# A 0..1 score for "does this decoded body look like real text" -- the trust
# gate both protocols apply to an alphanumeric page. It only inspects recovered
# ASCII, so it is protocol-independent. This is a strategy that fails on numeric
# (digits can't be validated this way), but alpha can, which is why it is only
# applied to alphanumeric pages.
_VOWELS = frozenset(b"aeiouAEIOUy")
_PUNCT  = frozenset(b".,:;/-()#@%+*=?!'\" \t\n")

def english_score(body):
    n = len(body)
    if n == 0:
        return 0.0
    letters = digits = spaces = punct = ctrl = vowels = 0
    run = maxrun = 0
    for c in body:
        if 65 <= c <= 90 or 97 <= c <= 122:
            letters += 1
            if c in _VOWELS:
                vowels += 1; run = 0
            else:
                run += 1
                if run > maxrun: maxrun = run
        else:
            run = 0
            if 48 <= c <= 57: digits += 1
            elif c == 32: spaces += 1
            elif c in _PUNCT: punct += 1
            elif c < 32 or c == 127: ctrl += 1
    # Letters and spaces are the backbone of real text; digits and a little
    # punctuation are legitimate (dates, codes) but capped so a punctuation
    # soup like '&"Q$2i{Cf' can't masquerade as a message.
    core = (letters + spaces) / n
    extra = min((digits + punct) / n, 0.30)
    cpen  = ctrl / n
    # vowel ratio among letters: English ~0.38; pager codes run lean (call
    # signs, "NOC", "PCB") so accept a wide band, only punish extremes.
    if letters >= 3:
        vr = vowels / letters
        vscore = 1.0 if 0.18 <= vr <= 0.55 else max(0.0, 1.0 - (min(abs(vr-0.18), abs(vr-0.55)) / 0.20))
    else:
        vscore = 0.3
    # long consonant runs are the strongest noise tell (random letters pile up
    # 6-10 consonants; English almost never exceeds 4).
    rpen = max(0.0, (maxrun - 4)) * 0.20
    s = core + 0.5 * extra - 2.5 * cpen - rpen
    s *= (0.4 + 0.6 * vscore)
    return max(0.0, min(1.0, s))
