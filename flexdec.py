#!/usr/bin/env python3
"""From-scratch FLEX decoder (numpy/scipy), built from the TIA-1500 protocol
tables in gr-pager. Front-end + faithful sign-based sync state machine, but with
a *switchable* 4-level slicer so we can A/B the legacy fixed-threshold approach
against per-frame adaptive level estimation and BCH-gated output.

Usage: flexdec.py <cfile> [slicer] [--diag]
  slicer = fixed | perframe   (default perframe)

=============================================================================
A PRIMER FOR THE READER WHO IS A SOFTWARE ENGINEER BUT NOT AN RF ENGINEER
=============================================================================
FLEX is the Motorola paging protocol (spec TIA-1500) that carries the messages
on the 929-932 MHz pager band. This decoder turns a recording of raw radio
samples into the text of those pages, in pure numpy/scipy. If pocsagdec.py is
the gentle introduction, this is the graduate course -- FLEX is markedly harder
than POCSAG, and most of the cleverness here exists to claw real pages out of
WEAK, fading carriers. Read pocsagdec.py's primer first; the radio fundamentals
(FSK, I/Q samples, FM demodulation, symbol timing, BCH error correction) are
explained there and assumed here.

What makes FLEX harder than POCSAG:

1. FOUR levels, not two. POCSAG is 2-FSK (one bit per symbol). FLEX's richest
   modes are 4-FSK: the carrier is shifted to one of FOUR frequencies
   (-4800, -1600, +1600, +4800 Hz; see FLEX_TONES), so each symbol carries TWO
   bits. A 4-level signal has three decision boundaries instead of one, and the
   four levels are NOT symmetric in practice, which is why we estimate them
   adaptively (two_means / kmeans4) instead of using fixed thresholds.

2. MULTIPLE modes on the same channel. FLEX runs at 1600 OR 3200 baud, 2 OR 4
   level, and announces which via a sync word at the start of every frame
   (FLEX_MODES). The 3200-baud modes interleave two bit-streams onto alternating
   symbols, which we must de-multiplex (demux_phases).

3. STRICT framing. FLEX frames are rigidly periodic: one every 1.875 s, i.e.
   exactly 30000 samples at our 16 kHz symbol-clock rate (FRAME_PERIOD), 88
   words per frame. That periodicity is a gift -- once we know the phase of the
   frame grid we can predict where EVERY frame sits, even ones too faded to
   detect on their own (add_comb_frames).

4. INTERLEAVING. FLEX doesn't send a codeword's 32 bits consecutively; it
   spreads 8 codewords across a 256-bit block in a fixed interleave pattern, so
   that a burst of noise damages one bit in each of 8 words (all individually
   correctable) instead of wiping out one whole word (uncorrectable). We undo
   that shuffle in deinterleave_decode before BCH.

5. ERROR CORRECTION is the same BCH(31,21) as POCSAG (this is why pocsagdec can
   import it), plus optional Chase soft-decision decoding for words that fail
   hard decode.

THE DETECTOR HAS THREE "TIERS" (selectable; we A/B them honestly):
  * The default FM-DISCRIMINATOR path: FM-demodulate to a 1-D wiggle, then slice
    its amplitude into 4 levels. Simple and robust; what gr-pager / multimon do.
  * Tier 1, the MATCHED-FILTER BANK (mf_bank_mag, --mfbank): instead of
    demodulating, correlate the complex signal against each of the 4 expected
    tones over each symbol and pick the strongest. This is the textbook OPTIMAL
    noncoherent detector for orthogonal FSK and squeezes out marginally weaker
    pages.
  * Tier 2, a per-frame CARRIER-OFFSET NULL (est_cfo): re-centre each frame's
    tones to cancel residual tuning error.
  * Tier 3, COHERENT detection (coherent_metric, --coh): track the carrier phase
    and exploit it. EMPIRICALLY THIS LOSES for FLEX (documented at length in
    coherent_metric) -- kept only as an instructive dead-end.

ACQUISITION (finding frames) has two strategies:
  * FlexSync: a faithful port of the classic shift-register sync state machine
    (matches gr-pager bit-for-bit). Locks the cleaner frames.
  * corr_frames: a matched correlator against the known 64-bit sync word, which
    integrates the whole word and so survives many bit errors -- far better on
    weak carriers. Combined with grid_phase + add_comb_frames it recovers frames
    the state machine never sees.

OUTPUT is confidence-graded, never hard-gated: every page is emitted but tagged
A/B/C/D by how much the FEC had to work and (for alpha pages) whether the body
reads as English (english_score). The caller decides what to trust.

The `main()` at the bottom is a research harness -- it wires all the above
together behind command-line flags so we can A/B any combination against a
frozen capture. The live receivers import the building blocks directly.
"""
import sys
from math import gcd
import numpy as np
from scipy import signal

# ---------------------------------------------------------------------------
# PROTOCOL + SAMPLE-RATE CONSTANTS
# ---------------------------------------------------------------------------
# Two sample rates live in this file. SAMP (250 kHz) is the rate of the complex
# I/Q the SDR captured -- we keep the front end at this rate. FS (16 kHz) is the
# lower "symbol-clock" rate we resample the demod to, chosen because it is a
# clean multiple of both baud rates (16000/1600 = 10, 16000/3200 = 5 samples per
# symbol) and is what gr-pager used, so our timing math matches the reference.
SAMP = 250000                    # I/Q capture rate (Hz)
FS = 16000                       # symbol-clock sample rate (gr-pager uses this)
SPB = 10                         # samples/baud @ 1600 baud (= FS/1600)
DEVIATION = 4800.0               # outermost FSK tone offset from center (Hz)
# Every FLEX frame starts with a 64-bit sync word = code_hi(16) | SYNC_MARKER(32)
# | code_lo(16). The fixed middle 32 bits (SYNC_MARKER) flag "a frame begins
# here"; the surrounding code bits identify the MODE (baud/levels). We detect a
# frame by matching the marker, then read the mode from the code bits.
SYNC_MARKER = 0xA6C6AAAA
MASK64 = (1 << 64) - 1
# Each entry: (mode sync code A-word, baud rate, number of FSK levels). The
# decoder learns which one is on air from the sync word rather than being told.
# (sync_A_word, baud, levels)
FLEX_MODES = [
    (0x870C78F3, 1600, 2),
    (0xB0684F97, 1600, 4),
    (0x7B1884E7, 3200, 2),
    (0xDEA0215F, 3200, 4),
    (0x4C7CB383, 3200, 4),
]
FLEX_BCD = "0123456789 U -]["    # numeric-page symbol table: 16 entries, index 0-15
# Page content types carried in the vector word; we ultimately care about the
# alpha ones (SPN/ALN). ENC=encrypted, TON=tone-only, NUM/NNM=numeric, BIN=binary.
PAGE_DESC = ["ENC", "UNK", "TON", "NUM", "SPN", "ALN", "BIN", "NNM"]
FRAME_WORDS = 88                 # a fully-decoded frame is 88 datawords
FRAME_PERIOD = 30000             # exact FLEX frame length: 1.875 s @ 16 kHz

# ---- bit utilities --------------------------------------------------------
def popcount(x):
    return bin(x & 0xFFFFFFFF).count("1")

_REV8 = [int('{:08b}'.format(i)[::-1], 2) for i in range(256)]
def reverse_bits32(v):
    return (_REV8[v & 0xFF] << 24) | (_REV8[(v >> 8) & 0xFF] << 16) | \
           (_REV8[(v >> 16) & 0xFF] << 8) | (_REV8[(v >> 24) & 0xFF])

# ---- BCH(31,21) -----------------------------------------------------------
# Forward error correction. BCH(31,21) means: 21 data bits are protected by 10
# parity bits to make a 31-bit codeword (FLEX adds a 32nd even-parity bit). The
# magic is that ANY received 31-bit word maps -- via its "syndrome" -- to the
# nearest valid codeword, automatically correcting up to 2 flipped bits with no
# retransmission. The syndrome is just the remainder of polynomial division by
# the generator BCH_POLY (0x769); crucially it is LINEAR over XOR, which is what
# lets _build_bch_table precompute every syndrome->error mapping once instead of
# searching at decode time. This identical code protects POCSAG too (hence the
# import in pocsagdec.py). For words still broken after 2-bit correction,
# bch3121_chase trades CPU for reach using soft (confidence) information.
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

# Dense array form of the syndrome->error table for the njit hot-path core
# (flexdec_numba). BCH syndrome is 11 bits; 0 = no correctable error (a valid
# 1/2-bit pattern is nonzero, and a nonzero syndrome never arises from no-error,
# so 0 is an unambiguous miss). Falls back to pure Python if numba is missing.
_BCH_ARR = np.zeros(1 << (BCH_N - BCH_K + 1), dtype=np.int64)
for _s, _e in _BCH_TBL.items():
    _BCH_ARR[_s] = _e
_REV8_ARR = np.array(_REV8, dtype=np.int64)
try:
    import os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from flexdec_numba import deint_decode_core as _deint_core
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

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

# ---- front end ------------------------------------------------------------
# The front end turns a raw I/Q file into the 16 kHz demod (1-D wiggle) the rest
# of the decoder slices into symbols. Three steps: (1) load_baseband selects ONE
# channel out of the capture -- it can shift any carrier within the captured band
# down to DC ("de-rotate by cfo"), low-pass to isolate it, and resample odd input
# rates to our internal 250 kHz; (2) demod_from_baseband FM-demodulates and
# resamples to 16 kHz; (3) an optional one-symbol "integrate-and-dump" matched
# filter (a boxcar average) maximises signal-to-noise at the symbol centre.
def load_baseband(cfile, cfo=0.0, lpf=12000.0, in_rate=None):
    """Read complex I/Q, optional CFO de-rotation, channel LPF. Returns the
    band-limited complex baseband (used both for FM demod and the MF bank).

    `in_rate` is the file's sample rate. If it differs from the internal working
    rate SAMP (250k), the channel is de-rotated to DC AT THE INPUT RATE first,
    then polyphase-decimated down to SAMP -- so the rest of the (validated)
    pipeline runs at exactly 250k regardless of capture bandwidth. This is how a
    2.646 MS/s 7-carrier wideband file is decoded one carrier at a time:
    --carrier is the offset (Hz) of the wanted channel from the file's center."""
    x = np.fromfile(cfile, dtype=np.complex64)
    if in_rate is None:
        in_rate = SAMP
    if cfo:
        n = np.arange(len(x))
        x = x * np.exp(-2j * np.pi * cfo / in_rate * n)   # de-rotate at INPUT rate
        cfo = 0.0                                          # already applied
    if in_rate != SAMP:
        g = gcd(int(round(in_rate)), SAMP)
        x = signal.resample_poly(x, SAMP // g, int(round(in_rate)) // g)
    taps = signal.firwin(127, lpf / (SAMP / 2))     # +/- lpf Hz at 250 k
    return signal.lfilter(taps, 1.0, x)

def demod_from_baseband(xb, mf=True, mflen=SPB):
    d = np.angle(xb[1:] * np.conj(xb[:-1]))         # rad/sample @ 250k
    d16 = signal.resample_poly(d, 8, 125)           # 250k*8/125 = 16000
    d16 = d16 * (SAMP / (2 * np.pi) * 3.0 / DEVIATION)   # outer (+/-4800Hz) -> +/-3
    if mf and mflen > 1:
        # integrate-and-dump matched filter over ONE symbol period. mflen must
        # match the actual symbol length in 16 kHz samples (5 for 3200 baud,
        # 10 for 1600); a too-long boxcar averages across symbols and destroys data.
        d16 = np.convolve(d16, np.ones(mflen) / mflen, mode="same")
    return d16

def front_end(cfile, cfo=0.0, mf=True, mflen=SPB, lpf=12000.0, return_baseband=False, in_rate=None):
    xb = load_baseband(cfile, cfo, lpf, in_rate=in_rate)
    d16 = demod_from_baseband(xb, mf, mflen)
    if return_baseband:
        return d16, xb
    return d16

# ---- Tier 1/2: 4-FSK matched-filter bank + carrier null -------------------
FLEX_TONES = np.array([-4800.0, -1600.0, 1600.0, 4800.0])  # ascending -> level 0..3

def est_cfo(xb, c0, c1):
    """Mean instantaneous frequency (Hz) over [c0,c1) of the complex baseband.
    For balanced 4-level FLEX the mean symbol frequency ~0, so this estimates
    the residual carrier offset. Phasor-sum estimator (robust to noise)."""
    c0 = max(1, int(c0)); c1 = min(len(xb), int(c1))
    if c1 - c0 < 2:
        return 0.0
    seg = xb[c0:c1]
    return float(np.angle(np.sum(seg[1:] * np.conj(seg[:-1]))) * SAMP / (2 * np.pi))

def mf_bank_mag(xb, pos16, baud, cfo=0.0, inv=False, complex_out=False):
    """Noncoherent 4-FSK matched-filter bank (Tier 1). For each symbol center
    (in 16 kHz sample units), correlate the 250 kHz complex baseband against
    each of the 4 FLEX tones over one symbol period; return |correlation|
    (nsym,4). This is the optimal noncoherent detector: FLEX's tone spacing
    (3200 Hz) equals the baud rate, so the tones are orthogonal over a symbol.
    `cfo` (Hz, Tier 2) shifts the reference tones to null residual carrier.
    `complex_out` returns the raw complex correlations instead of |.| (the phase
    is the carrier phase at the symbol center -- used by the coherent detector).

    Implementation: the reference phase factorizes. For symbol s, lag l the
    per-sample offset is rel[s,l] = frac[s] + lrel[l], where lrel[l] = l-half is
    identical for every symbol and frac[s] = round(pos[s])-pos[s] is the per-symbol
    rounding residual. So exp(-jw*rel) = exp(-jw*lrel[l]) * exp(-jw*frac[s]) and
    the correlation becomes a single (nsym,L)@(L,4) complex matmul (BLAS) times a
    per-symbol phase, instead of four full (nsym,L) exponentials -- ~9x faster,
    bit-faithful (validated page-identical on the frozen captures). The factoring
    only holds where idx is unclipped; clipped boundary rows fall back to the exact
    per-sample form."""
    pos = pos16 * (SAMP / FS)                        # symbol centers in 250k samples
    L = int(round(SAMP / baud)); half = L // 2
    n0 = np.round(pos).astype(np.int64)
    lrel = (np.arange(L) - half).astype(np.float64)
    idx_un = n0[:, None] + lrel.astype(np.int64)[None, :]
    idx = np.clip(idx_un, 0, len(xb) - 1)
    seg = xb[idx]                                    # (nsym, L) complex
    frac = n0 - pos                                  # (nsym,) rounding residual
    tones = (FLEX_TONES[::-1] if inv else FLEX_TONES) + cfo
    w = 2 * np.pi * tones / SAMP                      # (4,)
    tonemat = np.exp(-1j * np.outer(lrel, w))         # (L,4), computed once
    c = (seg.astype(np.complex128) @ tonemat) * np.exp(-1j * np.outer(frac, w))
    clipped = np.any(idx_un != idx, axis=1)
    if np.any(clipped):                               # exact form on boundary rows
        rel = idx[clipped] - pos[clipped, None]
        segc = seg[clipped]
        for k, f in enumerate(tones):
            c[clipped, k] = np.sum(segc * np.exp(-1j * 2 * np.pi * f / SAMP * rel), axis=1)
    return c.astype(np.complex128) if complex_out else np.abs(c)

def coherent_metric(C, baud, inv=False, alpha=0.05):
    """Tier 3: coherent CPFSK per-symbol phase tracking via a SQUARING loop.

    *** EMPIRICAL VERDICT (2026-05-30): coherent detection LOSES here. ***
    A/B on the 33 synced anchors: noncoherent |C| = 1942 BCH fails / 14 clean pages;
    every coherent variant tried (decision-directed, squaring, squaring+unwrap) =
    ~7800 fails / 1 clean. Reason is fundamental, not a bug: FLEX is integer-h CPFSK
    (tones at odd multiples of pi/symbol -> exactly orthogonal), the regime where
    coherent's edge over noncoherent is smallest (~1 dB) and the phase-tracking
    burden is highest. With ~18% symbol errors on faded frames the carrier-phase
    estimate is built from wrong-tone/weak correlations, so de-rotation flips bits
    instead of cleaning them. This is precisely why pagers use noncoherent FSK. Kept
    as a documented opt-in (--coh) dead-end; the noncoherent MF bank stays default.

    The per-frame CFO null (Tier 2) removes the mean carrier; this tracks the
    residual *carrier phase* symbol-by-symbol and replaces the noncoherent |C|
    with the coherent statistic Re(C * e^{-j*expected_phase}).

    Why squaring (and why the first, decision-directed attempt failed): FLEX
    mode-3 tones (+-4800/+-1600 Hz) at 3200 baud advance the signal phase by
    2*pi*f/baud = {+-3pi,+-pi} per symbol -- all ODD multiples of pi. A
    decision-directed loop must subtract the decided tone's half-symbol phase,
    which differs by pi between tones, so one wrong decision under fading shifts
    the tracked carrier by pi and the loop LATCHES to the wrong branch, globally
    flipping bits (measured: 1942 -> 7772 BCH fails). Squaring sidesteps this:
    the true-tone center-correlation phase is carrier + (n%2)*pi + dphi_k/2, so
    angle(C^2) = 2*carrier + pi for EVERY tone -- completely data-independent. We
    track 2*carrier with a zero-lag forward-backward smoother on the unit phasor
    (bounded, no runaway), halve it, and de-rotate each tone by its exact
    deterministic expected phase. The halving leaves a global +-1 sign on R (the
    carrier-mod-pi ambiguity), resolved downstream by decoding both signs and
    keeping the lower BCH-fail count. Returns R (nsym,4) coherent MF outputs."""
    tones = (FLEX_TONES[::-1] if inv else FLEX_TONES)
    dphi = 2 * np.pi * tones / baud          # exact per-symbol phase advance per tone
    nsym = C.shape[0]
    if nsym == 0:
        return np.zeros((0, 4))
    win = np.argmax(np.abs(C), axis=1)       # noncoherent winner (decision used ONLY here)
    cw = C[np.arange(nsym), win]
    sq = -(cw * cw)                          # angle = 2*carrier (the +pi removed by -1)
    s = sq / np.maximum(np.abs(sq), 1e-12)   # unit phasor; track on the circle (no unwrap)
    f = np.empty(nsym, complex); acc = s[0]
    for n in range(nsym):                    # causal EMA
        acc = (1 - alpha) * acc + alpha * s[n]; f[n] = acc
    b = np.empty(nsym, complex); acc = s[-1]
    for n in range(nsym - 1, -1, -1):        # anti-causal EMA -> zero net lag
        acc = (1 - alpha) * acc + alpha * s[n]; b[n] = acc
    theta = 0.5 * np.unwrap(np.angle(f + b)) # carrier estimate (unwrap 2*carrier, then halve)
    nmod = (np.arange(nsym) % 2) * np.pi     # deterministic cumulative data phase at center
    R = np.empty((nsym, 4))
    for k in range(4):
        R[:, k] = (C[:, k] * np.exp(-1j * (theta + nmod + dphi[k] / 2.0))).real
    return R

# --- 4 levels -> 2 bits, with confidence -----------------------------------
# A 4-level symbol carries 2 bits. FLEX uses a Gray-style map so that the two
# bits have clean physical meanings: bit_a = "is the frequency negative?" (the
# sign / inner-vs-outer split at the midpoint) and bit_b = "is it an OUTER tone?"
# (|freq| large). Decoding to bits is easy; the VALUE here is the "soft" part --
# we also return how CONFIDENT each bit is (mag_a, mag_b = distance from the
# decision boundary). Those confidences are what feed Chase soft-decode later:
# a bit decided right at the boundary is the first one Chase will try flipping.
def mfbank_softbits(mag):
    """4 tone magnitudes -> (bit_a,bit_b,mag_a,mag_b) via max-log soft metrics.
    level 0..3 (ascending freq): bit_a=(level<2)=freq<0; bit_b=(level in {0,3})
    =outer. Reliability is the metric difference between the competing groups."""
    a_neg = np.maximum(mag[:, 0], mag[:, 1])         # bit_a=1 (freq<0)
    a_pos = np.maximum(mag[:, 2], mag[:, 3])         # bit_a=0
    soft_a = a_pos - a_neg                            # >0 -> bit 0
    b_out = np.maximum(mag[:, 0], mag[:, 3])         # outer -> bit_b=1
    b_in = np.maximum(mag[:, 1], mag[:, 2])          # inner -> bit_b=0
    soft_b = b_in - b_out                             # >0 -> bit 0
    return ((soft_a < 0).astype(np.uint8), (soft_b < 0).astype(np.uint8),
            np.abs(soft_a), np.abs(soft_b))

def demux_phases(bit_a, bit_b, mag_a, mag_b, baud):
    """De-multiplex per-symbol bits+reliabilities into the 4 FLEX phases.
    1600 baud: A=bit_a B=bit_b. 3200 baud: even symbols -> A,B ; odd -> C,D."""
    z = np.zeros(len(bit_a), np.uint8); zf = np.full(len(bit_a), 1e3)
    if baud == 1600:
        return [bit_a, bit_b, z, z], [mag_a, mag_b, zf, zf]
    n2 = len(bit_a) - (len(bit_a) % 2)
    ev = slice(0, n2, 2); od = slice(1, n2, 2)
    return ([bit_a[ev], bit_b[ev], bit_a[od], bit_b[od]],
            [mag_a[ev], mag_b[ev], mag_a[od], mag_b[od]])

# ---- adaptive level estimation -------------------------------------------
# To slice the FM-discriminator signal into levels we need to know WHERE the
# levels actually sit. The textbook assumption (symmetric levels at +/-2, +/-6)
# is wrong in practice: tuning offset, filter group delay and FM-discriminator
# nonlinearity make the real four-level constellation asymmetric and drifting.
# So instead of fixed thresholds we LEARN the level centres from the data with
# tiny 1-D k-means: two_means finds the inner/outer split for 2-level decisions,
# kmeans4 finds all four centres for full 4-level slicing. This is the single
# biggest accuracy win over the legacy fixed-threshold slicer.
def two_means(mags, it=20):
    """1-D 2-means on magnitudes -> (low_center, high_center)."""
    lo, hi = np.percentile(mags, 25), np.percentile(mags, 75)
    for _ in range(it):
        mid = (lo + hi) / 2
        a, b = mags[mags <= mid], mags[mags > mid]
        if len(a) == 0 or len(b) == 0:
            break
        nlo, nhi = a.mean(), b.mean()
        if abs(nlo - lo) < 1e-4 and abs(nhi - hi) < 1e-4:
            lo, hi = nlo, nhi; break
        lo, hi = nlo, nhi
    return lo, hi

def kmeans4(v, it=30):
    """1-D 4-means on signed symbol values -> 4 ascending centers.
    No symmetry assumption (the real constellation is asymmetric)."""
    c = np.percentile(v, [12.5, 37.5, 62.5, 87.5]).astype(float)
    for _ in range(it):
        # assign each sample to nearest center
        d = np.abs(v[:, None] - c[None, :])
        lab = d.argmin(1)
        nc = c.copy()
        for k in range(4):
            m = v[lab == k]
            if len(m):
                nc[k] = m.mean()
        nc.sort()
        if np.max(np.abs(nc - c)) < 1e-4:
            c = nc; break
        c = nc
    return c

# ---- continuous-tracking acquisition: sync-word matched correlator --------
# Acquisition (not intra-frame detection) is the bottleneck on weak carriers:
# the hard 3-error shift-register sync only locks a handful of frames when the
# carrier is ~14 dB down. A data-aided matched filter on the KNOWN 64-bit FLEX
# frame-sync word survives many bit errors (it integrates the whole word) and
# yields exactly one sharp peak per frame, so it can't flood false locks.
def build_sync_template(mode, spb=SPB):
    """+/-1 demod template for the 64-bit FLEX frame sync of `mode`, upsampled
    to `spb` samples/bit (1600 baud sync -> spb=10). The 64-bit value is packed
    exactly as FlexSync.test_sync reads it: bits[63:48]=code_high16,
    bits[47:16]=SYNC_MARKER, bits[15:0]=code_low16. MSB = oldest bit (sent
    first). Demod sign convention matches test_sync's neg=(v<0): bit 1 -> -1,
    bit 0 -> +1."""
    code = FLEX_MODES[mode][0]
    val = (((code >> 16) & 0xFFFF) << 48) | (SYNC_MARKER << 16) | (code & 0xFFFF)
    bits = [(val >> (63 - i)) & 1 for i in range(64)]
    signs = np.array([-1.0 if b else 1.0 for b in bits], dtype=float)
    return np.repeat(signs, spb)

def corr_acquire(demod, modes=(0, 1, 2, 3, 4), height=0.45, distance=None):
    """Normalized matched-filter cross-correlation of the demod stream against
    each mode's frame-sync template. Returns a sorted list of
    (position, mode, score) peaks, one per detected frame. `position` is the
    sample index where template[0] (the first sync bit) aligns. Score is the
    normalized correlation (-1..1); near +1 = clean sync."""
    from scipy.signal import find_peaks
    d = np.asarray(demod, dtype=float)
    d2 = d * d
    out = []
    for m in modes:
        tmpl = build_sync_template(m)
        L = len(tmpl)
        num = signal.fftconvolve(d, tmpl[::-1], mode="valid")        # cross-corr
        energy = signal.fftconvolve(d2, np.ones(L), mode="valid")    # sliding SS
        denom = np.sqrt(np.maximum(energy, 1e-12) * L)
        ncc = num / denom
        dist = distance if distance is not None else FRAME_PERIOD - L
        pk, pr = find_peaks(ncc, height=height, distance=max(1, dist))
        for p, h in zip(pk, pr["peak_heights"]):
            out.append((int(p), m, float(h)))
    out.sort()
    return out

def grid_phase(positions):
    """Circular-median frame-grid phase (mod FRAME_PERIOD) from sync positions.
    Robust because every frame sync sits on the same 30000-sample transmitter
    grid; a handful of high-confidence peaks pin the phase for the whole stream."""
    r = np.asarray(positions, float) % FRAME_PERIOD
    ang = r / FRAME_PERIOD * 2 * np.pi
    ph = np.angle(np.mean(np.exp(1j * ang))) / (2 * np.pi) * FRAME_PERIOD
    return ph % FRAME_PERIOD

def corr_frames(demod, p0_offset=1517.0, grid=True, score_hi=0.6, include_thr=0.35):
    """Continuous-tracking acquisition: locate frames via the sync-word matched
    correlator instead of the hard 3-error shift register. High-confidence peaks
    (>score_hi) fix the global frame grid phase; with grid=True we then walk
    EVERY grid slot but admit a frame only where the dominant-mode sync
    correlation clears `include_thr` (a noise floor). This is the key to a weak
    carrier: combing all slots blindly lets Chase force-correct pure noise into
    valid-looking numeric pages (numeric charset is always 'printable', so the
    tier gate can't reject it). Score-gating decodes only slots that actually
    carry a sync, so on a near-idle channel we take just the few real frames
    (matching what a continuous tracker locks) and on a busy carrier we take all
    of them. Returns frame anchor dicts compatible with the decode path
    (syms=None; the MF-bank/FM paths recompute symbol centers from p0/spb/nsyms).
    grid=False (--corr-peaks) instead decodes only the detected high-score peaks."""
    peaks = corr_acquire(demod)
    hi = [(p, m, h) for p, m, h in peaks if h > score_hi]
    if not hi:
        return []
    from collections import Counter
    mode = Counter(m for _, m, _ in hi).most_common(1)[0][0]
    baud = FLEX_MODES[mode][1]; levels = FLEX_MODES[mode][2]
    spb = 16000 // baud
    # +2 symbols of slack so a par=1 timing-sweep shift still yields a full
    # 11-block (88-word) frame; the exact data-field length (baud*1760//1000)
    # leaves no margin and a par=1 shift truncates to 80 words -> parse_frame
    # rejects the frame (this silently zeroed every weak-carrier frame).
    nsyms = baud * 1760 // 1000 + 2
    hpos = [p for p, _, _ in hi]
    ph = grid_phase(hpos)
    N = len(demod)
    if grid:
        # dominant-mode NCC over the whole stream -> per-slot sync presence
        d = np.asarray(demod, float); d2 = d * d
        tmpl = build_sync_template(mode); L = len(tmpl)
        num = signal.fftconvolve(d, tmpl[::-1], mode="valid")
        en = signal.fftconvolve(d2, np.ones(L), mode="valid")
        ncc = num / np.sqrt(np.maximum(en, 1e-12) * L)
        kmax = int((N - 1 - ph) // FRAME_PERIOD)
        anchors = []
        for k in range(max(0, kmax) + 1):
            a = ph + k * FRAME_PERIOD
            ai = int(round(a))
            if 0 <= ai < len(ncc) and ncc[ai] > include_thr:
                anchors.append(a)
    else:
        anchors = sorted(hpos)
    frames = []
    for a in anchors:
        p0 = int(round(a + p0_offset))
        if p0 < 0 or p0 + nsyms * spb >= N - 1:
            continue
        frames.append(dict(mode=mode, baud=baud, levels=levels,
                           p0=p0, nsyms=nsyms, spb=spb, syms=None))
    return frames

# ---- sync state machine ---------------------------------------------------
# The classic acquisition method, ported faithfully from gr-pager so it matches
# the reference bit-for-bit. It is a shift register that walks the demod one
# sample at a time looking for the sync marker; when it sees one it measures the
# bit timing, reads the mode, skips the rest of the sync overhead (SYNC1/SYNC2),
# then records exactly one frame's worth of raw symbol values and goes idle. It
# is precise but UNFORGIVING -- a sync word with too many bit errors is missed
# entirely, which is why corr_frames (the matched-correlator alternative above)
# beats it on weak carriers. The states below are: IDLE (hunting for sync) ->
# SYNCING (inside the sync run) -> SYNC1/SYNC2 (fixed overhead) -> DATA
# (collecting the frame), then reset.
ST_IDLE, ST_SYNCING, ST_SYNC1, ST_SYNC2, ST_DATA = range(5)

class FlexSync:
    def __init__(self, demod, dc_alpha=5e-6):
        self.d = demod
        self.dc = 0.0
        self.alpha = dc_alpha
        self.reset_idle()
        self.frames = []          # list of dict(mode, baud, levels, syms[raw analog])

    def reset_idle(self):
        self.state = ST_IDLE
        self.sync = [0] * SPB
        self.index = 0
        self.start = self.center = self.end = 0
        self.count = 0
        self.mode = 0
        self.baud = 1600
        self.levels = 2
        self.spb = SPB
        self.hibit = False
        self.fiw = 0
        self.cur_syms = None

    def index_avg(self, s, e):
        if s < e:
            return (e + s) // 2
        return ((e + s) // 2 + self.spb // 2) % self.spb

    def test_sync(self, neg):       # neg = (v < 0)  (== sym<2)
        i = self.index
        self.sync[i] = ((self.sync[i] << 1) | (1 if neg else 0)) & MASK64
        val = self.sync[i]
        marker = (val >> 16) & 0xFFFFFFFF
        if popcount(marker ^ SYNC_MARKER) < 4:
            code = (((val >> 32) & 0xFFFF0000) | (val & 0xFFFF)) & 0xFFFFFFFF
            for k, (sy, bd, lv) in enumerate(FLEX_MODES):
                if popcount(code ^ (sy & 0xFFFFFFFF)) < 4:
                    self.mode = k
                    return True
        return False

    def run(self):
        d = self.d
        N = len(d)
        for p in range(N):
            self.dc += self.alpha * (d[p] - self.dc)
            v = d[p] - self.dc
            neg = v < 0
            self.index = (self.index + 1) % self.spb
            st = self.state
            if st == ST_IDLE:
                if self.test_sync(neg):
                    self.start = self.index
                    self.state = ST_SYNCING
            elif st == ST_SYNCING:
                if not self.test_sync(neg):
                    self.end = self.index
                    self.center = self.index_avg(self.start, self.end)
                    self.count = 0
                    self.state = ST_SYNC1
            elif st == ST_SYNC1:
                if self.index == self.center:
                    self.fiw = ((self.fiw << 1) | (1 if v > 0 else 0)) & 0xFFFFFFFFFFFFFFFF
                    self.count += 1
                    if self.count == 48:
                        self.baud = FLEX_MODES[self.mode][1]
                        self.levels = FLEX_MODES[self.mode][2]
                        self.spb = 16000 // self.baud
                        self.count = 0
                        self.state = ST_SYNC2
                        if self.baud == 3200:
                            self.center //= 2
                            self.index = self.index // 2 - self.spb // 2
            elif st == ST_SYNC2:
                if self.index == self.center:
                    self.count += 1
                    if self.count == self.baud // 40:
                        self.count = 0
                        self.cur_syms = []
                        self.cur_p0 = None
                        self.state = ST_DATA
            elif st == ST_DATA:
                if self.index == self.center:
                    if self.cur_p0 is None:
                        self.cur_p0 = p          # abs index of first data center
                    self.cur_syms.append(v)
                    self.count += 1
                    if self.count == self.baud * 1760 // 1000:
                        self.frames.append(dict(mode=self.mode, baud=self.baud,
                                                levels=self.levels, p0=self.cur_p0,
                                                nsyms=self.count, spb=self.spb,
                                                syms=np.array(self.cur_syms)))
                        self.reset_idle()
        return self.frames

# ---- symbol -> phase bits -> datawords ------------------------------------
# The chain in this section: raw symbol values -> hard bits + reliabilities
# (the *_phases / slice_soft sliceres) -> de-multiplex the 3200-baud A/C streams
# into FLEX's 4 "phases" (independent interleaved sub-channels) -> deinterleave
# each phase's 256-bit blocks back into 8 codewords -> BCH-correct -> datawords.
def syms_to_phases(syms, baud, levels, thr, dc):
    v = syms - dc
    pa, pb, pc, pd = [], [], [], []
    if baud == 1600:
        for x in v:
            pa.append(1 if x < 0 else 0)
            pb.append((1 if abs(x) > thr else 0) if levels == 4 else 0)
            pc.append(0); pd.append(0)
    else:
        hibit = False; a=b=c=dd=0
        for x in v:
            if not hibit:
                a = 1 if x < 0 else 0
                b = (1 if abs(x) > thr else 0) if levels == 4 else 0
                hibit = True
            else:
                c = 1 if x < 0 else 0
                dd = (1 if abs(x) > thr else 0) if levels == 4 else 0
                hibit = False
                pa.append(a); pb.append(b); pc.append(c); pd.append(dd)
    return [np.array(p, dtype=np.uint8) for p in (pa, pb, pc, pd)]

def syms_to_phases_centers(syms, baud, levels, centers):
    """4-level slicer using independently-estimated level centers (asymmetric).
    centers ascending -> symbol index 0..3 ; Gray map matches gr-pager:
      sym0(outer-) a=1 b=1 | sym1(inner-) a=1 b=0 | sym2(inner+) a=0 b=0 | sym3(outer+) a=0 b=1
    For 2-level, fall back to sign of (v - midpoint)."""
    if levels == 2:
        mid = (centers[1] + centers[2]) / 2 if len(centers) == 4 else 0.0
        sym = np.where(syms < mid, 1, 0)  # only bit_a meaningful; reuse a-path
        bit_a = (sym >= 1).astype(np.uint8)  # negative -> 1
        bit_b = np.zeros(len(syms), dtype=np.uint8)
    else:
        d = np.abs(syms[:, None] - centers[None, :])
        sym = d.argmin(1)
        bit_a = (sym < 2).astype(np.uint8)
        bit_b = ((sym == 0) | (sym == 3)).astype(np.uint8)
    pa, pb, pc, pd = [], [], [], []
    if baud == 1600:
        return [bit_a, bit_b, np.zeros(len(syms), np.uint8), np.zeros(len(syms), np.uint8)]
    # 3200: de-multiplex even/odd symbols into phase pairs (A,B)=even (C,D)=odd
    ev = slice(0, len(syms) - (len(syms) % 2), 2)
    od = slice(1, len(syms) - (len(syms) % 2) + 1, 2)
    return [bit_a[ev], bit_b[ev], bit_a[od], bit_b[od]]

def slice_soft(syms, baud, levels, thr, dc):
    """4-level slicer that returns both hard phase bits AND per-bit reliability
    magnitudes, de-multiplexed into the 4 FLEX phases. bit_a is the sign bit
    (boundary at dc); bit_b is outer-vs-inner (boundary at +/-thr). Reliability
    is the distance from the relevant decision boundary."""
    v = syms - dc
    av = np.abs(v)
    bit_a = (v < 0).astype(np.uint8)
    mag_a = av
    if levels == 4:
        bit_b = (av > thr).astype(np.uint8)
        mag_b = np.abs(av - thr)
    else:
        bit_b = np.zeros(len(v), np.uint8)
        mag_b = np.full(len(v), 1e3)         # bit_b unused -> max reliability
    return demux_phases(bit_a, bit_b, mag_a, mag_b, baud)

def decode_phases(phases, mags, chase=False):
    """Deinterleave + BCH-decode the 4 phase bit-streams. mags carries per-bit
    reliability for Chase-II soft decoding on hard failures."""
    nf = nc = nch = 0
    wordsets = []
    confsets = []
    for ph, mg in zip(phases, mags):
        if len(ph) < 256:
            wordsets.append(None); confsets.append(None); continue
        keep = (len(ph) // 256) * 256
        words, f_, c_, ch_, conf = deinterleave_decode(
            ph[:keep], mag=mg[:keep], chase=chase)
        nf += f_; nc += c_; nch += ch_
        wordsets.append(words); confsets.append(conf)
    return wordsets, confsets, nf, nc, nch

def deinterleave_decode(bits, mag=None, chase=False, L=4):
    """Undo FLEX's bit interleave, then BCH-decode the recovered codewords.

    WHY interleaving exists: radio errors come in BURSTS (a fade knocks out
    several consecutive bits). BCH can fix only 2 errors per 32-bit word, so a
    burst landing in one word would be fatal. FLEX defends against this by
    SPREADING 8 codewords across a 256-bit block: bit (i*8 + j) of the block is
    bit i of codeword j. Consecutive on-air bits therefore belong to DIFFERENT
    codewords, so a burst becomes one isolated (correctable) error in each of
    several words instead of many errors in one. This function reverses that
    shuffle -- read column j down the 256-bit block to rebuild codeword j -- then
    runs BCH (and Chase on failures). The numba fast path does the identical
    thing in compiled code for the live receiver's throughput.

    bits: uint8 -> (datawords, n_bch_fail, n_corrected, n_chase_recovered, conf).
    conf[k] = (status, margin) aligned to datawords: status 0=clean syndrome,
    1/2=BCH-corrected, 3=Chase-recovered, -1=uncorrectable; margin=mean soft-bit
    reliability for that word, normalized to the per-call median (~1.0 typical).
    The margin is FEC-INDEPENDENT, so a word that "passed" BCH but sits well below
    the local median margin is a likely miscorrection (status can't catch those).
    If chase and mag supplied, words that fail hard BCH are retried with Chase-II."""
    have_mag = mag is not None
    if _HAVE_NUMBA and have_mag:
        b = np.ascontiguousarray(bits, dtype=np.uint8)
        m = np.ascontiguousarray(mag, dtype=np.float64)
        ow, st, mg, nfail, ncorr, nchase = _deint_core(
            b, m, bool(chase), int(L), _BCH_ARR, _REV8_ARR)
        out = ow.tolist()
        conf = [(int(st[k]), float(mg[k])) for k in range(st.shape[0])]
        mvals = [mm for (_, mm) in conf if mm == mm]
        med = float(np.median(mvals)) if mvals else 0.0
        if med > 0:
            conf = [(s, (mm / med if mm == mm else float("nan"))) for (s, mm) in conf]
        return out, int(nfail), int(ncorr), int(nchase), conf
    out = []
    conf = []
    nfail = ncorr = nchase = 0
    nblocks = len(bits) // 256
    for blk in range(nblocks):
        cw = [0] * 8
        mw = [np.zeros(32) for _ in range(8)] if have_mag else None
        base = blk * 256
        for i in range(32):
            for j in range(8):
                idx = base + i * 8 + j
                cw[j] = (cw[j] << 1) | int(bits[idx])
                if have_mag:
                    mw[j][31 - i] = mag[idx]      # integer-bit b = 31 - i
        for j in range(8):
            word, nerr = bch3121(cw[j])
            status = nerr
            if nerr < 0 and chase:
                w2, m = bch3121_chase(cw[j], mw[j], L)
                if m is not None and m >= 0:
                    word = w2; status = 3; nchase += 1
                else:
                    nfail += 1
            elif nerr < 0:
                nfail += 1
            elif nerr > 0:
                ncorr += 1
            margin = float(np.mean(mw[j])) if have_mag else float("nan")
            conf.append((status, margin))
            word = reverse_bits32(word)
            word = (word & 0x001FFFFF) ^ 0x001FFFFF
            out.append(word)
    if have_mag:
        mvals = [m for (_, m) in conf if m == m]
        med = float(np.median(mvals)) if mvals else 0.0
        if med > 0:
            conf = [(s, (m / med if m == m else float("nan"))) for (s, m) in conf]
    return out, nfail, ncorr, nchase, conf

# ---- language-likeness gate (alpha pages only) ----------------------------
# A printable-ratio gate is too weak: aggressive Chase FEC can manufacture
# noise frames that are 85%+ "printable" yet are clearly not text. Real pager
# alpha bodies are English-ish: letters/spaces dominate, vowels appear in the
# expected proportion, consonant runs are short, and control chars are rare.
# Random 7-bit noise fails ALL of those even when it scrapes past pr>=0.85.
# This is precisely "a strategy that fails on numeric" -- digits can't be
# validated this way, but alpha can, which is why we only run it under --alpha.
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

# ---- frame -> pages (port of flex_frame::parse) ---------------------------
# Now we have 88 clean datawords; this section extracts the actual pages. A FLEX
# frame is self-describing, like a tiny filesystem:
#   * word 0 is the Block Info Word (BIW): it says where the address field ends
#     and the vector field begins (aoffset / voffset).
#   * the ADDRESS field lists capcodes (recipient IDs); parse_capcode rebuilds
#     each, handling both short and long address forms.
#   * for each address there is a paired VECTOR word that says what TYPE the page
#     is (alpha / numeric / tone) and WHERE its message words live in the frame.
#   * parse_alpha / parse_numeric then walk those message words and unpack the
#     characters (alpha = 7-bit ASCII; numeric = 4-bit BCD via FLEX_BCD).
# parse_frame ties it together and attaches a per-page confidence (pconf) built
# from the FEC status of exactly the words that page touched.
def is_alpha(t): return t in (4, 5)      # SPN, ALN
def is_numeric(t): return t in (3, 7)    # NUM, NNM
def is_tone(t): return t == 2

def parse_capcode(aw1, aw2):
    laddr = (aw1 < 0x008001) or (aw1 > 0x1E0000) or (aw1 > 0x1E7FFE)
    if laddr:
        cap = aw1 + ((aw2 ^ 0x001FFFFF) << 15) + 0x1F9000
    else:
        cap = aw1 - 0x8000
    return laddr, cap

def parse_alpha(frame, mw1, mw2, j, laddr):
    if not laddr:
        frag = (frame[mw1] >> 11) & 0x03
        mw1 += 1
    else:
        frag = (frame[j + 1] >> 11) & 0x03
        mw2 -= 1
    chars = []
    def add(c):
        c &= 0x7F
        if c != 0x03:
            chars.append(c)
    for i in range(mw1, mw2 + 1):
        dw = frame[i]
        if i > mw1 or frag != 0x03:
            add(dw)
        add(dw >> 7)
        add(dw >> 14)
    return bytes(chars)

def parse_numeric(frame, mw1, mw2, j, laddr, ptype):
    if not laddr:
        dw = frame[mw1]; mw1 += 1; mw2 += 1
    else:
        dw = frame[j + 1]
    digit = 0
    count = 4 + (10 if ptype == 7 else 2)
    out = []
    for i in range(mw1, mw2 + 1):
        for _ in range(21):
            digit = (digit >> 1) & 0x0F
            if dw & 1:
                digit ^= 0x08
            dw >>= 1
            count -= 1
            if count == 0:
                if digit != 0x0C:
                    out.append(ord(FLEX_BCD[digit]))
                count = 4
        dw = frame[i]
    return bytes(out)

def parse_frame(frame, conf=None):
    pages = []

    def pconf(indices):
        """Per-page confidence over the words this page reads. Returns
        (status_counts, min_margin) -- worst word dominates a page, so we report
        the minimum normalized margin and the count of failed/corrected words."""
        if conf is None:
            return None
        sts = []
        mgs = []
        for k in indices:
            if 0 <= k < len(conf):
                s, m = conf[k]
                sts.append(s)
                if m == m:
                    mgs.append(m)
        if not sts:
            return None
        return dict(words=len(sts),
                    verified=sum(1 for s in sts if s == 0),
                    corrected=sum(1 for s in sts if s in (1, 2)),
                    chase=sum(1 for s in sts if s == 3),
                    failed=sum(1 for s in sts if s < 0),
                    margin=(min(mgs) if mgs else float("nan")))

    if len(frame) != FRAME_WORDS:
        return pages
    biw = frame[0]
    if biw == 0 or biw == 0x001FFFFF:
        return pages
    voffset = (biw >> 10) & 0x3F
    aoffset = ((biw >> 8) & 0x03) + 1
    i = aoffset
    while i < voffset:
        j = voffset + i - aoffset
        if frame[i] in (0x00000000, 0x001FFFFF):
            i += 1; continue
        laddr, cap = parse_capcode(frame[i], frame[i + 1])
        if laddr:
            i += 1
        if cap < 0:
            i += 1; continue
        if j >= len(frame):
            i += 1; continue
        viw = frame[j]
        ptype = (viw >> 4) & 0x07
        mw1 = (viw >> 7) & 0x7F
        ln = (viw >> 14) & 0x7F
        if is_numeric(ptype):
            ln &= 0x07
        mw2 = mw1 + ln
        if mw1 == 0 and mw2 == 0:
            i += 1; continue
        if is_tone(ptype):
            mw1 = mw2 = 0
        if mw1 > 87 or mw2 > 87:
            i += 1; continue
        try:
            if is_alpha(ptype):
                body = parse_alpha(frame, mw1, mw2 - 1, j, laddr)
            elif is_numeric(ptype):
                body = parse_numeric(frame, mw1, mw2, j, laddr, ptype)
            elif is_tone(ptype):
                body = b""
            else:
                body = b""
        except (IndexError, ):
            body = b"<oob>"
        idxs = [j] if is_tone(ptype) else [j] + list(range(mw1, mw2 + 1))
        pages.append((cap, PAGE_DESC[ptype], body, pconf(idxs)))
        i += 1
    return pages

# ---- frame-comb extrapolation ---------------------------------------------
def add_comb_frames(frames, total_len, verbose=False):
    """FLEX frames are strictly periodic (1.875 s = 30000 samples @ 16 kHz).
    The sync state machine only locks the cleaner frames; but every frame sits
    on the same comb. Fit p0 = a + b*k to the reliably-synced anchors (b absorbs
    sample-clock drift), then synthesize the MISSED slots so they go through the
    same fractional-timing + parity sweep + BCH-gated decode. This is how a real
    receiver recovers fading frames it couldn't acquire from scratch."""
    from collections import Counter
    anchors = [f for f in frames if f.get("syms") is not None]
    if len(anchors) < 2:
        return frames
    p0 = np.array([f["p0"] for f in anchors], float)
    order = np.argsort(p0); p0 = p0[order]
    # Slot assignment uses the SPEC period (1.875 s -> 30000 samp @ 16 kHz);
    # this is known from TIA-1500, not estimated. The fit only refines the
    # receiver's sample-clock drift (b) and absolute phase (a). If sparse/noisy
    # anchors push b far from spec, fall back to the spec period exactly.
    k = np.round((p0 - p0[0]) / FRAME_PERIOD)
    A = np.vstack([np.ones_like(k), k]).T
    (a, b), *_ = np.linalg.lstsq(A, p0, rcond=None)
    if abs(b - FRAME_PERIOD) > 5.0:
        b = float(FRAME_PERIOD)
        a = float(np.mean(p0 - b * k))
    mode = Counter(f["mode"] for f in anchors).most_common(1)[0][0]
    tmpl = next(f for f in anchors if f["mode"] == mode)
    baud, levels, spb, nsyms = tmpl["baud"], tmpl["levels"], tmpl["spb"], tmpl["nsyms"]
    span = nsyms * spb
    have = {int(x) for x in k}
    # The comb is bidirectional: anchors can cluster late in the capture, but the
    # same periodic grid extends BACKWARD to t=0. Extrapolate both directions so
    # frames before the earliest anchor (kk<0) are decoded too -- otherwise a weak
    # carrier whose only clean sync words land late strands all earlier frames.
    kmin = int(np.floor((0.0 - a) / b))
    kmax = int(round((total_len - a - span) / b))
    out = list(frames); added = 0
    for kk in range(kmin, kmax + 1):
        if kk in have:
            continue
        pred = a + b * kk
        if pred < 0 or pred + span >= total_len - 1:
            continue
        out.append(dict(mode=mode, baud=baud, levels=levels, p0=pred,
                        nsyms=nsyms, spb=spb, syms=None, synth=True))
        added += 1
    if verbose:
        print(f"[comb] period b={b:.1f} samp ({b/FS*1000:.2f} ms), phase a={a:.1f}, "
              f"anchors={len(anchors)} synthesized={added} -> {len(out)} slots "
              f"(of ~{kmax-kmin+1} possible, kmin={kmin})")
    return out

# ---- driver ---------------------------------------------------------------
# RESEARCH HARNESS, not the production entry point. main() wires every option
# above behind command-line flags so we can A/B any combination (FM-disc vs
# matched-filter bank, hard-sync vs correlator vs comb, fixed vs adaptive slicer,
# Chase on/off, coherent on/off) against a frozen capture and compare decode
# counts. The live receivers (flex_inmem_mp.py etc.) instead import the building
# blocks directly. The big decision loop at the end grades every page A/B/C/D by
# how hard the FEC worked and whether its body reads as English -- it never hard-
# drops a page, it labels confidence and lets the caller choose.
def main():
    cfile = sys.argv[1] if len(sys.argv) > 1 else "/tmp/flex_ab/iq_929612500_250k.cfile"
    slicer = "perframe"
    diag = "--diag" in sys.argv
    for a in sys.argv[2:]:
        if a in ("fixed", "perframe", "k4"):
            slicer = a

    mflen = SPB
    lpf = 12000.0
    carrier = 0.0          # channel-select offset (Hz) within the +-fs/2 capture band
    in_rate = None         # input file sample rate (None -> assume SAMP=250k)
    center_mhz = 929.6125  # display-only: capture center freq for labelling carriers
    for a in sys.argv:
        if a.startswith("--mflen="):
            mflen = int(a.split("=", 1)[1])
        if a.startswith("--lpf="):
            lpf = float(a.split("=", 1)[1])
        if a.startswith("--carrier="):
            carrier = float(a.split("=", 1)[1])
        if a.startswith("--samp-rate="):
            in_rate = float(a.split("=", 1)[1])
        if a.startswith("--center="):
            center_mhz = float(a.split("=", 1)[1])
    mfbank = "--mfbank" in sys.argv             # Tier 1: 4-FSK matched-filter bank
    nocfo = "--nocfo" in sys.argv               # disable Tier 2 per-frame carrier null
    coh = "--coh" in sys.argv                   # Tier 3: coherent per-symbol phase track
    coh_alpha = 0.05
    for a in sys.argv:
        if a.startswith("--coh-alpha="):
            coh_alpha = float(a.split("=", 1)[1])
    inv = "--inv" in sys.argv                   # reverse tone->level map (polarity)
    xb = None
    # --carrier shifts a neighbouring FLEX channel down to baseband (load_baseband
    # de-rotates by `carrier` Hz; the existing LPF then isolates it, pushing the
    # original centre carrier out of band). Lets us decode every channel that falls
    # within the +-125 kHz of the SAME benchmark IQ -- a true multi-carrier decode
    # with no new capture. The sync state machine auto-detects the channel's mode.
    if mfbank:
        demod, xb = front_end(cfile, cfo=carrier, mf=("--mf" in sys.argv), mflen=mflen,
                              lpf=lpf, return_baseband=True, in_rate=in_rate)
    else:
        demod = front_end(cfile, cfo=carrier, mf=("--mf" in sys.argv), mflen=mflen, lpf=lpf,
                          in_rate=in_rate)
    demod = demod - np.median(demod)            # global DC/CFO removal
    corr = "--corr" in sys.argv                  # continuous-tracking correlator acquisition
    corr_grid = "--corr-peaks" not in sys.argv   # comb all grid slots (vs only detected peaks)
    corr_off = 1517.0
    alpha_only = "--alpha" in sys.argv          # optimize for ALN/SPN: drop NUM/NNM
    # alpha-only lets us comb aggressively: garbage numeric pages (which always
    # look 'printable') would otherwise flood the tiers, but we discard them, and
    # garbage alpha is self-evidently non-English (caught by the printable gate).
    corr_thr = -1.0 if alpha_only else 0.35
    for a in sys.argv:
        if a.startswith("--corr-off="):
            corr_off = float(a.split("=", 1)[1])
        if a.startswith("--corr-thr="):
            corr_thr = float(a.split("=", 1)[1])
    if corr:
        frames = corr_frames(demod, p0_offset=corr_off, grid=corr_grid,
                             include_thr=corr_thr)
    else:
        fs = FlexSync(demod)
        frames = fs.run()
    n_anchor = len(frames)
    if "--comb" in sys.argv:
        frames = add_comb_frames(frames, len(demod), verbose=True)

    _syms_list = [f["syms"] for f in frames if f.get("syms") is not None]
    all_syms = np.concatenate(_syms_list) if _syms_list else np.array([])
    if diag and len(all_syms):
        v = all_syms - np.median(all_syms)
        lo, hi = two_means(np.abs(v))
        print(f"[diag] frames={len(frames)} datasyms={len(all_syms)}")
        print(f"[diag] |v| inner~{lo:.3f} outer~{hi:.3f} -> adaptive thr~{(lo+hi)/2:.3f} (legacy fixed=2.0)")
        for q in (1, 5, 25, 50, 75, 95, 99):
            print(f"[diag]   pctl {q:2d}: v={np.percentile(v,q):+.3f}")
        hist, edges = np.histogram(v, bins=40, range=(-6, 6))
        peak = hist.max()
        for h, e in zip(hist, edges):
            bar = "#" * int(60 * h / peak)
            print(f"[hist] {e:+5.1f} | {bar}")

    xp = np.arange(len(demod), dtype=float)

    def sample_at(positions):
        return np.interp(positions, xp, demod)

    def decode_syms(syms, baud, levels, chase=False):
        # FM-discriminator slicer path (produces hard bits + reliabilities)
        if slicer == "fixed":
            thr, dc = 2.0, 0.0
        else:
            dc = np.median(syms)
            lo, hi = two_means(np.abs(syms - dc))
            thr = (lo + hi) / 2
        phases, mags = slice_soft(syms, baud, levels, thr, dc)
        return decode_phases(phases, mags, chase=chase)

    def decode_mf(pos16, baud, levels, chase=False, use_coh=False):
        # Tier 1/2/3 path: 4-FSK matched-filter bank on the complex baseband at the
        # given symbol centers (16 kHz units), with per-frame carrier null (Tier 2)
        # and optional coherent per-symbol phase tracking (Tier 3).
        cfo = 0.0 if nocfo else est_cfo(xb, pos16[0] * SAMP / FS,
                                        pos16[-1] * SAMP / FS)
        if use_coh:
            C = mf_bank_mag(xb, pos16, baud, cfo=cfo, inv=inv, complex_out=True)
            R = coherent_metric(C, baud, inv=inv, alpha=coh_alpha)
            # resolve the carrier-mod-pi global sign: decode both, keep fewer fails
            best = None
            for sgn in (1.0, -1.0):
                ba, bb, ma, mb = mfbank_softbits(sgn * R)
                ph, mg = demux_phases(ba, bb, ma, mb, baud)
                res = decode_phases(ph, mg, chase=chase)
                if best is None or res[2] < best[2]:
                    best = res
            return best
        metric = mf_bank_mag(xb, pos16, baud, cfo=cfo, inv=inv)
        bit_a, bit_b, mag_a, mag_b = mfbank_softbits(metric)
        phases, mags = demux_phases(bit_a, bit_b, mag_a, mag_b, baud)
        return decode_phases(phases, mags, chase=chase)

    sweep = "--sweep" in sys.argv
    frac = "--frac" in sys.argv          # sub-sample (fractional) phase search
    soft = "--soft" in sys.argv          # Chase-II soft-decision FEC on hard fails
    pages = []
    bch_fail = bch_corr = bch_chase = 0
    off_hist = {}
    for f in frames:
        spb = f["spb"]; p0 = f["p0"]; nsyms = f["nsyms"]
        if not sweep:
            offs = [0.0]
        elif frac:
            offs = list(np.arange(-spb / 2, spb / 2, 0.5))
        else:
            offs = [float(o) for o in range(-(spb // 2), spb - spb // 2)]
        # symbol-parity search: 3200-baud A/C de-mux can land on the wrong
        # symbol parity (whole frame fails). Try starting on symbol 0 or 1.
        pars = [0, 1] if f["baud"] == 3200 else [0]
        best = None
        # sweep uses fast HARD decode to pick timing/parity; soft Chase is run
        # once on the winner (it only helps words, never changes which sync is best)
        for par in pars:
            for off in offs:
                base = p0 + off + par * spb
                n = nsyms - par
                pos = base + np.arange(n) * spb
                if pos[0] < 0 or pos[-1] >= len(demod) - 1:
                    continue
                if mfbank:
                    wordsets, confsets, nf, nc, _ = decode_mf(pos, f["baud"], f["levels"])
                else:
                    syms = sample_at(pos)
                    wordsets, confsets, nf, nc, _ = decode_syms(syms, f["baud"], f["levels"])
                if best is None or nf < best[1]:
                    best = (off, nf, nc, wordsets, confsets, par, pos)
        if best is None:
            continue
        off, nf, nc, wordsets, confsets, par, bpos = best
        # Sweep above used the fast noncoherent hard decode to pick timing/parity.
        # Refine the winner once with the coherent detector (Tier 3) and/or Chase.
        if mfbank and (coh or soft):
            wordsets, confsets, nf, nc, nch = decode_mf(
                bpos, f["baud"], f["levels"], chase=soft, use_coh=coh)
            bch_chase += nch
        elif soft:
            bsyms = sample_at(bpos)
            wordsets, confsets, nf, nc, nch = decode_syms(bsyms, f["baud"], f["levels"], chase=True)
            bch_chase += nch
        key = (round(off, 1), par)
        off_hist[key] = off_hist.get(key, 0) + 1
        bch_fail += nf; bch_corr += nc
        if "--pf" in sys.argv:
            bsyms = sample_at(bpos)
            c = kmeans4(bsyms)
            gaps = np.diff(c)
            # within-level spread proxy: residual after assigning to nearest center
            d = np.abs(bsyms[:, None] - c[None, :]); lab = d.argmin(1)
            resid = np.sqrt(np.mean((bsyms - c[lab]) ** 2))
            merit = float(np.min(gaps) / (resid + 1e-9))
            print(f"[pf] off={off:+.1f} par={par} fail={nf:3d} "
                  f"c=[{c[0]:+.1f},{c[1]:+.1f},{c[2]:+.1f},{c[3]:+.1f}] "
                  f"resid={resid:.2f} merit={merit:.2f}")
        for words, cf in zip(wordsets, confsets):
            if words is not None:
                pages.extend(parse_frame(words, cf))
    if sweep:
        print(f"[sweep] chosen (offset,parity) histogram: {dict(sorted(off_hist.items()))}")

    # confidence-graded classification (NO hard BCH gate): emit every page, score it.
    # Tier A verified : all words clean syndrome (status 0) AND min-margin healthy.
    # Tier B fec      : BCH-corrected or Chase-recovered words, no uncorrectable word.
    # Tier C suspect  : passed BCH but min-margin < 0.5x local median (likely miscorrect),
    #                   or no soft info available.
    # Tier D failed   : >=1 uncorrectable word in the page (corrupted body).
    MARGIN_OK = 0.5
    ALPHA_EN_OK = 0.60   # english_score floor for a trustworthy --alpha page
    empty = 0
    tiers = {"A": 0, "B": 0, "C": 0, "D": 0}
    by_tier_type = {"A": {}, "B": {}, "C": {}, "D": {}}
    samples = []

    def tier_of(pc, pr, en):
        # RF/FEC confidence first, then a body-text sanity check: a page the RF
        # layer is sure of but whose body isn't mostly printable is NOT a
        # trustworthy *message* (binary/junk or a likely miscorrection), so it
        # can't sit in A/B regardless of margin.
        if pc is None:
            t = "C"
        elif pc["failed"] > 0:
            t = "D"
        elif pc["margin"] == pc["margin"] and pc["margin"] < MARGIN_OK:
            t = "C"
        elif pc["corrected"] or pc["chase"]:
            t = "B"
        else:
            t = "A"
        if t in ("A", "B") and pr < 0.85:
            t = "C"
        # --alpha: the comb is aggressive, so demand the body actually reads as
        # English. This rejects Chase-manufactured garbage that slips past the
        # printable gate, and conversely lets a genuinely-English-but-garbled
        # message keep its FEC tier.
        if alpha_only and en < ALPHA_EN_OK and t in ("A", "B"):
            t = "C"
        return t

    dump_readable = "--dump-readable" in sys.argv   # emit ALL readable alpha pages
    readable = []
    for cap, typ, body, pc in pages:
        if alpha_only and typ not in ("ALN", "SPN"):   # keep only alpha pages
            continue
        if len(body) == 0:
            empty += 1; continue
        pr = sum(1 for c in body if 32 <= c < 127) / len(body)
        en = english_score(body) if alpha_only else 1.0
        t = tier_of(pc, pr, en)
        tiers[t] += 1
        by_tier_type[t][typ] = by_tier_type[t].get(typ, 0) + 1
        if dump_readable and en >= ALPHA_EN_OK and pr >= 0.85:
            readable.append(body.decode("ascii", "replace"))
        if t in ("A", "B") or len(samples) < 36:
            mg = pc["margin"] if pc else float("nan")
            samples.append((t, typ, mg, pr, en, body.decode("ascii", "replace")))

    fe = "mfbank" + ("" if nocfo else "+cfo") + ("+coh" if coh else "") + ("+inv" if inv else "") if mfbank else "fm-disc"
    acq = ("corr-grid" if corr_grid else "corr-peaks") if corr else "hardsync"
    fe = f"{fe}/{acq}"
    ctxt = f" carrier={center_mhz + carrier/1e6:.4f}MHz (offset{carrier/1e3:+.0f}kHz)" if carrier else ""
    print(f"=== slicer={slicer} front-end={fe} soft={soft} comb={'--comb' in sys.argv}{ctxt} ===")
    print(f"frames        : {len(frames)} (anchors synced={n_anchor}, comb-synth={len(frames)-n_anchor})")
    print(f"BCH corrected : {bch_corr}   Chase-recovered: {bch_chase}   BCH FAILED: {bch_fail}")
    nonempty = sum(tiers.values())
    print(f"pages emitted : {len(pages)}  (empty={empty}, nonempty={nonempty})")
    print(f"confidence    : A_verified={tiers['A']}  B_fec={tiers['B']}  "
          f"C_suspect={tiers['C']}  D_failed={tiers['D']}   (trustworthy A+B={tiers['A']+tiers['B']})")
    print(f"by type/tier  : A={by_tier_type['A']} B={by_tier_type['B']} "
          f"C={by_tier_type['C']} D={by_tier_type['D']}")
    print("--- samples [tier margin printable% english] ---")
    for t, typ, mg, pr, en, s in sorted(samples, key=lambda x: x[0]):
        mtxt = f"{mg:4.2f}" if mg == mg else " nan"
        entxt = f" en={en:.2f}" if alpha_only else ""
        print(f"  [{t} m={mtxt} pr={pr:.2f}{entxt}] {typ}: {s!r}")
    if dump_readable:
        print("--- readable-dump ---")
        for s in readable:
            print(f"READABLE\t{s!r}")

if __name__ == "__main__":
    main()
