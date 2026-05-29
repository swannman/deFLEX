#!/usr/bin/env python3
"""7-channel nationwide FLEX paging receiver.

One SDRplay RSPdx source @ 930.7625 MHz / 2.646 MS/s, fanned out into 7
freq-xlating channels (one per nationwide FLEX carrier). Each channel:
  freq_xlate -> conjugate (FLEX inverted) -> FM quad-demod -> 22050 Hz int16 -> FIFO
multimon-ng readers (started first by the launcher) consume each FIFO.

Separate SDR from the AirSpy GridStream rig -- no impact on that capture.
"""
import math, sys
from gnuradio import gr, blocks, filter, analog, soapy
from gnuradio.filter import firdes

CARRIERS = [
    929_587_500,   # Spok          6400 ABCD (nominal; ~silent, unmeasured)
    929_612_500,   # Spok          6400 ABCD (3200 baud/4lvl) [measured 2026-05-29]
    929_662_500,   # Spok          6400 ABCD (3200 baud/4lvl) always-on test carrier
    929_937_500,   # American Msg  3200 AB   (1600 baud/4lvl) [measured 2026-05-29]
    931_062_500,   # American Msg  1200 POCSAG (FLEX bps unmeasured)
    931_212_500,   # Spok          6400 ABCD (3200 baud/4lvl) [measured 2026-05-29]
    931_937_500,   # SkyTel        3200 AB   (1600 baud/4lvl) [measured 2026-05-29]
]
CENTER = 930_762_500           # midpoint of the 929.5875-931.9375 span
SAMP_RATE = 2_646_000          # /120 -> 22050 Hz audio (multimon-ng's rate)
CHAN_DECIM = 120
AUDIO_RATE = SAMP_RATE // CHAN_DECIM   # 22050
DEVIATION = 4800.0
FIFO_DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/flex"


class flex_multi(gr.top_block):
    def __init__(self):
        gr.top_block.__init__(self, "FLEX 7CH")

        self.src = soapy.source("driver=sdrplay", "fc32", 1, "", "", [""], [""])
        self.src.set_sample_rate(0, SAMP_RATE)
        self.src.set_frequency(0, CENTER)
        self.src.set_gain_mode(0, False)
        self.src.set_gain(0, "RFGR", 4)
        self.src.set_gain(0, "IFGR", 25)
        self.src.set_antenna(0, "Antenna A")

        taps = firdes.low_pass(1.0, SAMP_RATE, 8000, 4000)
        demod_gain = AUDIO_RATE / (2 * math.pi * DEVIATION)
        self._blocks = []   # keep refs alive
        for carrier in CARRIERS:
            offset = carrier - CENTER
            xlate = filter.freq_xlating_fir_filter_ccc(CHAN_DECIM, taps, offset, SAMP_RATE)
            conj = blocks.conjugate_cc()                  # FLEX inverted polarity
            demod = analog.quadrature_demod_cf(demod_gain)
            f2s = blocks.float_to_short(1, 16000)
            sink = blocks.file_sink(gr.sizeof_short, f"{FIFO_DIR}/{carrier}.s16", False)
            sink.set_unbuffered(True)
            self.connect(self.src, xlate, conj, demod, f2s, sink)
            self._blocks += [xlate, conj, demod, f2s, sink]


if __name__ == "__main__":
    tb = flex_multi()
    tb.start()
    print(f"7-channel FLEX RX: RSPdx @ {CENTER/1e6:.4f} MHz, {SAMP_RATE} S/s, "
          f"{len(CARRIERS)} channels -> {AUDIO_RATE} Hz int16 -> {FIFO_DIR}/", file=sys.stderr)
    try:
        tb.wait()
    except KeyboardInterrupt:
        tb.stop(); tb.wait()
