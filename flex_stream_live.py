#!/usr/bin/env python3
"""Live consumer: tail a growing complex64 IQ file (the flex_7ch.py complex
baseband tap for one carrier) and stream it through flexdec_stream.StreamDecoder
in real time, printing each trustworthy A/B alpha page as it is decoded.

This is the live-SDR side of the streaming decoder: instead of replaying a frozen
.cfile, it feeds the bytes GNU Radio is writing right now. Runs in parallel with
multimon-ng (which consumes the FM-discriminator output of the SAME channel), so
the two logs can be A/B'd over an identical RF window.

Usage: flex_stream_live.py <iq_file> [in_rate] [run_seconds]
"""
import sys
import time
import numpy as np

sys.path.insert(0, "/tmp/flex_ab")
sys.path.insert(0, "/tmp/pager-bindings")
import flexdec_stream as S

ITEMSZ = 8   # complex64 = 8 bytes


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/flex/929612500_iq.cfile"
    in_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 22050.0
    run_s = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0   # 0 = run forever

    logf = open("/tmp/flex_ab/stream_live.log", "a", buffering=1)

    def on_page(rec):
        slot, typ, body, tier, pr, en = rec
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        line = "%s STREAM slot=%-6d [%s pr=%.2f en=%.2f] %s: %r" % (
            ts, slot, tier, pr, en, typ, body.decode("ascii", "replace"))
        print(line, flush=True)
        logf.write(line + "\n")

    sd = S.StreamDecoder(cfg=dict(in_rate=in_rate), on_page=on_page)

    # wait for the file to appear
    while True:
        try:
            f = open(path, "rb")
            break
        except FileNotFoundError:
            time.sleep(0.5)

    print("[live] tailing %s @ %.0f Hz  window=%.1fs advance=%.1fs" % (
        path, in_rate, sd.window / in_rate, sd.advance / in_rate), flush=True)

    t0 = time.time()
    carry = b""
    fed = 0
    last_report = t0
    while True:
        chunk = f.read(4 * 1024 * 1024)
        if chunk:
            buf = carry + chunk
            n = len(buf) - (len(buf) % ITEMSZ)
            carry = buf[n:]
            if n:
                samps = np.frombuffer(buf[:n], dtype=np.complex64)
                sd.feed(samps)
                fed += len(samps)
        else:
            time.sleep(0.3)   # at EOF of a still-growing file; wait for more
        now = time.time()
        if now - last_report >= 30:
            print("[live] fed %.1fs of IQ (%d samples), %d pages so far" % (
                fed / in_rate, fed, len(sd.pages)), flush=True)
            last_report = now
        if run_s and (now - t0) >= run_s:
            break

    sd.flush()
    print("[live] done: %d trustworthy A/B pages over %.1fs of IQ" % (
        len(sd.pages), fed / in_rate), flush=True)
    logf.close()


if __name__ == "__main__":
    main()
