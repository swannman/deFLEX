"""FLEX pager message tail web server.

Tails /var/log/flex/*.log (written by the receivers), post-processes the
raw ALN decodes, and streams clean messages to browsers over a WebSocket. The
displayed text is ONLY the human message body — no capcode/phone, frame, baud,
or fragment metadata is ever sent to the client.

Post-processing (server-side, using metadata that is NOT exposed to the page):
  * De-duplication: identical (carrier, body) seen again within DEDUP_WINDOW is
    suppressed. Kills network retransmits AND the dispatch broadcast that sends
    the same text to many unit RICs at once. The capcode is used only as a
    fragment-stitch key and is never sent to clients.
  * Fragment stitching: multimon-ng tags each ALN line with K/F/C
    (K = complete; F = fragment, needs continuation; C = continuation of a
    fragment). An F line opens a per-capcode buffer; subsequent C lines for that
    capcode are concatenated; the joined message is emitted once a K/F arrives
    for the capcode or STITCH_TIMEOUT elapses.

URL:  http://<host>:8091/
"""
import asyncio
import glob
import json
import os
import re
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

# ---- Config ----
LOG_GLOB = "/var/log/flex/*.log"
STATIC_DIR = Path("/usr/local/share/flex-tail")
RING_LEN = 300           # messages kept in memory for initial page load
POLL_INTERVAL = 0.5      # seconds between file polls
SEED_TAIL_BYTES = 16000  # how much of each log's tail to replay on startup
DEDUP_WINDOW = 180.0     # seconds an identical (capcode, body) is suppressed
STITCH_TIMEOUT = 8.0     # seconds to wait for continuation fragments

# Messages whose body contains any of these (case-insensitive) are dropped
# entirely — automated test/heartbeat pages with no human value.
DROP_SUBSTRINGS = ("test page",)

# Carriers excluded from the feed (none by default). 929662500 was assumed to be
# a test transmitter, but the band census found it carries live traffic.
SKIP_CARRIERS = set()

# Map carrier center freq (Hz, = log filename) to the paging network name.
CARRIER_NAME = {
    "929587500": "Spok",
    "929612500": "Spok",
    "929662500": "Spok",
    "929937500": "AmerMsg",
    "931062500": "AmerMsg",
    "931212500": "Spok",
    "931862500": "Spok",
    "931937500": "SkyTel",
    "152007500": "SNO911",     # VHF POCSAG fire/EMS dispatch (SNOCO simulcast)
    "453900000": "Alarm-70cm", # UHF POCSAG fire-alarm paging
}

# ---- State ----
ring: deque = deque(maxlen=RING_LEN)
clients: set[WebSocket] = set()
clients_lock = asyncio.Lock()
stats = {"emitted": 0, "deduped": 0, "stitched": 0, "dropped": 0}

app = FastAPI(title="FLEX Pager Feed")

_TS = re.compile(r"^(\S+)\s+(.*)$")


def parse_line(line: str):
    """Return (carrier-agnostic) record dict for an ALN line, or None.
    Keeps capcode + fragment flag for server-side processing (never displayed)."""
    m = _TS.match(line)
    if not m:
        return None
    ts, rest = m.group(1), m.group(2)
    if rest.startswith("FLEX_NEXT|"):
        p = rest.split("|")
        if len(p) < 8 or p[6] != "ALN":
            return None
        capcode = p[3]
        fragmark = p[7]
        flag = fragmark[-1] if fragmark and fragmark[-1] in "KFC" else "K"
        body = "|".join(p[8:]) if len(p) > 8 else ""
    elif rest.startswith("FLEX|"):
        # Legacy format has no fragment field -> always complete.
        p = rest.split("|")
        if len(p) < 6 or p[5] != "ALN":
            return None
        capcode = p[4]
        flag = "K"
        body = "|".join(p[6:]) if len(p) > 6 else ""
    else:
        return None
    # escaped control chars -> space, then collapse whitespace runs to one space
    # and trim the ends, so multi-line / padded pages render as one clean line
    body = body.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
    body = " ".join(body.split())
    if not body:
        return None
    return {"ts": ts, "capcode": capcode, "flag": flag, "body": body}


_B64_CHARS = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]{2,}@[A-Za-z0-9-]{2,}\.[A-Za-z]{2,}")
_URL_RE = re.compile(
    r"https?://|www\.[A-Za-z0-9-]+\.[A-Za-z]{2,}|[A-Za-z0-9-]+\.[A-Za-z]{2,}/")


def _is_url_or_email(tok: str) -> bool:
    # Require a real email/URL shape (word@word.tld, scheme, or host/path).
    # A bare '@' or '.' from FEC garble must NOT count, or it falsely exempts
    # blobs like '@.2A70...' that happen to carry a stray '@'.
    return bool(_EMAIL_RE.search(tok) or _URL_RE.search(tok))


def looks_encoded(body: str) -> bool:
    """True if the body looks like a base64/encrypted blob rather than human
    text. The reliable tell is a long unbroken run with no spaces — real pages
    have words, punctuation, and whitespace. Bit-errors can sprinkle non-base64
    symbols into the blob, so we key on run length rather than exact charset,
    while exempting URLs/emails (which are legitimately long and space-free).
    A real email/URL anywhere in the body (not just the longest token) means
    it's a structured message — e.g. fab alarm pages whose long token is a
    machine ID sitting next to a clean 'alarms@example.net'."""
    b = body.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ").strip()
    if not b:
        return False
    tokens = b.split()
    if any(_is_url_or_email(t) for t in tokens):
        return False
    longest = max(tokens, key=len)
    # A long unbroken run is almost never natural language.
    if len(longest) >= 24:
        return True
    # Shorter single-token bodies that are nearly pure base64 alphabet.
    nospace = b.replace(" ", "")
    if len(tokens) <= 1 and len(nospace) >= 20:
        b64 = sum(1 for c in nospace if c in _B64_CHARS)
        if b64 / len(nospace) >= 0.90:
            return True
    return False


def is_readable(body: str) -> bool:
    """Heuristic: genuine human text vs. encrypted/base64 page or FEC junk."""
    b = body.replace("\\t", " ").replace("\\n", " ").replace("\\r", " ")
    if len(b.strip()) < 3:
        return False
    if looks_encoded(body):
        return False
    printable = sum(1 for c in b if 32 <= ord(c) < 127)
    if printable / len(b) < 0.85:
        return False
    if sum(1 for c in b if c.isalpha()) < 3:
        return False
    return bool(re.search(r"[A-Za-z]{3,}", b))


class Processor:
    """Dedup + fragment-stitch pipeline. Calls emit_cb(msg_dict) for each
    finished message. Capcode is used only as a key; never placed in msg_dict."""

    def __init__(self, emit_cb):
        self.emit_cb = emit_cb
        self.pending: dict[str, dict] = {}   # capcode -> {parts, ts, carrier, deadline}
        self.seen: dict[tuple, float] = {}   # (carrier, body) -> expiry

    def _emit(self, capcode, ts, carrier, body, now, stitched):
        body = body.strip()
        if not body:
            return
        low = body.lower()
        if any(s in low for s in DROP_SUBSTRINGS):
            stats["dropped"] += 1
            return
        # Dedup on (carrier, body), NOT capcode: dispatch systems blast the same
        # message to many unit RICs at once, so keying on capcode lets every copy
        # through. Body-per-carrier collapses the broadcast to one line (and the
        # capcode is never displayed anyway).
        key = (carrier, body)
        exp = self.seen.get(key)
        self.seen[key] = now + DEDUP_WINDOW
        if exp and exp > now:
            stats["deduped"] += 1
            return
        stats["emitted"] += 1
        if stitched:
            stats["stitched"] += 1
        msg = {
            "ts": ts,
            "net": CARRIER_NAME.get(carrier, carrier),
            "body": body,
            "readable": is_readable(body),
            "stitched": stitched,
        }
        ring.append(msg)
        self.emit_cb(msg)

    def _flush(self, capcode, now):
        p = self.pending.pop(capcode, None)
        if p:
            self._emit(capcode, p["ts"], p["carrier"], "".join(p["parts"]),
                       now, stitched=len(p["parts"]) > 1)

    def feed(self, rec, carrier, now):
        cap, flag, body, ts = rec["capcode"], rec["flag"], rec["body"], rec["ts"]
        if flag == "F":
            self._flush(cap, now)  # close any prior open fragment first
            self.pending[cap] = {"parts": [body], "ts": ts,
                                  "carrier": carrier, "deadline": now + STITCH_TIMEOUT}
        elif flag == "C":
            p = self.pending.get(cap)
            if p:
                p["parts"].append(body)
                p["deadline"] = now + STITCH_TIMEOUT
            else:
                # Orphan continuation (missed the F) -> emit standalone.
                self._emit(cap, ts, carrier, body, now, stitched=False)
        else:  # "K" -> complete standalone
            self._flush(cap, now)
            self._emit(cap, ts, carrier, body, now, stitched=False)

    def tick(self, now):
        for cap in [c for c, p in self.pending.items() if p["deadline"] <= now]:
            self._flush(cap, now)
        if len(self.seen) > 4000:
            self.seen = {k: e for k, e in self.seen.items() if e > now}


async def _broadcast(msg: dict):
    data = json.dumps(msg)
    dead = []
    async with clients_lock:
        for ws in clients:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)


def _broadcast_soon(msg: dict):
    asyncio.create_task(_broadcast(msg))


# Live processor emits straight to websockets.
processor = Processor(_broadcast_soon)


def seed_ring():
    """Populate the ring from the tail of each log (post-processed) so a freshly
    opened page isn't blank. Uses a throwaway processor that only fills the ring."""
    seed_proc = Processor(lambda m: None)  # ring is filled inside _emit
    records = []
    for path in glob.glob(LOG_GLOB):
        carrier = os.path.basename(path)[:-4]
        if carrier in SKIP_CARRIERS:
            continue
        try:
            size = os.path.getsize(path)
            with open(path, "r", errors="replace") as f:
                if size > SEED_TAIL_BYTES:
                    f.seek(size - SEED_TAIL_BYTES)
                    f.readline()
                for line in f:
                    rec = parse_line(line)
                    if rec:
                        records.append((rec, carrier))
        except OSError:
            continue
    records.sort(key=lambda rc: rc[0]["ts"])
    now = time.time()
    for rec, carrier in records:
        seed_proc.feed(rec, carrier, now)
    seed_proc.tick(now + STITCH_TIMEOUT + 1)  # flush trailing fragments


async def tailer():
    offsets = {p: os.path.getsize(p) for p in glob.glob(LOG_GLOB)
               if os.path.exists(p)}
    while True:
        await asyncio.sleep(POLL_INTERVAL)
        now = time.time()
        for path in glob.glob(LOG_GLOB):
            carrier = os.path.basename(path)[:-4]
            if carrier in SKIP_CARRIERS:
                continue
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            last = offsets.get(path, 0)
            if size < last:        # logrotate copytruncate -> start over
                last = 0
            if size == last:
                continue
            try:
                with open(path, "r", errors="replace") as f:
                    f.seek(last)
                    chunk = f.read()
                    offsets[path] = f.tell()
            except OSError:
                continue
            for line in chunk.splitlines():
                rec = parse_line(line)
                if rec:
                    processor.feed(rec, carrier, now)
        processor.tick(now)


@app.on_event("startup")
async def startup():
    seed_ring()
    asyncio.create_task(tailer())


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "flex.html")


@app.get("/recent")
async def recent():
    return list(ring)


@app.get("/stats")
async def get_stats():
    return {**stats, "ring": len(ring), "pending": len(processor.pending),
            "clients": len(clients)}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    async with clients_lock:
        clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        async with clients_lock:
            clients.discard(ws)
