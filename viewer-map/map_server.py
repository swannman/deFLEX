#!/usr/bin/env python3
"""Live incident map for paging dispatch feeds (companion to viewer/, the text feed).

Tails the per-carrier POCSAG/FLEX logs written by the receivers, pulls a location
out of each page -- either coordinates embedded in the text or a street address
geocoded via the free US Census geocoder (cached) -- and serves a Leaflet map plus
a live, filterable incident feed at http://<host>:<port>/. Stdlib only; no deps.

Location extraction and the incident-type classifier (in map.html) are tuned for
US fire/EMS dispatch paging (the SNO911-style "FIRE TAC <addr>, <city>" and CAD
";<addr>, <city>;...;<lat>;<lon>" formats). Adjust extract() / incidentType() for
other dispatch formats.

  map_server.py --logs '/var/log/flex/*.pocsag.log' --region WA \
                --bbox 47.0,-122.7,48.6,-121.0 --title 'SNO911 Incidents' --port 8092

Only the --file logs the receivers already write are read; nothing here touches an
SDR. Geocoding sends ONLY the bare street address (never the message body) to the
US Census service."""
import argparse, calendar, glob, json, os, re, time, threading, urllib.request, urllib.parse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
DEDUP_S = 300.0                 # suppress an identical body re-seen within this window
SEED_BYTES = 2_000_000          # tail of each log scanned on startup

POINTS = deque(maxlen=8000)     # {ts, t(epoch), lat, lon, body}
LOCK = threading.Lock()
SEEN = {}
geocache = {}
cfg = {}                        # filled by main(): logs, cache, html, region, bbox, window_s, title

# any "<lat> ; <lon>" decimal pair (e.g. an embedded CAD coordinate)
LATLON = re.compile(r"(-?\d{1,3}\.\d{3,})\s*;\s*(-?\d{1,3}\.\d{3,})")

def in_bbox(lat, lon):
    b = cfg["bbox"]
    if not b:
        return -90 <= lat <= 90 and -180 <= lon <= 180
    return b[0] <= lat <= b[2] and b[1] <= lon <= b[3]

def extract(body):
    """Return ('ll',(lat,lon)) for an embedded coordinate, ('addr',str) for a street
    address to geocode, or None. Tuned for US fire/EMS dispatch paging."""
    for m in LATLON.finditer(body):
        lat, lon = float(m.group(1)), float(m.group(2))
        if in_bbox(lat, lon):
            return ("ll", (lat, lon))
    m = re.search(r"FIRE TAC\s+\d+\s+(.+?)\s+/", body)          # >>BLS1 - <<FIRE TAC 03 110 RASPBERRY LN, Sultan /
    if m and re.search(r"\d", m.group(1)):
        return ("addr", m.group(1).strip().rstrip(","))
    for fld in body.split(";"):                                 # CAD: ; 325 120TH AVE NE, City ;
        fld = fld.strip()
        if re.match(r"^\d+\s+\S", fld) and re.search(
                r"\b(AVE|ST|PL|WAY|DR|BLVD|RD|LN|CT|HWY|PKWY|TER)\b", fld, re.I):
            return ("addr", fld.rstrip(","))
    return None

def geocode(addr):
    if addr in geocache:
        return geocache[addr]
    one = addr + (", " + cfg["region"] if cfg["region"] else "")
    q = urllib.parse.urlencode({"address": one, "benchmark": "Public_AR_Current", "format": "json"})
    coords = None
    try:
        with urllib.request.urlopen(
                "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress?" + q,
                timeout=10) as r:
            d = json.load(r)
        ms = d["result"]["addressMatches"]
        if ms:
            lat, lon = ms[0]["coordinates"]["y"], ms[0]["coordinates"]["x"]
            if in_bbox(lat, lon):
                coords = [lat, lon]
    except Exception:
        coords = None
    geocache[addr] = coords
    try: json.dump(geocache, open(cfg["cache"], "w"))
    except Exception: pass
    time.sleep(0.2)
    return coords

def handle(line):
    if "|ALN|" not in line:
        return
    ts = line.split(" ", 1)[0]
    body = line.split("|ALN|", 1)[1].strip()
    now = time.time()
    try:
        te = calendar.timegm(time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ"))
    except Exception:
        return
    if te < now - cfg["window_s"]:
        return
    if SEEN.get(body, 0) > now:                   # dedup multi-RIC / cross-channel retransmits
        SEEN[body] = now + DEDUP_S
        return
    SEEN[body] = now + DEDUP_S
    if len(SEEN) > 8000:
        for k in [k for k, e in SEEN.items() if e < now]:
            SEEN.pop(k, None)
    g = extract(body)
    if not g:
        return
    coords = list(g[1]) if g[0] == "ll" else geocode(g[1])
    if not coords:
        return
    with LOCK:
        POINTS.append({"ts": ts, "t": te, "lat": coords[0], "lon": coords[1], "body": body})

def prune():
    cut = time.time() - cfg["window_s"]
    with LOCK:
        keep = [p for p in POINTS if p["t"] >= cut]
        POINTS.clear(); POINTS.extend(keep)

def tailer():
    global geocache
    try: geocache = json.load(open(cfg["cache"]))
    except Exception: geocache = {}
    offsets = {}
    seed = []
    for path in glob.glob(cfg["logs"]):
        try:
            sz = os.path.getsize(path)
            with open(path, errors="replace") as f:
                f.seek(max(0, sz - SEED_BYTES)); f.readline()
                seed += f.readlines()
            offsets[path] = sz
        except OSError:
            pass
    seed.sort(key=lambda l: l[:20])               # ISO ts prefix -> chronological
    for ln in seed: handle(ln)
    while True:
        time.sleep(2.0); prune()
        for path in glob.glob(cfg["logs"]):
            try: sz = os.path.getsize(path)
            except OSError: continue
            last = offsets.get(path, 0)
            if sz < last: last = 0
            if sz == last: offsets[path] = sz; continue
            try:
                with open(path, errors="replace") as f:
                    f.seek(last); chunk = f.read(); offsets[path] = f.tell()
            except OSError: continue
            for ln in chunk.splitlines(): handle(ln)

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, payload, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
    def do_GET(self):
        if self.path.startswith("/points"):
            with LOCK: payload = json.dumps(sorted(POINTS, key=lambda p: p["t"])).encode()
            self._send(payload, "application/json")
        else:
            try: html = open(cfg["html"], "r", errors="replace").read()
            except OSError: self._send(b"map.html missing", "text/plain"); return
            html = html.replace("__TITLE__", cfg["title"])
            self._send(html.encode(), "text/html; charset=utf-8")

def main():
    ap = argparse.ArgumentParser(description="Live incident map for paging dispatch feeds.")
    ap.add_argument("--logs", default="/var/log/flex/*.pocsag.log",
                    help="glob of receiver log files to tail (default /var/log/flex/*.pocsag.log)")
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--title", default="Paging Incident Map")
    ap.add_argument("--region", default="",
                    help="appended to addresses before geocoding, e.g. 'WA' (default none)")
    ap.add_argument("--bbox", default="",
                    help="minlat,minlon,maxlat,maxlon to constrain plotted points (default none)")
    ap.add_argument("--window-hours", type=float, default=24.0)
    ap.add_argument("--cache", default=os.path.join(HERE, "geocache.json"))
    ap.add_argument("--html", default=os.path.join(HERE, "map.html"))
    a = ap.parse_args()
    bbox = None
    if a.bbox:
        p = [float(x) for x in a.bbox.split(",")]
        bbox = (min(p[0], p[2]), min(p[1], p[3]), max(p[0], p[2]), max(p[1], p[3]))
    cfg.update(logs=a.logs, cache=a.cache, html=a.html, region=a.region, bbox=bbox,
               window_s=a.window_hours * 3600, title=a.title)
    threading.Thread(target=tailer, daemon=True).start()
    print("incident map: http://%s:%d/  (logs=%s window=%gh region=%r)"
          % (a.host, a.port, a.logs, a.window_hours, a.region or "-"))
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()

if __name__ == "__main__":
    main()
