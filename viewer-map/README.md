# viewer-map — live incident map

A geographic companion to [`viewer/`](../viewer/) (the text feed). It reads the
per-carrier logs the receivers write, locates each page, and serves a Leaflet map
with a live, filterable incident feed.

It's self-contained (Python stdlib only, no pip installs) and independent of the
text viewer — run either, both, or neither.

## What it does

- **Tails** the receiver logs (`--logs` glob) and pulls a location from each page:
  - coordinates embedded in the message (`…;<lat>;<lon>`), or
  - a street address geocoded via the **free US Census geocoder** (no API key),
    cached to disk so each address is looked up once. Only the bare address is
    sent to the geocoder — never the message body.
- **Map** (right): one circle per incident — **color = incident type**
  (medical / fire / vehicle / alarm / hazmat / other), **opacity = age**
  (faded = older, solid = newest). Auto-fits to wherever your data is.
- **Feed** (left ⅓): newest on top, click an item to pan the map to it.
- **Filters**: per-type checkboxes + a "last N hours" slider (rolling window).
- **Auto-follow**: when a new incident arrives and you haven't manually selected
  one, the map opens it; click a marker/item to pin your view, close it to resume.
- Times render in the **browser's local timezone**.

## Run

```bash
# Snohomish/Puget-Sound paging (matches the receivers' default log dir):
python3 viewer-map/map_server.py \
    --logs '/var/log/flex/*.pocsag.log' \
    --region WA --bbox 47.0,-122.7,48.6,-121.0 \
    --title 'SNO911 Incidents' --port 8092
# then open http://<host>:8092/
```

| Flag | Meaning |
|---|---|
| `--logs GLOB` | receiver log files to tail (default `/var/log/flex/*.pocsag.log`) |
| `--region STR` | appended to addresses before geocoding (e.g. `WA`); blank = none |
| `--bbox a,b,c,d` | `minlat,minlon,maxlat,maxlon` — drop points outside it (rejects geocode errors / out-of-area spillover) |
| `--title STR` | page title |
| `--window-hours N` | rolling window kept on the map (default 24) |
| `--port` / `--host` / `--cache` / `--html` | server bind + file locations |

## Tuning for other feeds

Two pieces are tuned for **US fire/EMS dispatch paging** and are the first things
to adjust for a different message format:

- `extract()` in `map_server.py` — how a street address / coordinate is pulled out
  of a page (the `FIRE TAC <addr>, <city>` and CAD `;<addr>, <city>;` patterns).
- `incidentType()` in `map.html` — maps the dispatch nature code to a color/type.

Everything else (geocoding, dedup, map, feed, filters) is format-agnostic.

## Notes

- `geocache.json` is written next to the script (gitignored) and persists across
  restarts, so only genuinely new addresses hit the geocoder.
- Leaflet + OpenStreetMap tiles load from their CDNs, so the **browser** viewing
  the map needs internet; the server itself only needs outbound HTTPS for geocoding.
