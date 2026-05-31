# FLEX pager web viewer

Live terminal-style web feed of decoded pager messages. Tails the logs written by
the receivers (`flex_receiver.py` / `pocsag_receiver.py`) and streams clean
message bodies to browsers over a WebSocket. A small FastAPI/uvicorn server.

## Files

- `flex_tail_server.py` — FastAPI + uvicorn server. Tails `/var/log/flex/*.log`,
  post-processes the raw ALN decodes, and pushes new messages over a WebSocket.
- `flex.html` — single-page client (auto-scrolling feed + "hide garbled" toggle),
  served by `flex_tail_server.py` from `STATIC_DIR`.

## What the server does

The displayed text is ONLY the human message body — capcode/phone, frame, baud,
and fragment metadata are never sent to the client. Server-side processing:

- **De-dup** — identical `(capcode, body)` within `DEDUP_WINDOW` (180s) is
  suppressed (network retransmits / same page on two carriers). Capcode is a key
  only, never exposed.
- **Fragment stitching** — K/F/C continuation flags: F opens a per-capcode
  buffer, C lines concatenate, emitted on the next K/F or `STITCH_TIMEOUT` (8s).
- **`looks_encoded()` / `is_readable()`** — the display "garbled" gate. Flags
  base64/encrypted blobs (a space-free run ≥24 chars, or a single ≥20-char
  ≥90%-base64 token) as not-readable so the page can hide them. A real
  email/URL **anywhere** in the body exempts it (so fab alarm pages whose long
  token is a machine ID next to a clean `alarms@example.net` are shown). The
  email/URL test requires a real `word@word.tld` / scheme / `host/path` shape so
  FEC garble carrying a stray `@` does not falsely exempt a blob.
- **`DROP_SUBSTRINGS`** — drops "test page" heartbeats entirely.
- **`SKIP_CARRIERS`** = {929662500} — always-on test carrier excluded from the
  feed (still logged for receiver-health confirmation).

## Run

Needs FastAPI + uvicorn. The server tails `/var/log/flex/*.log` and serves
`flex.html` (point `STATIC_DIR` at it):

```
uvicorn flex_tail_server:app --host 0.0.0.0 --port 8091
```

For continuous operation, run it under your service manager.
