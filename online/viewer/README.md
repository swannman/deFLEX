# FLEX pager web viewer

Live terminal-style web feed of decoded FLEX pager messages. Tails the logs
written by `flex-receiver.service` and streams clean message bodies to browsers
over a WebSocket. Runs on the production host as `flex-tail.service` at
`http://<host>:8091/`.

## Files

- `flex_tail_server.py` — FastAPI + uvicorn server. Tails `/var/log/flex/*.log`,
  post-processes the raw ALN decodes, and pushes new messages over a WebSocket.
  Deployed to `/usr/local/bin/flex_tail_server.py`.
- `flex.html` — single-page client (auto-scrolling feed + "hide garbled"
  toggle). Deployed to `/usr/local/share/flex-tail/flex.html` (`STATIC_DIR`).
- `flex-tail.service` — systemd unit. Install to `/etc/systemd/system/`.

## What the server does

The displayed text is ONLY the human message body — capcode/phone, frame, baud,
and fragment metadata are never sent to the client. Server-side processing:

- **De-dup** — identical `(capcode, body)` within `DEDUP_WINDOW` (180s) is
  suppressed (network retransmits / same page on two carriers). Capcode is a key
  only, never exposed.
- **Fragment stitching** — multimon-ng K/F/C flags: F opens a per-capcode
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

## Run / deploy

Uses the shared `gridstream-waterfall-venv` (FastAPI + uvicorn). On the host:

```
sudo cp flex_tail_server.py /usr/local/bin/
sudo cp flex.html /usr/local/share/flex-tail/
sudo cp flex-tail.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now flex-tail.service
```

Syntax-check after editing (write the cache to a writable path):

```
/usr/local/bin/gridstream-waterfall-venv/bin/python -c \
  'import py_compile; py_compile.compile("/usr/local/bin/flex_tail_server.py", cfile="/tmp/ft.pyc", doraise=True)'
sudo systemctl restart flex-tail.service
```
