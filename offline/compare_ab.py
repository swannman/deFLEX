#!/usr/bin/env python3
"""Per-carrier unique-readable-alpha A/B: flexdec vs multimon-ng.
flexdec source: fd_<off>.txt  READABLE<TAB><repr> lines (any tier, en>=0.60, pr>=0.85).
multimon source: mm_<off>.txt  FLEX_NEXT|...|ALN|...|<body> lines.
Messages fragment differently between tools, so we normalize (collapse ws, lower,
keep alnum+space) and report unique counts + fuzzy overlap (one normalized body
is a substring of the other)."""
import ast, os, re, sys
from difflib import SequenceMatcher

_HERE = os.path.dirname(os.path.abspath(__file__))
# Shared core (paging_core.py) lives at the repo root; cover the flat install too.
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
from paging_core import english_score   # same English gate the decoders use

RATIO = 0.80   # fuzzy-match threshold for clustering garbled near-duplicates

EN_OK = 0.60

CARR = [("-1150000", "929.6125"), ("-1100000", "929.6625"), ("-825000", "929.9375"),
        ("450000", "931.2125"), ("1175000", "931.9375")]


def gated(body):
    """True if body passes the same readability gate flexdec applies."""
    b = body.encode("ascii", "replace") if isinstance(body, str) else body
    if len(b) == 0:
        return False
    pr = sum(1 for c in b if 32 <= c < 127) / len(b)
    return pr >= 0.85 and english_score(b) >= EN_OK


def norm(s):
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def fd_bodies(off):
    out = []
    try:
        for ln in open("/tmp/fd_%s.txt" % off, errors="replace"):
            if ln.startswith("READABLE\t"):
                try:
                    out.append(ast.literal_eval(ln.split("\t", 1)[1].strip()))
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    return out


def mm_bodies(off):
    out = []
    try:
        for ln in open("/tmp/mm_%s.txt" % off, errors="replace"):
            if "|ALN|" in ln or "|SPN|" in ln:
                out.append(ln.rstrip("\n").split("|")[-1])
    except FileNotFoundError:
        pass
    return out


def close(a, b):
    if a in b or b in a:
        return True
    return SequenceMatcher(None, a, b).ratio() >= RATIO


def cluster(bodies, minlen=12):
    # gate -> normalize -> fuzzy-cluster (collapse garbled near-duplicate copies
    # and fragments into one representative). Returns list of cluster reps.
    keys = []
    for b in bodies:
        if not gated(b):
            continue
        n = norm(b)
        if len(n) >= minlen:
            keys.append(n)
    keys = sorted(set(keys), key=len, reverse=True)
    reps = []
    for k in keys:
        if not any(close(k, r) for r in reps):
            reps.append(k)
    return reps


def cross(a, b):
    return sum(1 for ka in a if any(close(ka, kb) for kb in b))


hdr = "%10s | %7s %7s %6s %8s %8s" % ("carrier", "fd_msgs", "mm_msgs", "shared", "fd_only", "mm_only")
print(hdr)
print("-" * len(hdr))
tot_fd = tot_mm = tot_ov = 0
for off, name in CARR:
    fd = cluster(fd_bodies(off))
    mm = cluster(mm_bodies(off))
    ov = cross(fd, mm)
    print("%10s | %7d %7d %6d %8d %8d" % (name, len(fd), len(mm), ov, len(fd) - ov, len(mm) - cross(mm, fd)))
    tot_fd += len(fd); tot_mm += len(mm); tot_ov += ov
print("-" * len(hdr))
print("%10s | %7d %7d %6d" % ("TOTAL", tot_fd, tot_mm, tot_ov))
