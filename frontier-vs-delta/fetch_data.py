"""
fetch_data.py - download and verify the BTS on-time data this study runs on.

Source: US DOT Bureau of Transportation Statistics, TranStats,
"On-Time : Marketing Carrier On-Time Performance (Beginning January 2018)".
Monthly zip archives are served publicly (no account, no click-through licence) from

    https://transtats.bts.gov/PREZIP/On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_<YYYY>_<M>.zip

The data is a work of the US federal government (17 U.S.C. s.105) compiled from filings the
carriers are legally required to make under 14 CFR Part 234. It is public domain. BTS asks
only for citation.

Why the MARKETING carrier table and not the older "Reporting Carrier" table: regional
partners (Endeavor 9E, SkyWest OO, Republic YX) operate a large share of a legacy carrier's
network under that carrier's flight numbers. The Reporting Carrier table files them under
the operator's code, which silently drops them from the legacy carrier and, for Endeavor,
from the table altogether. The marketing table carries both `IATA_Code_Marketing_Airline`
and `IATA_Code_Operating_Airline`, so "Delta" can be measured as the network a Delta
ticket-holder actually flies on, and mainline-only is still available as a view.

Data goes to ./data next to this file.

    python fetch_data.py          # download anything missing, verify every hash
    python fetch_data.py --check  # verify only, download nothing
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = ("https://transtats.bts.gov/PREZIP/"
       "On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_%d_%d.zip")

# (year, month) -> SHA-256 of the archive as served on 2026-09-20. Every month from
# December 2025 to June 2026 - the newest month BTS had published at that date - so the
# sample is continuous. BTS occasionally re-issues a month with revisions; a mismatch is
# reported loudly, not silently accepted.
MONTHS = {
    (2025, 12): "5597d2d99bc2c2ca92bad7837effe853cac07126e9447498db7bbfb25b834522",
    (2026, 1):  "120ee1cdf2f0958db9a59adb7c275d7ad15aa0910e4daeaf2f2ed70972d3673c",
    (2026, 2):  "b4675446c9a7b4073299a9c893c64af0d364a44cb0e1483909457ceac70c582e",
    (2026, 3):  "d606ae6cb912eed1757f13e5540a41a8efe753df68285d685c92eb1b6d804b78",
    (2026, 4):  "b1562f00f7eec5bcb3bdd5061eae10b0622c8a895ac3f03fdb832bcf258a28c0",
    (2026, 5):  "8937678dc8cd5140102a626825e64911833ce80bc3b44d81a83c805779d6d44e",
    (2026, 6):  "8344497b06cc052f0122c90d012a2e78d4308a7b3eec3cc105ce9199f02c8960",
}


def data_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def local_name(year: int, month: int) -> str:
    return "mkt_%d_%d.zip" % (year, month)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print("\r    %5.1f%% of %.1f MB" % (100.0 * done / total, total / 1e6), end="", flush=True)
    print()
    tmp.replace(dest)


def ensure(check_only: bool = False) -> list[Path]:
    """Return the list of monthly zips, downloading any that are missing. Verifies every
    hash and prints a warning for any mismatch (the numbers may then differ from the
    published ones)."""
    root = data_dir()
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for (y, m), want in MONTHS.items():
        p = root / local_name(y, m)
        if not p.exists():
            if check_only:
                print("  MISSING  %s" % p)
                continue
            print("  downloading %s" % p.name)
            download(URL % (y, m), p)
        got = sha256(p)
        if got != want:
            print("  WARNING  %s  sha256 %s != expected %s  (BTS may have re-issued this month)"
                  % (p.name, got[:16], want[:16]))
        else:
            print("  ok       %s  sha256 %s..." % (p.name, got[:16]))
        paths.append(p)
    total = sum(p.stat().st_size for p in paths)
    print("  %d file(s), %.1f MB on disk" % (len(paths), total / 1e6))
    return paths


if __name__ == "__main__":
    print("data dir:", data_dir())
    ensure(check_only="--check" in sys.argv)
