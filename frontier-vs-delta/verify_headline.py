"""
verify_headline.py - independent re-derivation of the recovery-capacity headline.

`analyze.py` builds per-route sorted arrays of departure times and uses binary search.
This file instead does a self-join on (date, origin, dest) and counts rows directly. The
two share only `fetch_data.py` (download + hash check). If both land on the same numbers
the numbers are real; if `results.json` exists, the comparison is printed.

Also prints a WORKED EXAMPLE - one real cancelled Frontier flight with every other flight
that carrier operated on that route that day, alongside Delta's same-day schedule.

    python verify_headline.py
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

import fetch_data

COLS = ["FlightDate", "IATA_Code_Marketing_Airline", "IATA_Code_Operating_Airline",
        "Flight_Number_Marketing_Airline", "Origin", "Dest", "CRSDepTime",
        "Cancelled", "CancellationCode"]


def load() -> pd.DataFrame:
    frames = []
    for zpath in fetch_data.ensure():
        with zipfile.ZipFile(zpath) as z:
            name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
            with z.open(name) as fh:
                frames.append(pd.read_csv(fh, usecols=lambda c: c.strip() in COLS, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"IATA_Code_Marketing_Airline": "mk", "IATA_Code_Operating_Airline": "op",
                            "Flight_Number_Marketing_Airline": "fnum"})
    df = df[df.mk.isin(["F9", "DL"])].copy()
    df["FlightDate"] = pd.to_datetime(df.FlightDate)
    df["dep"] = df.CRSDepTime.fillna(0).astype(int)
    return df


def recovery_by_selfjoin(s: pd.DataFrame, sample_days: set) -> dict:
    cancelled = s[s.Cancelled == 1][["FlightDate", "Origin", "Dest", "dep"]].reset_index(drop=True)
    cancelled["cx_id"] = cancelled.index
    operated = s[s.Cancelled == 0][["FlightDate", "Origin", "Dest", "dep"]]

    j = cancelled.merge(operated, on=["FlightDate", "Origin", "Dest"], how="left", suffixes=("_cx", "_op"))
    later = j.assign(is_later=j.dep_op.notna() & (j.dep_op > j.dep_cx)).groupby("cx_id").is_later.sum()

    nxt = cancelled.copy()
    nxt["FlightDate"] = nxt.FlightDate + pd.Timedelta(days=1)
    observable = nxt.FlightDate.isin(sample_days)
    j2 = nxt.merge(operated, on=["FlightDate", "Origin", "Dest"], how="left", suffixes=("_cx", "_op"))
    nxt_count = j2.groupby("cx_id").dep_op.count()

    n = len(cancelled)
    zero = int((later == 0).sum())
    obs_ids = cancelled.cx_id[observable]
    stranded = int(((later.loc[obs_ids] == 0) & (nxt_count.loc[obs_ids] <= 1)).sum())
    two_plus = int(((later.loc[obs_ids] == 0) & (nxt_count.loc[obs_ids] == 0)).sum())
    return {"cancellations": n, "zero_same_day": zero, "zero_same_day_pct": round(100.0 * zero / n, 2),
            "observable": int(observable.sum()), "stranded": stranded,
            "stranded_pct": round(100.0 * stranded / int(observable.sum()), 2),
            "two_plus_days": two_plus,
            "two_plus_days_pct": round(100.0 * two_plus / int(observable.sum()), 2)}


def worked_example(df: pd.DataFrame, origin="ATL", dest="LAS") -> None:
    f9 = df[(df.mk == "F9") & (df.Origin == origin) & (df.Dest == dest)]
    cx_days = f9[f9.Cancelled == 1].FlightDate.unique()
    if not len(cx_days):
        print("  (no cancelled Frontier %s-%s in sample)" % (origin, dest))
        return
    day = sorted(cx_days)[0]
    print("\n  --- Frontier %s->%s on %s ---" % (origin, dest, day.date()))
    for _, r in f9[f9.FlightDate == day].sort_values("dep").iterrows():
        status = "CANCELLED (%s)" % r.CancellationCode if r.Cancelled == 1 else "operated"
        print("     F9 %-5d dep %04d   %s" % (int(r.fnum), r.dep, status))
    dl = df[(df.mk == "DL") & (df.Origin == origin) & (df.Dest == dest) & (df.FlightDate == day)]
    print("  --- Delta %s->%s the SAME day: %d flights, %d cancelled, operated by %s ---" % (
        origin, dest, len(dl), int(dl.Cancelled.sum()), sorted(dl.op.unique().tolist())))
    print("     departures:", " ".join("%04d" % d for d in sorted(dl.dep)))


def main():
    print("Loading...")
    df = load()
    sample_days = set(df.FlightDate.unique())
    print("F9+DL rows: %d   days: %d" % (len(df), df.FlightDate.nunique()))

    views = {"Frontier": df[df.mk == "F9"], "Delta": df[df.mk == "DL"],
             "Delta (mainline only)": df[(df.mk == "DL") & (df.op == "DL")]}
    print("\n=== INDEPENDENT RE-DERIVATION (self-join, not binary search) ===")
    mine = {name: recovery_by_selfjoin(s, sample_days) for name, s in views.items()}
    for name, v in mine.items():
        print("  %-22s %5d cancellations | no later flight that day %5d (%.2f%%) | stranded %5d of %5d (%.2f%%) | 2+ days %4d (%.2f%%)" % (
            name, v["cancellations"], v["zero_same_day"], v["zero_same_day_pct"], v["stranded"], v["observable"],
            v["stranded_pct"], v["two_plus_days"], v["two_plus_days_pct"]))

    path = Path(__file__).resolve().parent / "results.json"
    if path.exists():
        theirs = json.loads(path.read_text(encoding="utf-8"))["headline"]["main"]
        ok = all(abs(mine[n]["zero_same_day_pct"] - theirs[n]["zero_same_day_pct"]) < 0.005 and
                 abs(mine[n]["stranded_pct"] - theirs[n]["stranded_pct"]) < 0.005 and
                 abs(mine[n]["two_plus_days_pct"] - theirs[n]["two_plus_days_pct"]) < 0.005 for n in mine)
        print("\n  analyze.py results.json:", {n: (theirs[n]["zero_same_day_pct"], theirs[n]["stranded_pct"], theirs[n]["two_plus_days_pct"]) for n in mine})
        print("  MATCH:", "YES - two independent implementations agree" if ok else "NO - investigate")
    else:
        print("\n  (no results.json yet - run analyze.py to compare)")

    print("\n=== WORKED EXAMPLE ===")
    worked_example(df)


if __name__ == "__main__":
    main()
