"""
analyze.py - Frontier vs Delta: what happens to the passenger when a flight is cancelled.

Data: US DOT / Bureau of Transportation Statistics, "On-Time : Marketing Carrier On-Time
Performance (Beginning January 2018)" - the dataset behind the DOT Air Travel Consumer
Report. Public domain; downloaded into ./data and hash-verified by `fetch_data.py`, which
also lists the months used. The run reports the months it actually loaded and whether they
are continuous, and writes results.json next to this file.

Motivation: a weather cancellation of Frontier ATL-LAS on 2026-09-19 with no rebooking
offered before the following Monday, while a Delta flight on the same route the same
evening departed late but departed. The question is not "who cancels more" - it is
"when they cancel, what is left".

What it reports:
  * HEADLINE - recovery capacity: after a cancellation, what that carrier still has left
    that day and the next on the same route.
  * overall cancellation and delay rates, with day-clustered confidence intervals
  * behaviour on high-disruption days (quintiles of system-wide cancellation rate)
  * a matched origin-airport x date comparison, so weather is held constant
  * cancellations by scheduled hour of day, evening vs morning
  * the Atlanta - Las Vegas route specifically
The dataset contains no fare data; nothing here measures price.

Method notes:
  * "Delta" means the Delta NETWORK - every flight marketed as DL, including Delta
    Connection flights operated by Endeavor (9E), SkyWest (OO) and Republic (YX). A
    Delta ticket-holder is on all of them. Mainline-only (operated by DL) is reported as a
    secondary view. Frontier has no regional partners, so F9 is the same under both.
  * All confidence intervals are day-clustered bootstraps: flights on the same day share
    weather, ATC programs and crew cascades, so the effective sample size is days.
  * The "stranded" and "two-plus days" metrics need the NEXT day to be in the sample.
    Cancellations on a day whose successor is not present (the last day of the sample, or of
    any month whose successor is missing) are excluded from those metrics only; the same-day
    metric still counts them.

    python analyze.py            # downloads data if missing, writes results.json, prints
    python analyze.py --post     # print only the figures used in the public post
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import fetch_data

OUT = Path(__file__).resolve().parent

COLS = [
    "FlightDate", "IATA_Code_Marketing_Airline", "IATA_Code_Operating_Airline",
    "Origin", "Dest", "CRSDepTime", "ArrDelayMinutes",
    "Cancelled", "CancellationCode", "Diverted",
]

CANCEL_CODE = {"A": "Carrier", "B": "Weather", "C": "National Air System", "D": "Security"}

RNG = np.random.default_rng(20260920)
N_BOOT = 2000


# ---------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------

def load() -> pd.DataFrame:
    frames = []
    for zpath in fetch_data.ensure():
        with zipfile.ZipFile(zpath) as z:
            name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
            with z.open(name) as fh:
                df = pd.read_csv(fh, usecols=lambda c: c.strip() in COLS, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        frames.append(df)
        print("  loaded %-18s %8d rows" % (zpath.name, len(df)))
    out = pd.concat(frames, ignore_index=True)
    out["FlightDate"] = pd.to_datetime(out["FlightDate"])
    out["dep"] = out.CRSDepTime.fillna(0).astype(int)
    out.rename(columns={"IATA_Code_Marketing_Airline": "mk", "IATA_Code_Operating_Airline": "op"},
               inplace=True)
    return out


def views(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The three carrier definitions every table below is computed on."""
    return {
        "Frontier": df[df.mk == "F9"],
        "Delta": df[df.mk == "DL"],                          # network - the default
        "Delta (mainline only)": df[(df.mk == "DL") & (df.op == "DL")],
    }


def pct(n, d):
    return round(100.0 * n / d, 2) if d else None


# ---------------------------------------------------------------------------------
# day-clustered inference
# ---------------------------------------------------------------------------------

def boot_rate_ci(day_tab: pd.DataFrame, n_boot: int = N_BOOT):
    """95% CI for a cancellation rate, resampling DAYS not flights."""
    if not len(day_tab):
        return None, None
    f = day_tab["flights"].to_numpy(float)
    c = day_tab["cancels"].to_numpy(float)
    idx = RNG.integers(0, len(f), size=(n_boot, len(f)))
    rates = 100.0 * c[idx].sum(axis=1) / np.maximum(f[idx].sum(axis=1), 1)
    return round(float(np.percentile(rates, 2.5)), 2), round(float(np.percentile(rates, 97.5)), 2)


def boot_ratio_ci(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = N_BOOT):
    """95% CI for (rate_a / rate_b), resampling the SHARED day index jointly."""
    days = a.index.intersection(b.index)
    if not len(days):
        return None, None
    af, ac = a.loc[days, "flights"].to_numpy(float), a.loc[days, "cancels"].to_numpy(float)
    bf, bc = b.loc[days, "flights"].to_numpy(float), b.loc[days, "cancels"].to_numpy(float)
    idx = RNG.integers(0, len(days), size=(n_boot, len(days)))
    ra = ac[idx].sum(axis=1) / np.maximum(af[idx].sum(axis=1), 1)
    rb = bc[idx].sum(axis=1) / np.maximum(bf[idx].sum(axis=1), 1)
    ratio = ra / np.maximum(rb, 1e-12)
    return round(float(np.percentile(ratio, 2.5)), 2), round(float(np.percentile(ratio, 97.5)), 2)


def day_table(s: pd.DataFrame) -> pd.DataFrame:
    return s.groupby("FlightDate").agg(flights=("Cancelled", "size"), cancels=("Cancelled", "sum"))


def separates(lo, hi) -> bool:
    return bool(lo is not None and (lo > 1.0 or hi < 1.0))


# ---------------------------------------------------------------------------------
# THE HEADLINE: recovery capacity
# ---------------------------------------------------------------------------------

def recovery(s: pd.DataFrame, sample_days: set, *, exclude_diverted: bool = False,
             cause: str | None = None, drop_days=None) -> dict:
    """For every cancelled flight of one carrier: how many of that carrier's flights on
    the same origin-destination pair actually OPERATED later the same day, and how many
    operated the next day.

      zero_same_day : nothing operated later that day on the route
      stranded      : zero_same_day AND at most one flight operated the next day
      two_plus_days : zero_same_day AND nothing operated the next day either
                      (both computed only where the next day is in the sample)

    Counts flights, not seats, and ignores whether the seats were sellable - both of
    which understate the problem for a party of several travelling together.
    """
    pool = s[s.Cancelled == 0]
    if exclude_diverted:
        pool = pool[pool.Diverted == 0]
    sched = {k: np.sort(g["dep"].to_numpy())
             for k, g in pool.groupby(["FlightDate", "Origin", "Dest"], sort=False)}

    cx = s[s.Cancelled == 1]
    if cause:
        cx = cx[cx.CancellationCode == cause]
    if drop_days is not None:
        cx = cx[~cx.FlightDate.isin(drop_days)]

    later, nxt = [], []
    for d, o, t, dep in zip(cx.FlightDate, cx.Origin, cx.Dest, cx.dep):
        a = sched.get((d, o, t))
        later.append(0 if a is None else int(len(a) - np.searchsorted(a, dep, side="right")))
        d1 = d + pd.Timedelta(days=1)
        if d1 in sample_days:
            a2 = sched.get((d1, o, t))
            nxt.append(0 if a2 is None else int(len(a2)))
        else:
            nxt.append(-1)                                   # next day not observable

    lt, nd = np.array(later), np.array(nxt)
    n = len(lt)
    obs = nd >= 0
    zero = int((lt == 0).sum())
    stranded = int(((lt == 0) & (nd <= 1) & obs).sum())
    two_plus = int(((lt == 0) & (nd == 0) & obs).sum())

    per_day = s.groupby(["FlightDate", "Origin", "Dest"]).size().rename("per_day").reset_index()
    thin = cx.merge(per_day, on=["FlightDate", "Origin", "Dest"], how="left")

    return {
        "cancellations": int(n),
        "zero_same_day": zero,
        "zero_same_day_pct": pct(zero, n),
        "stranded_n_observable": int(obs.sum()),
        "stranded": stranded,
        "stranded_pct": pct(stranded, int(obs.sum())),
        "two_plus_days": two_plus,
        "two_plus_days_pct": pct(two_plus, int(obs.sum())),
        "median_next_day_options": int(np.median(nd[obs])) if obs.any() else None,
        "cancelled_on_route_scheduled_once_or_less_that_day_pct": pct(int((thin.per_day <= 1).sum()), len(thin)),
    }


def headline(df: pd.DataFrame) -> dict:
    sample_days = set(df.FlightDate.unique())
    daily = df.groupby("FlightDate").agg(f=("Cancelled", "size"), c=("Cancelled", "sum"))
    worst5 = set(daily.assign(r=daily.c / daily.f).sort_values("r", ascending=False).head(5).index)

    out = {"definition": "per cancelled flight, that carrier's own flights that OPERATED later the "
                         "same day / the next day on the same origin-destination pair",
           "main": {}, "sensitivity": {}}
    for name, s in views(df).items():
        out["main"][name] = recovery(s, sample_days)
    for label, kw in [("exclude_diverted_from_pool", dict(exclude_diverted=True)),
                      ("drop_5_worst_system_days", dict(drop_days=worst5)),
                      ("weather_coded_cancellations_only", dict(cause="B")),
                      ("carrier_coded_cancellations_only", dict(cause="A"))]:
        out["sensitivity"][label] = {name: recovery(s, sample_days, **kw)
                                     for name, s in views(df).items()}
    out["worst_5_system_days"] = sorted(str(d.date()) for d in worst5)
    return out


# ---------------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------------

def overall_reliability(df: pd.DataFrame) -> dict:
    res = {}
    for name, s in views(df).items():
        n = len(s)
        cx = int(s.Cancelled.sum())
        dv = int(s.Diverted.sum())
        flown = s[(s.Cancelled == 0) & (s.Diverted == 0)]
        d15 = int((flown.ArrDelayMinutes >= 15).sum())
        lo, hi = boot_rate_ci(day_table(s))
        res[name] = {
            "flights_scheduled": n,
            "cancel_rate_pct": pct(cx, n),
            "cancel_rate_ci95": [lo, hi],
            "arr_delay_15plus_pct": pct(d15, len(flown)),
            "arr_delay_180plus_pct": pct(int((flown.ArrDelayMinutes >= 180).sum()), len(flown)),
            "disruption_rate_pct": pct(cx + dv + d15, n),
        }
    return res


def under_system_stress(df: pd.DataFrame) -> dict:
    """Behaviour on high-disruption days - no difference is asserted unless the CIs say so."""
    daily = df.groupby("FlightDate").agg(sys_flights=("Cancelled", "size"), sys_cancels=("Cancelled", "sum"))
    daily["sys_rate"] = 100.0 * daily.sys_cancels / daily.sys_flights
    qs = ["Q1 calmest", "Q2", "Q3", "Q4", "Q5 worst"]
    daily["quintile"] = pd.qcut(daily.sys_rate, 5, labels=qs)
    d = df.merge(daily[["quintile"]], left_on="FlightDate", right_index=True)
    v = views(d)

    per_q = {}
    for q in qs:
        tabs = {name: day_table(s[s.quintile == q]) for name, s in v.items()}
        row = {name: {"flights": int(t.flights.sum()),
                      "cancel_rate_pct": pct(int(t.cancels.sum()), int(t.flights.sum()))}
               for name, t in tabs.items()}
        lo, hi = boot_ratio_ci(tabs["Frontier"], tabs["Delta"])
        row["frontier_over_delta_ratio_ci95"] = [lo, hi]
        row["separates"] = separates(lo, hi)
        per_q[q] = row

    q5 = daily[daily.quintile == "Q5 worst"]
    worst = daily.sort_values("sys_rate", ascending=False).head(5)
    storm = df[df.FlightDate.isin(worst.index)]
    bad = storm.groupby("Origin").agg(f=("Cancelled", "size"), c=("Cancelled", "sum"))
    bad = bad[(bad.f >= 200) & (100.0 * bad.c / bad.f >= 40)]
    exposure = {name: pct(int(s.Origin.isin(bad.index).sum()), len(s)) for name, s in v.items()}

    return {
        "method": "days ranked into quintiles by SYSTEM-WIDE cancellation rate; day-clustered "
                  "bootstrap CIs on the Frontier/Delta ratio within each quintile",
        "by_quintile": per_q,
        "any_quintile_separates": any(per_q[q]["separates"] for q in qs),
        "q5_share_of_all_system_cancellations_pct": round(100.0 * q5.sys_cancels.sum() / daily.sys_cancels.sum(), 1),
        "top_5_days": [{"date": str(i.date()), "system_cancel_rate_pct": round(r.sys_rate, 1)}
                       for i, r in worst.iterrows()],
        "storm_exposure_pct_of_own_flights_at_40pct_plus_cancelled_airports": exposure,
        "quintiles_where_carriers_separate": [q for q in qs if per_q[q]["separates"]],
    }


def storm_sensitivity(df: pd.DataFrame) -> dict:
    daily = df.groupby("FlightDate").agg(f=("Cancelled", "size"), c=("Cancelled", "sum"))
    worst5 = set(daily.assign(r=daily.c / daily.f).sort_values("r", ascending=False).head(5).index)
    out = {}
    for label, sub in [("all_days", df), ("excluding_5_worst_days", df[~df.FlightDate.isin(worst5)])]:
        v = views(sub)
        row = {name: pct(int(s.Cancelled.sum()), len(s)) for name, s in v.items()}
        lo, hi = boot_ratio_ci(day_table(v["Frontier"]), day_table(v["Delta"]))
        row["frontier_over_delta_ci95"] = [lo, hi]
        row["separates"] = separates(lo, hi)
        out[label] = row
    return out


def matched_airport_day(df: pd.DataFrame) -> dict:
    """Hold weather constant: only (origin airport x date) cells where both flew."""
    d = df[df.mk.isin(["F9", "DL"])]
    g = (d.groupby(["FlightDate", "Origin", "mk"])
          .agg(flights=("Cancelled", "size"), cancels=("Cancelled", "sum")).reset_index())
    p = g.pivot_table(index=["FlightDate", "Origin"], columns="mk", values=["flights", "cancels"], fill_value=0)
    both = p[(p[("flights", "F9")] > 0) & (p[("flights", "DL")] > 0)]
    tabs = {}
    for code in ["F9", "DL"]:
        t = both[("cancels", code)].groupby(level=0).sum().to_frame("cancels")
        t["flights"] = both[("flights", code)].groupby(level=0).sum()
        tabs[code] = t[["flights", "cancels"]]
    lo, hi = boot_ratio_ci(tabs["F9"], tabs["DL"])
    return {
        "cells": int(len(both)),
        "coverage_of_all_frontier_flights_pct": pct(int(both[("flights", "F9")].sum()), int((df.mk == "F9").sum())),
        "Frontier_cancel_pct": pct(int(both[("cancels", "F9")].sum()), int(both[("flights", "F9")].sum())),
        "Delta_cancel_pct": pct(int(both[("cancels", "DL")].sum()), int(both[("flights", "DL")].sum())),
        "frontier_over_delta_ci95": [lo, hi],
        "separates": separates(lo, hi),
    }


def time_of_day(df: pd.DataFrame) -> dict:
    """Evening (19-21h) vs morning (06-09h) cancellation rate, per carrier, with a
    difference-in-differences against Delta to test whether the pattern is Frontier-specific.
    Also the full hourly profile."""
    d = df[df.CRSDepTime.notna()].copy()
    d["hour"] = (d.CRSDepTime.astype(int) // 100).clip(0, 23)
    d["slot"] = np.where(d.hour.between(19, 21), "evening",
                np.where(d.hour.between(6, 9), "morning", "other"))
    v = views(d)

    res, tabs = {}, {}
    for name, s in v.items():
        ev, mo = s[s.slot == "evening"], s[s.slot == "morning"]
        e, m = pct(int(ev.Cancelled.sum()), len(ev)), pct(int(mo.Cancelled.sum()), len(mo))
        lo, hi = boot_ratio_ci(day_table(ev), day_table(mo))
        res[name] = {"evening_19_21_pct": e, "evening_n": len(ev),
                     "morning_06_09_pct": m, "morning_n": len(mo),
                     "evening_over_morning_ratio": round(e / m, 2) if m else None,
                     "ratio_ci95": [lo, hi], "separates": bool(lo is not None and lo > 1.0)}
        tabs[name] = (day_table(ev), day_table(mo))

    def did(a, b):
        (ae, am), (be, bm) = tabs[a], tabs[b]
        days = ae.index.intersection(am.index).intersection(be.index).intersection(bm.index)
        idx = RNG.integers(0, len(days), size=(N_BOOT, len(days)))

        def lograte(t):
            f = t.loc[days, "flights"].to_numpy(float)[idx].sum(axis=1)
            c = t.loc[days, "cancels"].to_numpy(float)[idx].sum(axis=1)
            return np.log(np.maximum(c / np.maximum(f, 1), 1e-9))

        x = (lograte(ae) - lograte(am)) - (lograte(be) - lograte(bm))
        return {"log_point": round(float(np.mean(x)), 2),
                "ci95": [round(float(np.percentile(x, 2.5)), 2), round(float(np.percentile(x, 97.5)), 2)],
                "ratio_of_ratios": round(float(np.exp(np.mean(x))), 2)}

    res["diff_in_diff_Frontier_vs_Delta"] = did("Frontier", "Delta")
    res["diff_in_diff_Frontier_vs_Delta_mainline"] = did("Frontier", "Delta (mainline only)")

    prof = {}
    for name, s in v.items():
        g = s.groupby("hour").agg(n=("Cancelled", "size"), c=("Cancelled", "sum"))
        prof[name] = {int(h): {"n": int(r.n), "cancel_pct": pct(int(r.c), int(r.n))}
                      for h, r in g.iterrows() if r.n >= 500}
    res["hourly_profile_min500_flights"] = prof
    return res


def cancellation_causes(df: pd.DataFrame) -> dict:
    out = {}
    for name, s in views(df).items():
        cx = s[s.Cancelled == 1]
        counts = cx.CancellationCode.value_counts()
        total = int(counts.sum())
        out[name] = {"total_cancellations": total,
                     "by_cause_pct": {CANCEL_CODE.get(k, k): pct(int(c), total) for k, c in counts.items()}}
    return out


def atl_las(df: pd.DataFrame) -> dict:
    out = {}
    for name, s in views(df).items():
        if name.startswith("Delta ("):
            continue
        for o, t in [("ATL", "LAS"), ("LAS", "ATL")]:
            r = s[(s.Origin == o) & (s.Dest == t)]
            if not len(r):
                continue
            per_day = r.groupby("FlightDate").size()
            flown = r[(r.Cancelled == 0) & (r.Diverted == 0)]
            out["%s %s-%s" % (name, o, t)] = {
                "flights": len(r),
                "scheduled_per_day_avg": round(per_day.mean(), 2),
                "days_with_only_one_flight_pct": pct(int((per_day <= 1).sum()), len(per_day)),
                "cancel_rate_pct": pct(int(r.Cancelled.sum()), len(r)),
                "arr_delay_15plus_pct": pct(int((flown.ArrDelayMinutes >= 15).sum()), len(flown)),
                "operating_carriers": sorted(r.op.unique().tolist()),
            }
    return out


# ---------------------------------------------------------------------------------
# the figures used in the public post
# ---------------------------------------------------------------------------------

def post_numbers(results: dict) -> list[tuple[str, str, str]]:
    """(figure in the post, exact value, how it rounds)."""
    h = results["headline"]["main"]
    c1 = results["overall_reliability"]
    c6 = results["cancellations_by_time_of_day"]
    f, d, dm = h["Frontier"], h["Delta"], h["Delta (mainline only)"]

    def zero(v):
        return "%.2f%%  (%d of %d)" % (v["zero_same_day_pct"], v["zero_same_day"], v["cancellations"])

    def strand(v):
        return "%.2f%%  (%d of %d observable)" % (v["stranded_pct"], v["stranded"], v["stranded_n_observable"])

    def two(v):
        return "%.2f%%  (%d of %d observable)" % (v["two_plus_days_pct"], v["two_plus_days"], v["stranded_n_observable"])

    def evmo(v):
        return "%.2f%% / %.2f%%  ratio %.2fx" % (v["evening_19_21_pct"], v["morning_06_09_pct"], v["evening_over_morning_ratio"])

    return [
        ("months in the sample", "%s to %s (%d months)" % (results["sample"]["months_present"][0],
                                                          results["sample"]["months_present"][-1],
                                                          len(results["sample"]["months_present"])),
         "continuous" if results["sample"]["continuous"] else "DISCONTINUOUS"),
        ("overall cancellation rate, Frontier", "%.2f%%" % c1["Frontier"]["cancel_rate_pct"], "CI %s" % c1["Frontier"]["cancel_rate_ci95"]),
        ("overall cancellation rate, Delta network", "%.2f%%" % c1["Delta"]["cancel_rate_pct"], "CI %s" % c1["Delta"]["cancel_rate_ci95"]),
        ("Frontier cancels -> no later flight that day on the route", zero(f), "%d%%" % round(f["zero_same_day_pct"])),
        ("Delta network cancels -> no later flight that day", zero(d), "%d%%" % round(d["zero_same_day_pct"])),
        ("   (Delta mainline only, for reference)", zero(dm), "%d%%" % round(dm["zero_same_day_pct"])),
        ("Frontier: nothing that day AND <=1 flight next day", strand(f), "%d%%" % round(f["stranded_pct"])),
        ("Delta network: nothing that day AND <=1 flight next day", strand(d), "%d%%" % round(d["stranded_pct"])),
        ("   (Delta mainline only, for reference)", strand(dm), "%d%%" % round(dm["stranded_pct"])),
        ("Frontier: nothing that day AND nothing next day (2+ days)", two(f), "%d%%" % round(f["two_plus_days_pct"])),
        ("Delta network: nothing that day AND nothing next day", two(d), "%d%%" % round(d["two_plus_days_pct"])),
        ("   (Delta mainline only, for reference)", two(dm), "%d%%" % round(dm["two_plus_days_pct"])),
        ("Frontier evening (19-21h) cancellation rate", "%.2f%%" % c6["Frontier"]["evening_19_21_pct"], "%.1f%%" % c6["Frontier"]["evening_19_21_pct"]),
        ("Frontier morning (06-09h) cancellation rate", "%.2f%%" % c6["Frontier"]["morning_06_09_pct"], "%.1f%%" % c6["Frontier"]["morning_06_09_pct"]),
        ("Frontier evening/morning ratio", "%.2fx" % c6["Frontier"]["evening_over_morning_ratio"], "CI %s" % c6["Frontier"]["ratio_ci95"]),
        ("Delta network evening / morning", evmo(c6["Delta"]), "CI %s" % c6["Delta"]["ratio_ci95"]),
        ("Delta mainline evening / morning", evmo(c6["Delta (mainline only)"]), "CI %s" % c6["Delta (mainline only)"]["ratio_ci95"]),
    ]


def print_post(results: dict) -> None:
    print("\n=== FIGURES USED IN THE POST (exact -> rounded) ===")
    for figure, exact, rounded in post_numbers(results):
        print("  %-60s %-40s %s" % (figure, exact, rounded))


# ---------------------------------------------------------------------------------

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"

    if "--post" in sys.argv and path.exists():
        print_post(json.loads(path.read_text(encoding="utf-8")))
        return

    print("Loading BTS monthly files...")
    df = load()
    months = sorted(df.FlightDate.dt.strftime("%Y-%m").unique().tolist())
    periods = pd.PeriodIndex(months, freq="M")
    missing_inside = [str(p) for p in pd.period_range(periods[0], periods[-1], freq="M") if p not in periods]
    continuous = not missing_inside
    print("total rows: %d   months: %s   distinct days: %d   continuous: %s"
          % (len(df), ", ".join(months), df.FlightDate.nunique(), continuous))
    net = df[df.mk == "DL"].groupby("op").size().sort_values(ascending=False)
    print("Delta network by operating carrier:", net.to_dict())

    results = {
        "source": "US DOT / BTS, On-Time: Marketing Carrier On-Time Performance (Beginning January 2018)",
        "sample": {
            "months_present": months,
            "continuous": continuous,
            "months_missing_inside_range": missing_inside,
            "distinct_days": int(df.FlightDate.nunique()),
            "total_flights_all_carriers": len(df),
            "LIMITATION": ("%d month(s), %s to %s, %s. September 2026 - the month of the incident "
                           "that motivated the study - is not yet published by BTS."
                           % (len(months), months[0], months[-1],
                              "continuous" if continuous else "DISCONTINUOUS (missing %s)" % ", ".join(missing_inside))),
        },
        "carrier_definitions": {
            "Frontier": "marketed F9 (no regional partners; identical to operated F9)",
            "Delta": "marketed DL - the network a Delta ticket-holder flies on, incl. Delta Connection",
            "Delta (mainline only)": "marketed DL and operated DL",
            "delta_network_by_operator": {str(k): int(v) for k, v in net.items()},
        },
        "headline": headline(df),
        "overall_reliability": overall_reliability(df),
        "behaviour_under_system_stress": under_system_stress(df),
        "storm_sensitivity": storm_sensitivity(df),
        "matched_airport_day": matched_airport_day(df),
        "cancellations_by_time_of_day": time_of_day(df),
        "cancellation_causes": cancellation_causes(df),
        "route_ATL_LAS": atl_las(df),
    }
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== HEADLINE - when they cancel, what is left? ===")
    for name, v in results["headline"]["main"].items():
        print("  %-22s cancellations %5d   no later flight that day %6.2f%%   stranded %6.2f%%   2+ days %5.2f%% (n=%d)   thin-route %5.2f%%" % (
            name, v["cancellations"], v["zero_same_day_pct"], v["stranded_pct"], v["two_plus_days_pct"],
            v["stranded_n_observable"], v["cancelled_on_route_scheduled_once_or_less_that_day_pct"]))
    print("  sensitivity (no-later-flight % / stranded %):")
    for label, row in results["headline"]["sensitivity"].items():
        print("    %-36s " % label + "   ".join(
            "%s %5.2f / %5.2f" % (n[:8], v["zero_same_day_pct"], v["stranded_pct"]) for n, v in row.items()))

    print("\n=== OVERALL - cancellation and delay rates ===")
    for n, v in results["overall_reliability"].items():
        print("  %-22s cancel %5.2f%% %-14s arr15+ %5.2f%%  arr180+ %5.2f%%" % (
            n, v["cancel_rate_pct"], str(v["cancel_rate_ci95"]), v["arr_delay_15plus_pct"], v["arr_delay_180plus_pct"]))

    c2 = results["behaviour_under_system_stress"]
    print("\n=== HIGH-DISRUPTION DAYS - quintiles where the carriers separate: %s ===" % (c2["quintiles_where_carriers_separate"] or "none"))
    for q, row in c2["by_quintile"].items():
        print("  %-11s F9 %5.2f%%  DL %5.2f%%  ratio CI %s" % (
            q, row["Frontier"]["cancel_rate_pct"], row["Delta"]["cancel_rate_pct"], row["frontier_over_delta_ratio_ci95"]))
    print("  Q5 holds %.1f%% of all system cancellations; storm exposure %s" % (
        c2["q5_share_of_all_system_cancellations_pct"],
        c2["storm_exposure_pct_of_own_flights_at_40pct_plus_cancelled_airports"]))

    print("\n=== storm sensitivity ===")
    for k, v in results["storm_sensitivity"].items():
        print("  %-24s F9 %s%%  DL %s%%  ratio CI %s  separates=%s" % (
            k, v["Frontier"], v["Delta"], v["frontier_over_delta_ci95"], v["separates"]))

    m = results["matched_airport_day"]
    print("\n=== matched airport x day ===\n  %d cells, %s%% of Frontier flights: F9 %s%% vs DL %s%%, ratio CI %s, separates=%s" % (
        m["cells"], m["coverage_of_all_frontier_flights_pct"], m["Frontier_cancel_pct"],
        m["Delta_cancel_pct"], m["frontier_over_delta_ci95"], m["separates"]))

    c6 = results["cancellations_by_time_of_day"]
    print("\n=== TIME OF DAY - evening (19-21h) vs morning (06-09h) ===")
    for n in views(df):
        v = c6[n]
        print("  %-22s evening %5.2f%%  morning %5.2f%%  ratio %s CI %s  separates=%s" % (
            n, v["evening_19_21_pct"], v["morning_06_09_pct"], v["evening_over_morning_ratio"],
            v["ratio_ci95"], v["separates"]))
    print("  diff-in-diff vs Delta network :", c6["diff_in_diff_Frontier_vs_Delta"])
    print("  diff-in-diff vs Delta mainline:", c6["diff_in_diff_Frontier_vs_Delta_mainline"])
    print("  hourly cancel % (hours with >=500 flights):")
    hours = sorted({int(h) for p in c6["hourly_profile_min500_flights"].values() for h in p})
    print("    %-22s " % "hour" + " ".join("%5d" % h for h in hours))
    for n, p in c6["hourly_profile_min500_flights"].items():
        print("    %-22s " % n + " ".join("%5.2f" % p[h]["cancel_pct"] if h in p else "    -" for h in hours))

    print("\n=== route ATL-LAS ===")
    for n, v in results["route_ATL_LAS"].items():
        print("  %-18s %5.2f/day   only-one-flight days %5.2f%%   cancel %5.2f%%   operated by %s" % (
            n, v["scheduled_per_day_avg"], v["days_with_only_one_flight_pct"], v["cancel_rate_pct"],
            v["operating_carriers"]))

    print_post(results)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
