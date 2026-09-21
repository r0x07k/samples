# Frontier vs Delta — recovery after a cancellation, from US DOT data

Small, reproducible study behind a public post about what happens to a passenger *after* an
airline cancels a flight. It asks one question the usual "cancellation rate" statistic cannot
answer: **when this carrier cancels, is there anything left that day on your route?**

## Data

US Department of Transportation, Bureau of Transportation Statistics (BTS), TranStats:
**On-Time: Marketing Carrier On-Time Performance (Beginning January 2018)**. This is the
dataset behind the monthly DOT *Air Travel Consumer Report*; carriers are legally required to
file it (14 CFR Part 234). It is a work of the US federal government and therefore public
domain (17 U.S.C. §105); BTS's own data portal labels the series "Public Domain U.S. Government"
(data.bts.gov, dataset 56fa-sf82, license `usa.gov/publicdomain/label/1.0`). Nothing here
redistributes it — `fetch_data.py` downloads it from BTS.

Months used: **December 2025 through June 2026** — every month BTS had published when this was
written (seven months, 212 days, 4,501,892 flights across all reporting carriers). September
2026 — the month that prompted the study — was not yet available.

Web form for readers who want to check a single airline or flight without downloading
anything: https://www.transtats.bts.gov/ontime/

## Method

For every cancelled flight of a carrier, count that carrier's own flights on the same
origin–destination pair that **actually operated** later the same day, and the next day.

- *No later flight that day* — none operated after the cancelled departure time.
- *Stranded* — no later flight that day **and at most one** operated the next day.
- *Two-plus days* — no later flight that day **and none** the next day either.
  (Both computed only where the next day is in the sample.)

"Delta" means every flight **marketed** as Delta, including Delta Connection flights operated by
Endeavor, SkyWest and Republic — the network a Delta ticket-holder is actually rebooked within.
Mainline-only is reported as a secondary view. Frontier has no regional partners.

All confidence intervals are day-clustered bootstraps (flights on the same day are not
independent trials). The study also reports overall cancellation and delay rates, behaviour on
high-disruption days, a matched airport×date comparison, cancellation by hour of day, and the
Atlanta–Las Vegas route specifically.

## Run it

```
pip install pandas numpy
python analyze.py            # downloads ~243 MB from BTS on first run, verifies SHA-256,
                             # writes results.json, prints every table and the post's figures
python analyze.py --post     # reprint just the figures used in the post
python verify_headline.py    # independent re-derivation of the headline by a different method
python fetch_data.py --check # verify the data files without running anything
```

The data lands in `./data` next to the scripts (seven zip files, about 244 MB in total) and
`results.json` is written next to the scripts. Runtime is a few minutes on a laptop.

## Headline results (as of 2026-09-20)

| | cancellations | no later flight that day | stranded | two-plus days |
|---|---|---|---|---|
| Frontier | 2,591 | 78.7% | 57.0% | 22.6% |
| Delta (network) | 22,760 | 52.7% | 18.3% | 7.0% |
| Delta (mainline only) | 12,988 | 44.1% | 15.5% | 4.4% |

Frontier's evening (19–21h) departures are cancelled 1.56× as often as its morning (06–09h)
ones (2.73% vs 1.75%, 95% CI [1.21, 2.05]). Overall cancellation rates for the two carriers do
not separate statistically (2.10% vs 2.39%).

## Limitations

- No July–September data; the incident's own season is absent.
- Counts flights, not seats; a party of several needs several seats on the same flight.
- Credits no re-routing through hubs, which understates the larger carrier's options.
- Contains no fare data; nothing here measures price.
