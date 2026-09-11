# nflverse-charts

Reusable NFL team/player chart scripts built on [nflverse](https://nflverse.com/) via [`nflreadpy`](https://nflreadpy.nflverse.com/).

Python CLIs that pull play-by-play / player stats and render matplotlib PNGs (and some CSV/MD tables). No fantasy-roster or private projection data included.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Data downloads on first run (cached by nflverse).

## Scripts

| Script | What it charts |
| --- | --- |
| `weekly_charts.py` | Route-runner mix + PROE scatter for a season / through-week |
| `proe_scatter.py` / `proe_table.py` / `team_proe_by_week.py` | Pass rate over expected |
| `offense_3_and_out.py` | Offensive 3-and-out rate by team |
| `offense_success_by_down.py` | Success rate by down |
| `offense_efficiency_scatter.py` | Offensive efficiency scatter |
| `team_epa_components.py` / `epa_drive_tiers.py` | EPA components / drive tiers |
| `qb_expected_pass_epa.py` | QB expected pass EPA |
| `top3_target_share.py` | Top-3 WR target share |
| `team_personnel_by_week.py` | Personnel groupings by week |
| `hvt_chart.py` / `team_hvt_bar.py` | High-value touches (RB HVT scatter / team bar) |
| `rb_opportunity_table.py` | RB snaps / targets / carry shares table |
| `wopr_chart.py` | Weighted Opportunity Rating (WOPR) for one game |

## Examples

```bash
cd scripts

python weekly_charts.py --season 2025
python weekly_charts.py --season 2025 --through-week 14

python offense_3_and_out.py --season 2025 --through-week 11 --out ../out/3ao.png
python offense_success_by_down.py --season 2025 --through-week 18 --out ../out/success.png --csv ../out/success.csv

python hvt_chart.py --season 2025 --weeks 8-11
python wopr_chart.py --game-id 2026_01_NE_SEA
python proe_table.py --season 2025
```

Sample PNGs from a 2025 REG run live in [`examples/`](examples/).

## Notes

- Chart titles intentionally omit duration windows like "through Week N" (except true weekly time-series).
- Route-runner chart uses FTN charting `n_offense_backfield` on QB dropbacks — **not** Trumedia route-runner counts (those are not in nflverse).
- Team logos (when used) are loaded from nflverse assets where available; scripts degrade if logos are missing.

## License

MIT — see [LICENSE](LICENSE). NFL data © nflverse / respective providers.
