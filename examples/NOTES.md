# NFL Charts Recreate — Notes

Season default: **2025 REG**. Most SamHoppen-style refs labeled “through Week 11” → weeks **1–11**.

## Skipped / stubbed (proprietary or incomplete in nflverse)

### 03 — Dual-team RB / WR-TE usage table
- Needs **Route %**, **TPRR / wTPRR**, **MTF**, **RYOE** (PFF / NGS / Trumedia).
- Feasible with nflverse alone: Snap %, Targets, Air Yards, aDOT, WOPR, HVT, GZ touches via `load_snap_counts` + `load_pbp`.
- Not recreated as a full visual match. Related script: `rb_opportunity_table.py`.

### 07 — Key QB Stats heatmap
- Needs **PFF Grade**, **TO-Worthy Play Rate**, **Clean-Pocket** splits, **Pressure to Sack Rate** (PFF/Trumedia).
- Feasible subset: EPA/Play, Success Rate, CPOE, aDOT, Non-PA EPA/Success from `load_pbp`.
- Full heatmap not recreated (would invent / omit proprietary cells).

### 14 — WR usage (WOPR vs Routes Run Rate)
- Ref footer: **Trumedia**. Routes Run Rate is not available as a per-player rate in nflverse.
- `load_participation().route` is route *type* on a play, not participation rate.
- WOPR is nflverse-feasible (`wopr_chart.py` pattern); Y-axis is not → skipped.

## Recreated with caveats
- **06 personnel**: ref cites Trumedia; chart uses `load_participation().offense_personnel` (nflverse) classified into 11/12/21/Other — labeled as nflverse proxy.
- **01 target share**: active-game denominator = games with rush or target involvement; pass attempts = `pass_attempt==1` for posteam in those games.
- **10 / 12 HVT**: `hvt_chart.py` with `--min-opp-per-game` / `--min-opp-total` matching reference filters.

### 14 — Offensive 3-and-out rate
- Series-level (not classic 3-play drives): `series_success == 0` / all non-kneel series.
- Matches Hoppen subtitle: turnover or failure to gain first down or touchdown.
- Script: `offense_3_and_out.py` (`--season` / `--through-week`).
- Source: `nfl.load_pbp(seasons=...)` REG.
