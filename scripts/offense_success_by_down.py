#!/usr/bin/env python
"""Offensive success rate by down — 32-team facet grid (Sam Hoppen–style).

Success rate = percent of offensive plays with positive EPA (epa > 0).
Filters (aligned with offense_efficiency_scatter.py):
  REG, week <= through_week, downs 1–3, epa not null,
  exclude qb_kneel / qb_spike / no_play; keep pass/run (or pass==1 / rush==1).

League average = overall mean(epa>0) across all teams and downs 1–3
(single scalar drawn as a dashed line on every facet).

Usage:
  python offense_success_by_down.py
  python offense_success_by_down.py --season 2025 --through-week 11
  python offense_success_by_down.py --season 2025 --through-week 18 --out out.png --csv out.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import nflreadpy as nfl
import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent

# Hoppen / weekly_charts division order (LA not LAR)
DIVISIONS = [
    ("AFC East", ["BUF", "MIA", "NE", "NYJ"]),
    ("AFC North", ["BAL", "CIN", "CLE", "PIT"]),
    ("AFC South", ["HOU", "IND", "JAX", "TEN"]),
    ("AFC West", ["DEN", "KC", "LAC", "LV"]),
    ("NFC East", ["DAL", "NYG", "PHI", "WAS"]),
    ("NFC North", ["CHI", "DET", "GB", "MIN"]),
    ("NFC South", ["ATL", "CAR", "NO", "TB"]),
    ("NFC West", ["ARI", "LA", "SEA", "SF"]),
]
TEAM_ORDER = [t for _, ts in DIVISIONS for t in ts]
DOWNS = [1, 2, 3]


def _hex(c, fb="#333"):
    if not c:
        return fb
    c = str(c).strip()
    if not c.startswith("#"):
        c = "#" + c
    return c if len(c) == 7 else fb


def load_team_meta():
    return (
        nfl.load_teams()
        .filter(pl.col("team_abbr").is_in(TEAM_ORDER))
        .select(
            [
                "team_abbr",
                "team_nick",
                "team_name",
                "team_color",
                "team_color2",
            ]
        )
    )


def compute(season: int, through_week: int):
    pbp = nfl.load_pbp(seasons=season)
    plays = pbp.filter(
        (pl.col("season_type") == "REG")
        & (pl.col("week") <= through_week)
        & pl.col("posteam").is_not_null()
        & pl.col("posteam").is_in(TEAM_ORDER)
        & pl.col("epa").is_not_null()
        & pl.col("down").is_in(DOWNS)
        & (~pl.col("play_type").is_in(["qb_kneel", "qb_spike", "no_play"]))
        & (
            pl.col("play_type").is_in(["pass", "run"])
            | (pl.col("pass") == 1)
            | (pl.col("rush") == 1)
        )
    ).with_columns((pl.col("epa") > 0).cast(pl.Float64).alias("is_success"))

    by_team_down = (
        plays.group_by(["posteam", "down"])
        .agg(
            pl.len().alias("n_plays"),
            pl.col("is_success").sum().alias("n_success"),
            (100.0 * pl.col("is_success").mean()).alias("success_rate"),
        )
        .rename({"posteam": "team"})
        .sort(["team", "down"])
    )

    league_avg = float(plays["is_success"].mean()) * 100.0
    meta = {
        "season": season,
        "through_week": through_week,
        "n_plays": int(plays.height),
        "league_avg": league_avg,
        "source": (
            f"nfl.load_pbp(seasons={season}) REG weeks 1-{through_week}; "
            "downs 1-3; success = epa>0; excl qb_kneel/qb_spike/no_play; pass/run"
        ),
        "definition": (
            "success rate = mean(epa > 0) on offensive pass/run plays, downs 1–3; "
            "league avg = overall mean across all teams and downs 1–3"
        ),
    }
    return by_team_down, meta


def _title_for_week(meta: dict) -> str:
    return "Offensive success rate by down"


def plot(df: pl.DataFrame, teams: pl.DataFrame, meta: dict, out: Path):
    lookup = {
        (r["team"], int(r["down"])): float(r["success_rate"]) for r in df.to_dicts()
    }
    n_lookup = {
        (r["team"], int(r["down"])): int(r["n_plays"]) for r in df.to_dicts()
    }
    meta_rows = {r["team_abbr"]: r for r in teams.to_dicts()}
    league = meta["league_avg"]

    fig, axes = plt.subplots(4, 8, figsize=(18, 10.5), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")

    for i, team in enumerate(TEAM_ORDER):
        ax = axes[i // 8][i % 8]
        ax.set_facecolor("white")
        row = meta_rows.get(team, {})
        c1 = _hex(row.get("team_color"), "#444")
        c2 = _hex(row.get("team_color2"), "#222")
        nick = (row.get("team_nick") or team).upper()

        vals = [lookup.get((team, d), 0.0) for d in DOWNS]
        xs = np.arange(3)
        ax.bar(
            xs,
            vals,
            color=c1,
            edgecolor=c2,
            linewidth=1.25,
            width=0.68,
            zorder=3,
        )
        ax.axhline(
            league,
            color="#222",
            linestyle="--",
            linewidth=1.0,
            zorder=4,
        )
        ax.set_ylim(25, 65)
        ax.set_yticks([30, 40, 50, 60])
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%d%%"))
        ax.set_xticks(xs)
        ax.set_xticklabels(["1", "2", "3"], fontsize=8)
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("#CCC")
        ax.spines["bottom"].set_color("#CCC")
        ax.tick_params(axis="y", labelsize=7, colors="#444", length=0)
        ax.tick_params(axis="x", labelsize=8, colors="#333", length=0)

        # y labels only on left column; x labels only on bottom row
        if i % 8 != 0:
            ax.tick_params(axis="y", labelleft=False)
        if i // 8 != 3:
            ax.tick_params(axis="x", labelbottom=False)

        ax.set_title(nick, color=c1, fontsize=9.5, fontweight="bold", pad=6)

    fig.text(
        0.06,
        0.975,
        _title_for_week(meta),
        fontsize=15,
        fontweight="bold",
        va="top",
        color="#111",
    )
    fig.text(
        0.06,
        0.935,
        "Success rate = percent of plays with positive EPA; "
        f"dashed line = league average across all downs (~{league:.0f}%)",
        fontsize=9.5,
        color="#444",
        va="top",
    )
    fig.text(
        0.5,
        0.035,
        "Down",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#222",
    )
    fig.text(
        0.98,
        0.012,
        "Figure: Quant | Data: nflverse",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#666",
    )

    plt.tight_layout(rect=[0.03, 0.05, 0.99, 0.91])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)

    # attach n_plays into printed summary via side channel
    _ = n_lookup


def main():
    ap = argparse.ArgumentParser(
        description="Offensive success rate by down — 32-team facet grid"
    )
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--through-week", type=int, default=11)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    ww = f"{args.through_week:02d}"
    if args.through_week >= 18:
        default_stem = HERE / f"offense_success_by_down_{args.season}_reg"
    else:
        default_stem = HERE / f"offense_success_by_down_{args.season}_w01-{ww}"
    out = args.out or Path(str(default_stem) + ".png")
    csv_path = args.csv or out.with_suffix(".csv")

    df, meta = compute(args.season, args.through_week)
    teams = load_team_meta()
    plot(df, teams, meta, out)

    # wide-ish CSV: team, down, n_plays, n_success, success_rate + league_avg column
    out_df = df.with_columns(pl.lit(meta["league_avg"]).alias("league_avg"))
    out_df.write_csv(csv_path)

    print("wrote", out)
    print("wrote", csv_path)
    print("meta:", meta)
    # quick per-team down-1 check
    for team in ("BUF", "DET", "SF", "NE"):
        for d in DOWNS:
            rows = df.filter((pl.col("team") == team) & (pl.col("down") == d)).to_dicts()
            if rows:
                r = rows[0]
                print(
                    f"  {team} down {d}: {r['success_rate']:.1f}% "
                    f"({int(r['n_success'])}/{int(r['n_plays'])})"
                )


if __name__ == "__main__":
    main()
