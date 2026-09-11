#!/usr/bin/env python
"""Team EPA component breakdown table with diverging bars.

Positive = good for all columns (defense EPA inverted).

Usage:
  python team_epa_components.py --season 2025 --through-week 11 --out out.png
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import nflreadpy as nfl
import numpy as np
import polars as pl
import requests
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

HERE = Path(__file__).resolve().parent
LOGO_DIR = HERE / "logos"
COLS = [
    ("pass_off", "Pass Off"),
    ("rush_off", "Rush Off"),
    ("pass_def", "Pass Def"),
    ("run_def", "Run Def"),
    ("turnovers", "Turnovers"),
    ("penalties", "Penalties"),
    ("st", "Special Teams"),
]


def load_team_meta():
    return nfl.load_teams().select(
        ["team_abbr", "team_name", "team_logo_espn", "team_logo_squared"]
    )


def download_logos(teams):
    LOGO_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for row in teams.iter_rows(named=True):
        abbr = row["team_abbr"]
        dest = LOGO_DIR / f"{abbr}.png"
        if dest.exists() and dest.stat().st_size > 0:
            paths[abbr] = dest
            continue
        for url in [row.get("team_logo_espn"), row.get("team_logo_squared")]:
            if not url:
                continue
            try:
                r = requests.get(url, timeout=20)
                r.raise_for_status()
                dest.write_bytes(r.content)
                paths[abbr] = dest
                break
            except Exception:
                continue
    return paths


def compute(season, through_week):
    pbp = nfl.load_pbp(seasons=season)
    pbp = pbp.filter(
        (pl.col("season_type") == "REG")
        & (pl.col("week") <= through_week)
        & pl.col("epa").is_not_null()
    )
    # Offensive pass/rush
    off = pbp.filter(pl.col("posteam").is_not_null())
    pass_off = (
        off.filter(pl.col("pass") == 1)
        .group_by("posteam")
        .agg(pl.col("epa").sum().alias("pass_off"))
        .rename({"posteam": "team"})
    )
    rush_off = (
        off.filter((pl.col("rush") == 1) | ((pl.col("rush_attempt") == 1) & (pl.col("pass") != 1)))
        .group_by("posteam")
        .agg(pl.col("epa").sum().alias("rush_off"))
        .rename({"posteam": "team"})
    )
    # Defense: invert so positive is good
    pass_def = (
        pbp.filter((pl.col("pass") == 1) & pl.col("defteam").is_not_null())
        .group_by("defteam")
        .agg((-pl.col("epa").sum()).alias("pass_def"))
        .rename({"defteam": "team"})
    )
    run_def = (
        pbp.filter(
            ((pl.col("rush") == 1) | ((pl.col("rush_attempt") == 1) & (pl.col("pass") != 1)))
            & pl.col("defteam").is_not_null()
        )
        .group_by("defteam")
        .agg((-pl.col("epa").sum()).alias("run_def"))
        .rename({"defteam": "team"})
    )
    # Turnovers: takeaways positive EPA for defense side of ball; net for team
    # Approximate: interceptions + fumbles for/against via posteam EPA on turnover plays
    to_plays = pbp.filter(
        (pl.col("interception") == 1)
        | (pl.col("fumble_lost") == 1)
        | (pl.col("fumble") == 1)
    )
    # Giveaways: posteam negative impact = sum epa on giveaways (usually negative for offense)
    # Takeaways: when defteam gets the ball — use -epa of posteam (i.e. def benefits)
    give = (
        pbp.filter(
            ((pl.col("interception") == 1) | (pl.col("fumble_lost") == 1))
            & pl.col("posteam").is_not_null()
        )
        .group_by("posteam")
        .agg(pl.col("epa").sum().alias("give_epa"))
        .rename({"posteam": "team"})
    )
    take = (
        pbp.filter(
            ((pl.col("interception") == 1) | (pl.col("fumble_lost") == 1))
            & pl.col("defteam").is_not_null()
        )
        .group_by("defteam")
        .agg((-pl.col("epa").sum()).alias("take_epa"))
        .rename({"defteam": "team"})
    )
    # Penalties
    pen = (
        pbp.filter((pl.col("penalty") == 1) & pl.col("posteam").is_not_null())
        .group_by("posteam")
        .agg(pl.col("epa").sum().alias("penalties"))
        .rename({"posteam": "team"})
    )
    # Special teams: kickoff, punt, field_goal, extra_point
    st = (
        pbp.filter(
            pl.col("play_type").is_in(["kickoff", "punt", "field_goal", "extra_point"])
            & pl.col("posteam").is_not_null()
        )
        .group_by("posteam")
        .agg(pl.col("epa").sum().alias("st"))
        .rename({"posteam": "team"})
    )

    teams = nfl.load_teams().filter(
        pl.col("team_abbr").is_in(
            ["ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB",
             "HOU","IND","JAX","KC","LA","LAC","LV","MIA","MIN","NE","NO","NYG","NYJ",
             "PHI","PIT","SEA","SF","TB","TEN","WAS"]
        )
    ).select(pl.col("team_abbr").alias("team"), "team_name")

    df = teams
    for part in (pass_off, rush_off, pass_def, run_def, give, take, pen, st):
        df = df.join(part, on="team", how="left")
    df = df.with_columns(
        [
            pl.col(c).fill_null(0.0)
            for c in ["pass_off", "rush_off", "pass_def", "run_def", "give_epa", "take_epa", "penalties", "st"]
        ]
    ).with_columns(
        (pl.col("give_epa") + pl.col("take_epa")).alias("turnovers")
    ).sort("team_name")

    meta = {
        "season": season,
        "through_week": through_week,
        "source": f"load_pbp({season}) REG 1-{through_week}; sum(epa) by component; def inverted",
    }
    return df, meta


def plot(df, logos, meta, out):
    rows = df.to_dicts()
    n = len(rows)
    # layout: team col + 7 metric cols
    fig_h = max(12, 0.42 * n + 2.5)
    fig, axes = plt.subplots(1, 8, figsize=(16, fig_h), gridspec_kw={"width_ratios": [1.6] + [1] * 7})
    fig.patch.set_facecolor("white")

    # Team column
    ax0 = axes[0]
    ax0.set_xlim(0, 1)
    ax0.set_ylim(-0.5, n - 0.5)
    ax0.invert_yaxis()
    ax0.axis("off")
    ax0.set_title("Team", fontsize=10, fontweight="bold", pad=8)
    for i, r in enumerate(rows):
        path = logos.get(r["team"])
        if path:
            img = Image.open(path).convert("RGBA")
            img.thumbnail((80, 80), Image.Resampling.LANCZOS)
            im = OffsetImage(np.asarray(img), zoom=14.0 / img.size[1])
            ax0.add_artist(AnnotationBbox(im, (0.12, i), frameon=False, pad=0))
        ax0.text(0.28, i, r["team_name"] or r["team"], va="center", fontsize=8)

    # find global max abs for shared scale per column? per-column scale is fine
    for j, (key, title) in enumerate(COLS):
        ax = axes[j + 1]
        vals = [float(r[key] or 0) for r in rows]
        vmax = max(abs(v) for v in vals) or 1.0
        ax.set_xlim(-vmax * 1.15, vmax * 1.15)
        ax.set_ylim(-0.5, n - 0.5)
        ax.invert_yaxis()
        ax.axvline(0, color="#333", linewidth=0.8)
        ax.set_yticks([])
        ax.set_title(title, fontsize=9, fontweight="bold", pad=8)
        for i, v in enumerate(vals):
            color = "#3A9B6E" if v >= 0 else "#6B4C9A"
            ax.barh(i, v, height=0.65, color=color, alpha=0.75, zorder=2)
            ax.text(0, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5, color="#111", zorder=3)
        ax.tick_params(axis="x", labelsize=7)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="x", color="#EEE", zorder=0)

    fig.suptitle(
        "Breaking down team performance by various components, measured by total EPA",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.975,
        "Positive EPA is better in all cases",
        ha="center",
        fontsize=10,
        color="#444",
    )
    fig.text(
        0.02,
        0.01,
        "¹ Net EPA on all turnovers (takeaways and giveaways)",
        fontsize=8,
        color="#555",
    )
    fig.text(0.98, 0.01, f"Data: nflverse {meta['source']}", ha="right", fontsize=8, color="#666")
    plt.tight_layout(rect=[0.01, 0.03, 0.99, 0.96])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--through-week", type=int, default=11)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    df, meta = compute(args.season, args.through_week)
    logos = download_logos(load_team_meta())
    plot(df, logos, meta, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
