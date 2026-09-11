#!/usr/bin/env python
"""Offensive EPA/play vs success rate scatter with trend line.

Excludes kneeldowns and spikes.

Usage:
  python offense_efficiency_scatter.py --season 2025 --through-week 11 --out out.png
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
TEAMS32 = [
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB",
    "HOU","IND","JAX","KC","LA","LAC","LV","MIA","MIN","NE","NO","NYG","NYJ",
    "PHI","PIT","SEA","SF","TB","TEN","WAS",
]


def load_team_meta():
    return nfl.load_teams().filter(pl.col("team_abbr").is_in(TEAMS32)).select(
        ["team_abbr", "team_logo_espn", "team_logo_squared"]
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
    plays = pbp.filter(
        (pl.col("season_type") == "REG")
        & (pl.col("week") <= through_week)
        & pl.col("posteam").is_not_null()
        & pl.col("epa").is_not_null()
        & pl.col("success").is_not_null()
        & (~pl.col("play_type").is_in(["qb_kneel", "qb_spike"]))
        & (pl.col("play_type").is_in(["pass", "run"]) | (pl.col("pass") == 1) | (pl.col("rush") == 1))
    )
    agg = (
        plays.group_by("posteam")
        .agg(
            [
                pl.col("epa").mean().alias("epa_play"),
                (100.0 * pl.col("success").mean()).alias("success_rate"),
                pl.len().alias("plays"),
            ]
        )
        .rename({"posteam": "team"})
        .filter(pl.col("team").is_in(TEAMS32))
    )
    meta = {
        "season": season,
        "through_week": through_week,
        "source": f"load_pbp({season}) REG 1-{through_week}; mean(epa), mean(success); exclude kneel/spike",
    }
    return agg, meta


def plot(df, logos, meta, out):
    rows = df.to_dicts()
    xs = np.array([r["epa_play"] for r in rows])
    ys = np.array([r["success_rate"] for r in rows])
    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor("white")
    # trend
    coef = np.polyfit(xs, ys, 1)
    xline = np.linspace(xs.min() - 0.05, xs.max() + 0.05, 50)
    ax.plot(xline, coef[0] * xline + coef[1], "k--", linewidth=2.0, zorder=2)
    for r in rows:
        path = logos.get(r["team"])
        if not path:
            continue
        img = Image.open(path).convert("RGBA")
        img.thumbnail((80, 80), Image.Resampling.LANCZOS)
        im = OffsetImage(np.asarray(img), zoom=22.0 / img.size[1])
        ax.add_artist(AnnotationBbox(im, (r["epa_play"], r["success_rate"]), frameon=False, pad=0, zorder=3))
    ax.set_xlabel("Offensive EPA/Play", fontsize=12)
    ax.set_ylabel("Offensive Success Rate", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))
    ax.grid(True, color="#DDD", zorder=0)
    # annotations relative to trend
    ax.text(
        xs.min() + 0.02,
        ys.max() - 1,
        "More consistent, less explosive",
        fontsize=10,
        fontweight="bold",
        color="#222",
    )
    ax.text(
        xs.max() - 0.12,
        ys.min() + 1,
        "More explosive, less consistent",
        fontsize=10,
        fontweight="bold",
        color="#222",
    )
    fig.text(0.06, 0.97, "Offensive EPA/play and success rate",
             fontsize=15, fontweight="bold", va="top")
    fig.text(0.06, 0.935, "Excludes kneeldowns and spikes", fontsize=10, color="#444", va="top")
    fig.text(0.98, 0.02, f"Data: nflverse {meta['source']}", ha="right", fontsize=8, color="#666")
    plt.tight_layout(rect=[0.04, 0.04, 0.98, 0.90])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor="white")
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
