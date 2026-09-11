#!/usr/bin/env python
"""Faceted team PROE by week (division-ordered small multiples).

Usage:
  python team_proe_by_week.py --season 2025 --through-week 11 --out out.png
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


def _hex(c, fb="#333"):
    if not c:
        return fb
    c = str(c).strip()
    if not c.startswith("#"):
        c = "#" + c
    return c if len(c) == 7 else fb


def load_team_meta():
    return nfl.load_teams().filter(pl.col("team_abbr").is_in(TEAM_ORDER)).select(
        ["team_abbr", "team_color", "team_logo_espn", "team_logo_squared"]
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
        & pl.col("pass_oe").is_not_null()
    )
    weekly = (
        plays.group_by(["posteam", "week"])
        .agg((100.0 * pl.col("pass_oe").mean()).alias("proe"))
        .rename({"posteam": "team"})
        .filter(pl.col("team").is_in(TEAM_ORDER))
    )
    meta = {
        "season": season,
        "through_week": through_week,
        "source": f"load_pbp({season}) REG; weekly mean(pass_oe)*100",
    }
    return weekly, meta


def plot(df, teams, logos, meta, out):
    colors = {r["team_abbr"]: _hex(r["team_color"]) for r in teams.to_dicts()}
    weeks = list(range(1, meta["through_week"] + 1))
    fig, axes = plt.subplots(4, 8, figsize=(18, 10), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    by = {(r["team"], r["week"]): r["proe"] for r in df.to_dicts()}
    for i, team in enumerate(TEAM_ORDER):
        ax = axes[i // 8][i % 8]
        ys = [by.get((team, w), np.nan) for w in weeks]
        ax.axhline(0, color="black", linestyle=":", linewidth=1)
        ax.plot(weeks, ys, color=colors.get(team, "#333"), linewidth=1.8)
        ax.set_xlim(0.5, meta["through_week"] + 0.5)
        ax.set_ylim(-25, 20)
        ax.set_xticks([1, 5, 9] if meta["through_week"] >= 9 else weeks)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.6)
        for sp in ax.spines.values():
            sp.set_color("#CCC")
        path = logos.get(team)
        if path:
            img = Image.open(path).convert("RGBA")
            img.thumbnail((80, 80), Image.Resampling.LANCZOS)
            im = OffsetImage(np.asarray(img), zoom=14.0 / img.size[1])
            ab = AnnotationBbox(im, (0.5, 1.12), xycoords="axes fraction", frameon=False, pad=0)
            ax.add_artist(ab)
        else:
            ax.set_title(team, fontsize=8)
        if i % 8 == 0:
            ax.set_ylabel("PROE", fontsize=8)
        if i // 8 == 3:
            ax.set_xlabel("Week", fontsize=8)
    fig.suptitle("Team pass rate over expectation, by week", fontsize=16, fontweight="bold", y=0.98)
    fig.text(0.5, 0.955, "Dotted line indicates league average PROE of 0%", ha="center", fontsize=10, color="#444")
    fig.text(0.99, 0.01, f"Data: nflverse {meta['source']}", ha="right", fontsize=8, color="#666")
    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.94])
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
    teams = load_team_meta()
    logos = download_logos(teams)
    plot(df, teams, logos, meta, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
