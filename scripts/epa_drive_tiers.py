#!/usr/bin/env python
"""EPA per drive team tiers scatter (offense X, defense Y inverted).

Excludes kneeldown drives. Diagonal lines = constant net EPA/drive.

Usage:
  python epa_drive_tiers.py --season 2025 --through-week 11 --out out.png
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib.pyplot as plt
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
    pbp = pbp.filter(
        (pl.col("season_type") == "REG")
        & (pl.col("week") <= through_week)
        & pl.col("epa").is_not_null()
        & pl.col("posteam").is_not_null()
        & pl.col("fixed_drive").is_not_null()
    )
    # Flag kneel drives: any qb_kneel on the drive
    kneel_drives = (
        pbp.filter(pl.col("play_type") == "qb_kneel")
        .select(["game_id", "fixed_drive", "posteam"])
        .unique()
        .with_columns(pl.lit(True).alias("is_kneel"))
    )
    drives = (
        pbp.group_by(["game_id", "fixed_drive", "posteam", "defteam"])
        .agg(pl.col("epa").sum().alias("drive_epa"))
        .join(kneel_drives, on=["game_id", "fixed_drive", "posteam"], how="left")
        .filter(pl.col("is_kneel").is_null())
    )
    off = (
        drives.group_by("posteam")
        .agg(pl.col("drive_epa").mean().alias("off_epa_drive"))
        .rename({"posteam": "team"})
    )
    deff = (
        drives.group_by("defteam")
        .agg(pl.col("drive_epa").mean().alias("def_epa_drive"))
        .rename({"defteam": "team"})
    )
    df = off.join(deff, on="team").filter(pl.col("team").is_in(TEAMS32))
    meta = {
        "season": season,
        "through_week": through_week,
        "source": f"load_pbp({season}) REG 1-{through_week}; mean drive EPA; exclude qb_kneel drives",
        "n_drives": int(drives.height),
    }
    return df, meta


def plot(df, logos, meta, out):
    rows = df.to_dicts()
    fig, ax = plt.subplots(figsize=(11, 10))
    fig.patch.set_facecolor("white")
    xs = [r["off_epa_drive"] for r in rows]
    ys = [r["def_epa_drive"] for r in rows]
    xlo, xhi = min(xs) - 0.2, max(xs) + 0.2
    ylo, yhi = min(ys) - 0.2, max(ys) + 0.2
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(yhi, ylo)  # invert: strong defense (low EPA allowed) at top
    # diagonal net tiers: off - def = C  =>  def = off - C
    for c in np.arange(-2.5, 2.6, 0.5):
        ax.plot([xlo, xhi], [xlo - c, xhi - c], color="#C8C8C8", linewidth=0.9, zorder=1)
    for r in rows:
        path = logos.get(r["team"])
        if not path:
            continue
        img = Image.open(path).convert("RGBA")
        img.thumbnail((80, 80), Image.Resampling.LANCZOS)
        im = OffsetImage(np.asarray(img), zoom=22.0 / img.size[1])
        ax.add_artist(AnnotationBbox(im, (r["off_epa_drive"], r["def_epa_drive"]), frameon=False, pad=0, zorder=3))
    ax.set_xlabel("Weak Offense  <---  Offensive EPA/Drive  --->  Strong Offense", fontsize=11)
    ax.set_ylabel("Weak Defense  <---  Defensive EPA/Drive  --->  Strong Defense", fontsize=11)
    fig.text(0.06, 0.97, "EPA per drive team tiers",
             fontsize=15, fontweight="bold", va="top")
    fig.text(0.06, 0.935, "Excludes kneeldown drives", fontsize=10, color="#444", va="top")
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
    print("wrote", args.out, meta)


if __name__ == "__main__":
    main()
