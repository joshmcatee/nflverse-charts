#!/usr/bin/env python
"""Team actual pass rate vs expected pass rate (PROE scatter with logos).

Usage:
  python proe_scatter.py --season 2025 --through-week 11 --out out.png
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
    "BUF","MIA","NE","NYJ","BAL","CIN","CLE","PIT","HOU","IND","JAX","TEN",
    "DEN","KC","LAC","LV","DAL","NYG","PHI","WAS","CHI","DET","GB","MIN",
    "ATL","CAR","NO","TB","ARI","LA","SEA","SF",
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
        & pl.col("xpass").is_not_null()
        & pl.col("pass").is_not_null()
    )
    agg = (
        plays.group_by("posteam")
        .agg(
            [
                (100.0 * pl.col("pass").mean()).alias("actual"),
                (100.0 * pl.col("xpass").mean()).alias("expected"),
                (100.0 * pl.col("pass_oe").mean()).alias("proe"),
                pl.len().alias("plays"),
            ]
        )
        .rename({"posteam": "team"})
        .filter(pl.col("team").is_in(TEAMS32))
    )
    meta = {
        "season": season,
        "through_week": through_week,
        "source": f"load_pbp({season}) REG weeks1-{through_week}; mean(pass), mean(xpass), mean(pass_oe)",
        "plays": int(plays.height),
    }
    return agg, meta


def plot(df, logos, meta, out):
    rows = df.to_dicts()
    fig, ax = plt.subplots(figsize=(10.5, 10))
    fig.patch.set_facecolor("white")
    xs = [r["expected"] for r in rows]
    ys = [r["actual"] for r in rows]
    lo = min(min(xs), min(ys)) - 1
    hi = max(max(xs), max(ys)) + 1
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    # diagonal guides
    for d in range(-20, 21, 5):
        ax.plot([lo, hi], [lo + d, hi + d], color="#CFCFCF" if d else "black",
                linewidth=2.2 if d == 0 else 0.8, zorder=1)
    for r in rows:
        path = logos.get(r["team"])
        if not path:
            ax.plot(r["expected"], r["actual"], "o", color="#333")
            continue
        img = Image.open(path).convert("RGBA")
        img.thumbnail((80, 80), Image.Resampling.LANCZOS)
        im = OffsetImage(np.asarray(img), zoom=22.0 / img.size[1])
        ax.add_artist(AnnotationBbox(im, (r["expected"], r["actual"]), frameon=False, pad=0, zorder=3))
    ax.set_xlabel("Expected Pass Rate", fontsize=12)
    ax.set_ylabel("Actual Pass Rate", fontsize=12)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))
    ax.text(lo + 1.5, hi - 2, "Throw more than expected", fontsize=11, fontweight="bold", color="#222")
    ax.text(hi - 18, lo + 2, "Throw less than expected", fontsize=11, fontweight="bold", color="#222")
    fig.text(0.06, 0.97, "Pass rate over expected",
             fontsize=15, fontweight="bold", va="top")
    fig.text(0.06, 0.935,
             "Team near black line with PROE of 0% | Expected pass rate based on nflfastR's model",
             fontsize=9, color="#444", va="top")
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
