#!/usr/bin/env python
"""Team high-value touches per game (vertical bars + logos).

HVT = carries inside the 10 (yardline_100 <= 10) + receptions.

Usage:
  python team_hvt_bar.py --season 2025 --through-week 11 --out out.png
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


def _hex(c, fb="#333"):
    if not c:
        return fb
    c = str(c).strip()
    if not c.startswith("#"):
        c = "#" + c
    return c if len(c) == 7 else fb


def load_team_meta():
    return nfl.load_teams().select(
        ["team_abbr", "team_color", "team_color2", "team_logo_espn", "team_logo_squared"]
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
        & pl.col("posteam").is_not_null()
        & (pl.col("play_type") != "qb_kneel")
    )
    inside = pbp.filter(
        (pl.col("rush_attempt") == 1)
        & (pl.col("yardline_100") <= 10)
        & pl.col("rusher_player_id").is_not_null()
    )
    recs = pbp.filter(pl.col("complete_pass") == 1)
    hvt = (
        pl.concat(
            [
                inside.select("posteam", "game_id"),
                recs.select("posteam", "game_id"),
            ]
        )
        .group_by("posteam")
        .agg(pl.len().alias("hvt"))
        .rename({"posteam": "team"})
    )
    games = (
        pbp.select(["posteam", "game_id"])
        .unique()
        .group_by("posteam")
        .len()
        .rename({"posteam": "team", "len": "games"})
    )
    df = hvt.join(games, on="team").with_columns(
        (pl.col("hvt") / pl.col("games")).alias("hvt_per_game")
    ).sort("hvt_per_game", descending=True)
    meta = {
        "season": season,
        "through_week": through_week,
        "source": f"load_pbp({season}) REG 1-{through_week}; HVT=inside-10 rushes + completions",
    }
    return df, meta


def plot(df, teams, logos, meta, out):
    rows = df.to_dicts()
    colors = {
        r["team_abbr"]: (_hex(r["team_color"]), _hex(r["team_color2"], "#222"))
        for r in teams.to_dicts()
    }
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("white")
    xs = np.arange(len(rows))
    ys = [float(r["hvt_per_game"]) for r in rows]
    for i, r in enumerate(rows):
        c1, c2 = colors.get(r["team"], ("#444", "#222"))
        ax.bar(i, ys[i], color=c1, edgecolor=c2, linewidth=1.1, width=0.75, zorder=2)
        path = logos.get(r["team"])
        if path:
            img = Image.open(path).convert("RGBA")
            img.thumbnail((80, 80), Image.Resampling.LANCZOS)
            im = OffsetImage(np.asarray(img), zoom=16.0 / img.size[1])
            ax.add_artist(AnnotationBbox(im, (i, ys[i]), frameon=False, box_alignment=(0.5, 0), pad=0, zorder=3,
                                         xybox=(0, 4), boxcoords="offset points"))
            ax.add_artist(AnnotationBbox(im, (i, 0), frameon=False, box_alignment=(0.5, 1), pad=0, zorder=3,
                                         xybox=(0, -2), boxcoords="offset points"))
    ax.set_xticks([])
    ax.set_ylabel("HVT/game", fontsize=12)
    ax.set_ylim(0, max(ys) * 1.18)
    ax.yaxis.set_major_locator(plt.MultipleLocator(2))
    ax.grid(axis="y", color="#DDD", zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.text(0.06, 0.96, "Total team high value touches per game",
             fontsize=15, fontweight="bold", va="top")
    fig.text(0.06, 0.915,
             "High value touches (HVT) include carries inside the ten and receptions",
             fontsize=10, color="#444", va="top")
    fig.text(0.98, 0.02, f"Data: nflverse {meta['source']}", ha="right", fontsize=8, color="#666")
    plt.tight_layout(rect=[0.03, 0.06, 0.98, 0.88])
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
    teams = load_team_meta()
    logos = download_logos(teams)
    plot(df, teams, logos, meta, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
