#!/usr/bin/env python
"""QB EPA/play in expected passing situations (xpass > 0.70).

Usage:
  python qb_expected_pass_epa.py --season 2025 --through-week 11 --out out.png
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
from matplotlib.patches import FancyBboxPatch
from PIL import Image

HERE = Path(__file__).resolve().parent
LOGO_DIR = HERE / "logos"


def _hex(c, fb="#333333"):
    if not c:
        return fb
    c = str(c).strip()
    return ("#" + c if not c.startswith("#") else c) if len(c) in (6, 7) else fb


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


def compute(season, through_week, min_db=100, xpass_thr=0.70):
    pbp = nfl.load_pbp(seasons=season)
    plays = pbp.filter(
        (pl.col("season_type") == "REG")
        & (pl.col("week") <= through_week)
        & pl.col("xpass").is_not_null()
        & (pl.col("xpass") > xpass_thr)
        & pl.col("passer_player_id").is_not_null()
        & pl.col("epa").is_not_null()
        & (
            (pl.col("qb_dropback") == 1)
            | (pl.col("pass") == 1)
            | (pl.col("rush_attempt") == 1)
        )
    )
    # Prefer qb_dropback when available
    if "qb_dropback" in plays.columns:
        drops = plays.filter(pl.col("qb_dropback") == 1)
        if drops.height >= min_db:
            plays = drops

    agg = (
        plays.group_by(["passer_player_id", "passer_player_name"])
        .agg(
            [
                pl.col("epa").mean().alias("epa_per_play"),
                pl.len().alias("dropbacks"),
                pl.col("posteam").last().alias("team"),
            ]
        )
        .filter(pl.col("dropbacks") >= min_db)
        .sort("epa_per_play", descending=True)
    )
    try:
        rost = nfl.load_rosters(seasons=season).select(
            pl.col("gsis_id").alias("passer_player_id"),
            pl.col("full_name").alias("full_name"),
        ).unique(subset=["passer_player_id"], keep="last")
        agg = agg.join(rost, on="passer_player_id", how="left").with_columns(
            pl.coalesce(["full_name", "passer_player_name"]).alias("player")
        )
    except Exception:
        agg = agg.with_columns(pl.col("passer_player_name").alias("player"))

    meta = {
        "season": season,
        "through_week": through_week,
        "min_db": min_db,
        "xpass_thr": xpass_thr,
        "source": f"load_pbp({season}) REG; xpass>{xpass_thr}; qb_dropback==1; mean(epa)",
    }
    return agg, meta


def short(name):
    parts = str(name).split()
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0][0]}. {' '.join(parts[1:])}"


def plot(df, teams, logos, meta, out):
    rows = df.to_dicts()
    colors = {
        r["team_abbr"]: (_hex(r["team_color"]), _hex(r["team_color2"], "#222"))
        for r in teams.to_dicts()
    }
    fig_h = max(8, 0.38 * len(rows) + 2.2)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    fig.patch.set_facecolor("white")
    ys = np.arange(len(rows))
    xs = [float(r["epa_per_play"]) for r in rows]
    for i, r in enumerate(rows):
        c1, c2 = colors.get(r["team"], ("#444", "#222"))
        ax.barh(i, xs[i], height=0.7, color=c1, edgecolor=c2, linewidth=1.2, zorder=2)
        path = logos.get(r["team"])
        if path:
            img = Image.open(path).convert("RGBA")
            img.thumbnail((80, 80), Image.Resampling.LANCZOS)
            im = OffsetImage(np.asarray(img), zoom=16.0 / img.size[1])
            x_logo = xs[i] + (0.012 if xs[i] >= 0 else -0.012)
            ab = AnnotationBbox(
                im, (x_logo, i), frameon=False, box_alignment=(0 if xs[i] >= 0 else 1, 0.5), pad=0
            )
            ax.add_artist(ab)
    ax.axvline(0, color="black", linewidth=1.4, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([short(r["player"]) for r in rows], fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#DDD", zorder=0)
    ax.set_xlabel("EPA per Play", fontsize=12)
    pad = 0.05
    ax.set_xlim(min(xs) - pad - 0.05, max(xs) + pad + 0.08)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.text(
        0.05,
        0.97,
        "How quarterbacks perform in expected passing situations",
        fontsize=14,
        fontweight="bold",
        va="top",
    )
    fig.text(
        0.05,
        0.935,
        f"Expected passing situation is when expected pass probability is greater than "
        f"{int(meta['xpass_thr']*100)}% | Minimum {meta['min_db']} dropbacks",
        fontsize=9,
        color="#444",
        va="top",
    )
    fig.text(
        0.98,
        0.015,
        f"Data: nflverse load_pbp {meta['season']} REG | {meta['source']}",
        ha="right",
        fontsize=7.5,
        color="#666",
    )
    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.90])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--through-week", type=int, default=11)
    ap.add_argument("--min-dropbacks", type=int, default=100)
    ap.add_argument("--xpass-threshold", type=float, default=0.70)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    df, meta = compute(args.season, args.through_week, args.min_dropbacks, args.xpass_threshold)
    teams = load_team_meta()
    logos = download_logos(teams)
    plot(df, teams, logos, meta, args.out)
    print("wrote", args.out, "n=", df.height)


if __name__ == "__main__":
    main()
