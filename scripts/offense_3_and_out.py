#!/usr/bin/env python
"""Offensive 3-and-out rate bar chart (Sam Hoppen–style).

Series-level definition (matches Hoppen subtitle):
  Three-and-out rates include any series of a drive that results in a turnover
  or failure to gain a first down or touchdown.

  Rate = unsuccessful series / all offensive series
  Unsuccessful = series_success == 0  (nflfastR: not first down and not TD)
  Excludes QB-kneel series.

Data: nflverse via nflreadpy load_pbp.

Usage:
  python offense_3_and_out.py
  python offense_3_and_out.py --season 2025 --through-week 11
  python offense_3_and_out.py --season 2025 --through-week 11 --out out.png
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import nflreadpy as nfl
import numpy as np
import polars as pl
import requests
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

HERE = Path(__file__).resolve().parent
LOGO_DIR = HERE / "logos"
TEAMS32 = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB", "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA",
    "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB",
    "TEN", "WAS",
]


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
        .filter(pl.col("team_abbr").is_in(TEAMS32))
        .select(
            [
                "team_abbr",
                "team_color",
                "team_color2",
                "team_logo_espn",
                "team_logo_squared",
            ]
        )
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
                Image.open(io.BytesIO(r.content)).verify()
                dest.write_bytes(r.content)
                paths[abbr] = dest
                break
            except Exception:
                continue
    return paths


def compute(season: int, through_week: int):
    """Series-level 3-and-out rate from load_pbp.

    One row per (game_id, posteam, series). Unsuccessful = series_success == 0
    (turnover or failure to gain a first down / touchdown). QB-kneel series
    excluded.
    """
    pbp = nfl.load_pbp(seasons=season)
    pbp = pbp.filter(
        (pl.col("season_type") == "REG")
        & (pl.col("week") <= through_week)
        & pl.col("posteam").is_not_null()
        & pl.col("series").is_not_null()
        & pl.col("series_success").is_not_null()
    )

    series = (
        pbp.group_by(["game_id", "posteam", "series"])
        .agg(
            pl.col("series_success").first().alias("series_success"),
            pl.col("series_result").first().alias("series_result"),
            pl.col("week").first().alias("week"),
        )
        .filter(pl.col("series_result") != "QB kneel")
        .filter(pl.col("posteam").is_in(TEAMS32))
    )

    df = (
        series.group_by("posteam")
        .agg(
            pl.len().alias("n_series"),
            (pl.col("series_success") == 0).sum().alias("n_unsuccessful"),
            (pl.col("series_success") == 1).sum().alias("n_successful"),
        )
        .with_columns(
            (pl.col("n_unsuccessful") / pl.col("n_series")).alias("rate"),
            (pl.col("n_unsuccessful") / pl.col("n_series") * 100).alias("rate_pct"),
        )
        .rename({"posteam": "team"})
        .sort("rate", descending=True)
    )

    meta = {
        "season": season,
        "through_week": through_week,
        "source": (
            f"nfl.load_pbp(seasons={season}) REG weeks 1-{through_week}; "
            "series-level: rate = (series_success==0) / series; excl QB kneel"
        ),
        "n_series": int(series.height),
        "definition": (
            "series_success==0 (turnover or failure to gain first down/TD) "
            "/ all non-kneel offensive series"
        ),
    }
    return df, meta


def plot(df, teams, logos, meta, out: Path):
    rows = df.to_dicts()
    colors = {
        r["team_abbr"]: (_hex(r["team_color"]), _hex(r["team_color2"], "#222"))
        for r in teams.to_dicts()
    }

    fig, ax = plt.subplots(figsize=(14, 7.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    xs = np.arange(len(rows))
    ys = [float(r["rate_pct"]) for r in rows]

    for i, r in enumerate(rows):
        c1, c2 = colors.get(r["team"], ("#444", "#222"))
        ax.bar(
            i,
            ys[i],
            color=c1,
            edgecolor=c2,
            linewidth=1.15,
            width=0.72,
            zorder=2,
        )

    # Logos under bars only (Hoppen style)
    for i, r in enumerate(rows):
        path = logos.get(r["team"])
        if not path:
            continue
        img = Image.open(path).convert("RGBA")
        img.thumbnail((80, 80), Image.Resampling.LANCZOS)
        im = OffsetImage(np.asarray(img), zoom=16.0 / img.size[1])
        ax.add_artist(
            AnnotationBbox(
                im,
                (i, 0),
                frameon=False,
                box_alignment=(0.5, 1),
                pad=0,
                zorder=3,
                xybox=(0, -4),
                boxcoords="offset points",
            )
        )

    ax.set_xticks([])
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ymax = max(40.0, max(ys) * 1.08)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Three-and-out Rate", fontsize=12)
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
    # No horizontal grid — match Hoppen minimal look
    ax.grid(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#333")
    ax.spines["bottom"].set_color("#333")
    ax.tick_params(axis="y", labelsize=10, colors="#333")

    fig.text(
        0.06,
        0.965,
        "Offensive 3-and-out rate",
        fontsize=15,
        fontweight="bold",
        va="top",
        color="#111",
    )
    fig.text(
        0.06,
        0.918,
        "Three-and-out rates include any series of a drive that results in a "
        "turnover or failure to gain a first down or touchdown",
        fontsize=9.5,
        color="#444",
        va="top",
    )
    fig.text(
        0.98,
        0.018,
        "Figure: Quant | Data: nflverse",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#666",
    )

    plt.tight_layout(rect=[0.04, 0.08, 0.98, 0.88])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Offensive 3-and-out rate bar chart (series-level, nflverse)"
    )
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--through-week", type=int, default=11)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="PNG path (default: offense_3_and_out_{season}_w01-{ww}.png next to script)",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV path (default: same stem as --out)",
    )
    args = ap.parse_args()

    ww = f"{args.through_week:02d}"
    default_stem = HERE / f"offense_3_and_out_{args.season}_w01-{ww}"
    out = args.out or Path(str(default_stem) + ".png")
    csv_path = args.csv or out.with_suffix(".csv")

    df, meta = compute(args.season, args.through_week)
    teams = load_team_meta()
    logos = download_logos(teams)
    plot(df, teams, logos, meta, out)

    df.write_csv(csv_path)

    top3 = df.head(3).to_dicts()
    bot3 = df.tail(3).to_dicts()
    print("wrote", out)
    print("wrote", csv_path)
    print("meta:", meta)
    print("TOP3:")
    for r in top3:
        print(f"  {r['team']}: {r['rate_pct']:.2f}% ({r['n_unsuccessful']}/{r['n_series']})")
    print("BOTTOM3:")
    for r in bot3:
        print(f"  {r['team']}: {r['rate_pct']:.2f}% ({r['n_unsuccessful']}/{r['n_series']})")


if __name__ == "__main__":
    main()
