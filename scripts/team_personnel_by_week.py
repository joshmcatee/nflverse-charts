#!/usr/bin/env python
"""Faceted stacked area: team offensive personnel usage by week (11/12/21/Other).

Personnel parsed from nflverse load_participation().offense_personnel
(RB/WR/TE counts on plays with a QB). Ref chart cites Trumedia; this is the
nflverse proxy — labeled as such.

Usage:
  python team_personnel_by_week.py --season 2025 --through-week 11 --out out.png
"""
from __future__ import annotations

import argparse
import io
import re
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
COLORS = {"11": "#C5B6E0", "12": "#7EB8B0", "21": "#7CB87C", "Other": "#F0C4A8"}
STACK = ["11", "12", "21", "Other"]


def classify_personnel(s: str | None) -> str | None:
    if not s:
        return None
    # Skip special teams / defense-heavy strings
    if re.search(r"\b(CB|P|K|LS|FS|SS|ILB|OLB|MLB|DE|DT|NT)\b", s):
        # still ok if also has classic OL+QB pattern
        if not re.search(r"\bQB\b", s):
            return None
    if not re.search(r"\bQB\b", s):
        return None
    def cnt(pos):
        m = re.search(rf"(\d+)\s*{pos}\b", s)
        return int(m.group(1)) if m else 0
    rb = cnt("RB") + cnt("FB")
    wr = cnt("WR")
    te = cnt("TE")
    if rb == 1 and wr == 3 and te == 1:
        return "11"
    if rb == 1 and wr == 2 and te == 2:
        return "12"
    if rb == 2 and wr == 2 and te == 1:
        return "21"
    # if no skill counts at all, skip
    if rb + wr + te == 0:
        return None
    return "Other"


def load_team_meta():
    return nfl.load_teams().filter(pl.col("team_abbr").is_in(TEAM_ORDER)).select(
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
    part = nfl.load_participation(seasons=season)
    part = part.with_columns(
        pl.col("nflverse_game_id").str.extract(r"_(\d{2})_", 1).cast(pl.Int64).alias("week"),
        pl.col("possession_team").alias("team"),
    ).filter(
        (pl.col("week") <= through_week)
        & pl.col("team").is_in(TEAM_ORDER)
        & pl.col("offense_personnel").is_not_null()
    )
    # classify in python for clarity
    rows = part.select(["team", "week", "offense_personnel"]).to_dicts()
    out_rows = []
    for r in rows:
        cat = classify_personnel(r["offense_personnel"])
        if cat:
            out_rows.append({"team": r["team"], "week": r["week"], "pers": cat})
    df = pl.DataFrame(out_rows)
    counts = df.group_by(["team", "week", "pers"]).len().rename({"len": "n"})
    tot = df.group_by(["team", "week"]).len().rename({"len": "tot"})
    rates = counts.join(tot, on=["team", "week"]).with_columns(
        (100.0 * pl.col("n") / pl.col("tot")).alias("pct")
    )
    meta = {
        "season": season,
        "through_week": through_week,
        "source": f"load_participation({season}) offense_personnel; weeks 1-{through_week}",
        "n_plays": len(out_rows),
    }
    return rates, meta


def plot(df, logos, meta, out):
    weeks = list(range(1, meta["through_week"] + 1))
    fig, axes = plt.subplots(4, 8, figsize=(18, 10), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    lookup = {(r["team"], r["week"], r["pers"]): r["pct"] for r in df.to_dicts()}
    for i, team in enumerate(TEAM_ORDER):
        ax = axes[i // 8][i % 8]
        bottoms = np.zeros(len(weeks))
        for pers in STACK:
            vals = np.array([lookup.get((team, w, pers), 0.0) for w in weeks])
            ax.fill_between(weeks, bottoms, bottoms + vals, color=COLORS[pers], linewidth=0)
            bottoms = bottoms + vals
        ax.set_xlim(1, meta["through_week"])
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))
        for sp in ax.spines.values():
            sp.set_color("#CCC")
        path = logos.get(team)
        if path:
            img = Image.open(path).convert("RGBA")
            img.thumbnail((80, 80), Image.Resampling.LANCZOS)
            im = OffsetImage(np.asarray(img), zoom=14.0 / img.size[1])
            ax.add_artist(AnnotationBbox(im, (0.5, 1.12), xycoords="axes fraction", frameon=False, pad=0))
        if i % 8 == 0:
            ax.set_ylabel("% of Plays", fontsize=8)
        if i // 8 == 3:
            ax.set_xlabel("Week", fontsize=8)
    # legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in STACK]
    fig.legend(handles, ["11 personnel", "12 personnel", "21 personnel", "Other"],
               loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Team personnel usage, by week", fontsize=16, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.955,
        "11 personnel: 1 RB, 3 WR, and 1 TE | 12 personnel: 1 RB, 2 WR, and 2 TE | 21 personnel: 2 RB, 2 WR, and 1 TE",
        ha="center",
        fontsize=9,
        color="#444",
    )
    fig.text(0.99, 0.01, f"Data: nflverse {meta['source']} (proxy for Trumedia personnel)", ha="right", fontsize=7.5, color="#666")
    plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.94])
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
    print("wrote", args.out, meta)


if __name__ == "__main__":
    main()
