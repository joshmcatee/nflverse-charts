#!/usr/bin/env python
"""Combined target share of top-3 players per team (stacked horizontal bars).

Target share for a player = targets in games player had a target-or-rush-or-rec
snap proxy (appears in PBP as receiver/rusher/passer) / team pass attempts
in those same games where player was "active" on offense.

Practical nflverse definition matching common SamHoppen-style charts:
  - Player targets from PBP (pass_attempt==1 & receiver_player_id not null)
  - Active game = any game with targets + carries + receptions for that player
  - Denominator = team pass attempts (posteam) in those active games only
  - Min 3 active games; top 3 by target share per team; stack & sort by sum

Usage:
  python top3_target_share.py --season 2025 --through-week 11
"""
from __future__ import annotations

import argparse
import io
import sys
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
SEG_COLORS = ["#8FCB8F", "#6A8FC0", "#F0C89A"]


def load_team_meta() -> pl.DataFrame:
    return nfl.load_teams().select(
        ["team_abbr", "team_logo_espn", "team_logo_squared"]
    )


def download_logos(teams: pl.DataFrame) -> dict[str, Path]:
    LOGO_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
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
                Image.open(io.BytesIO(r.content)).verify()
                paths[abbr] = dest
                break
            except Exception:
                continue
    return paths


def compute(season: int, through_week: int, min_games: int = 3) -> tuple[pl.DataFrame, dict]:
    pbp = nfl.load_pbp(seasons=season)
    pbp = pbp.filter(
        (pl.col("season_type") == "REG")
        & (pl.col("week") <= through_week)
        & pl.col("posteam").is_not_null()
    )

    # Team pass attempts per game
    team_att = (
        pbp.filter(pl.col("pass_attempt") == 1)
        .group_by(["posteam", "game_id", "week"])
        .agg(pl.len().alias("team_pass_att"))
        .rename({"posteam": "team"})
    )

    # Player targets
    tgts = (
        pbp.filter(
            (pl.col("pass_attempt") == 1) & pl.col("receiver_player_id").is_not_null()
        )
        .group_by(["receiver_player_id", "receiver_player_name", "posteam", "game_id", "week"])
        .agg(pl.len().alias("targets"))
        .rename(
            {
                "receiver_player_id": "player_id",
                "receiver_player_name": "player",
                "posteam": "team",
            }
        )
    )

    # Active games: any offensive touch (target, rush, reception involvement)
    rush = (
        pbp.filter(
            (pl.col("rush_attempt") == 1)
            & pl.col("rusher_player_id").is_not_null()
            & (pl.col("play_type") != "qb_kneel")
        )
        .select(
            pl.col("rusher_player_id").alias("player_id"),
            pl.col("rusher_player_name").alias("player"),
            pl.col("posteam").alias("team"),
            "game_id",
            "week",
        )
        .unique()
    )
    active = (
        pl.concat(
            [
                tgts.select(["player_id", "player", "team", "game_id", "week"]),
                rush,
            ],
            how="diagonal_relaxed",
        )
        .unique(subset=["player_id", "game_id"])
    )

    # Join targets (0 if active but no targets) + team attempts for active games
    active_att = active.join(team_att, on=["team", "game_id", "week"], how="left")
    active_att = active_att.join(
        tgts.select(["player_id", "game_id", "targets"]),
        on=["player_id", "game_id"],
        how="left",
    ).with_columns(pl.col("targets").fill_null(0))

    per_player = (
        active_att.group_by(["player_id", "player", "team"])
        .agg(
            [
                pl.col("game_id").n_unique().alias("games"),
                pl.col("targets").sum().alias("targets"),
                pl.col("team_pass_att").sum().alias("team_pass_att"),
            ]
        )
        .filter(
            (pl.col("games") >= min_games)
            & (pl.col("team_pass_att") > 0)
            & (pl.col("targets") > 0)
        )
        .with_columns(
            (100.0 * pl.col("targets") / pl.col("team_pass_att")).alias("tgt_share")
        )
    )

    # Prefer roster full names
    try:
        rost = nfl.load_rosters(seasons=season).select(
            pl.col("gsis_id").alias("player_id"),
            pl.col("full_name").alias("full_name"),
        ).unique(subset=["player_id"], keep="last")
        per_player = per_player.join(rost, on="player_id", how="left").with_columns(
            pl.coalesce(["full_name", "player"]).alias("player")
        )
    except Exception:
        pass

    # Top 3 per team
    ranked = (
        per_player.sort(["team", "tgt_share"], descending=[False, True])
        .with_columns(pl.col("tgt_share").rank(method="ordinal", descending=True).over("team").alias("rk"))
        .filter(pl.col("rk") <= 3)
    )

    meta = {
        "season": season,
        "through_week": through_week,
        "min_games": min_games,
        "source": f"nflverse load_pbp({season}) REG weeks 1-{through_week}; "
        "share = player targets / team pass_attempt in player's active games",
    }
    return ranked, meta


def short_name(name: str) -> str:
    parts = str(name).replace(".", "").split()
    if len(parts) <= 1:
        return name
    # keep Jr/II with last
    return f"{parts[0][0]}{' '.join(parts[1:])}" if False else (
        f"{parts[0][0]}. {' '.join(parts[1:])}"
    )


def plot(df: pl.DataFrame, logos: dict[str, Path], meta: dict, out: Path) -> None:
    teams = (
        df.group_by("team")
        .agg(pl.col("tgt_share").sum().alias("total"))
        .sort("total", descending=True)
    )
    order = teams["team"].to_list()
    totals = {r["team"]: r["total"] for r in teams.to_dicts()}

    fig_h = max(10, 0.38 * len(order) + 2)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_pos = np.arange(len(order))
    left = np.zeros(len(order))

    # Build segment data
    by_team = {t: [] for t in order}
    for r in df.to_dicts():
        if r["team"] in by_team:
            by_team[r["team"]].append(r)
    for t in order:
        by_team[t] = sorted(by_team[t], key=lambda x: x["rk"])

    for seg_i in range(3):
        widths = []
        labels = []
        for t in order:
            players = by_team[t]
            if seg_i < len(players):
                w = float(players[seg_i]["tgt_share"])
                nm = short_name(players[seg_i]["player"])
                labels.append(f"{nm} ({w:.1f}%)")
                widths.append(w)
            else:
                labels.append("")
                widths.append(0.0)
        bars = ax.barh(
            y_pos,
            widths,
            left=left,
            height=0.72,
            color=SEG_COLORS[seg_i],
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )
        for i, (bar, lab, w) in enumerate(zip(bars, labels, widths)):
            if w >= 4.5 and lab:
                ax.text(
                    left[i] + w / 2,
                    bar.get_y() + bar.get_height() / 2,
                    lab,
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="#222",
                    zorder=3,
                )
            elif w >= 2.5 and lab:
                ax.text(
                    left[i] + w / 2,
                    bar.get_y() + bar.get_height() / 2,
                    lab,
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="#222",
                    zorder=3,
                )
        left = left + np.array(widths)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([""] * len(order))
    ax.set_xlim(0, max(85, float(max(totals.values())) + 5))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))
    ax.set_xlabel("Target Share", fontsize=12)
    ax.grid(axis="x", color="#D0D0D0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    for sp in ax.spines.values():
        sp.set_color("#888")

    # Logos on y
    for i, t in enumerate(order):
        path = logos.get(t)
        if not path:
            ax.text(-1.5, i, t, ha="right", va="center", fontsize=8)
            continue
        img = Image.open(path).convert("RGBA")
        img.thumbnail((80, 80), Image.Resampling.LANCZOS)
        im = OffsetImage(np.asarray(img), zoom=18.0 / img.size[1])
        ab = AnnotationBbox(im, (0, i), xybox=(-18, 0), xycoords=("data", "data"),
                            boxcoords="offset points", frameon=False, pad=0)
        ax.add_artist(ab)

    tw = meta["through_week"]
    fig.text(0.06, 0.97, "Combined target share of the top 3 players",
             fontsize=15, fontweight="bold", va="top")
    fig.text(
        0.06,
        0.935,
        f"Minimum {meta['min_games']} games | Target shares only consider games that player was active | "
        "Players out indefinitely removed from bar",
        fontsize=9,
        color="#444",
        va="top",
    )
    fig.text(
        0.98,
        0.01,
        f"Data: nflverse load_pbp {meta['season']} REG 1-{tw}",
        ha="right",
        fontsize=8,
        color="#666",
    )
    plt.tight_layout(rect=[0.04, 0.03, 1, 0.91])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--through-week", type=int, default=11)
    ap.add_argument("--min-games", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    df, meta = compute(args.season, args.through_week, args.min_games)
    logos = download_logos(load_team_meta())
    plot(df, logos, meta, args.out)
    print("wrote", args.out, "teams", df["team"].n_unique(), meta)


if __name__ == "__main__":
    main()
