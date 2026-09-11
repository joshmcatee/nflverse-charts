#!/usr/bin/env python
"""WOPR (Weighted Opportunity Rating) horizontal bar chart for a 2026 NFL game.

WOPR = 1.5 * Target Share + 0.7 * Air Yards Share
  Target Share     = player targets / team pass targets (game)
  Air Yards Share  = player air yards / team air yards (game)

Targets: pass_attempt == 1 with receiver_player_id not null.
Player air yards: sum(air_yards) on those targeted plays (null -> 0).
Team air yards: sum(air_yards) on all pass_attempt == 1 for posteam (null -> 0),
matching nflfastR / nflverse player_stats air_yards_share denominator.

Data: nflverse via nflreadpy. WR only.

Usage:
  python wopr_chart.py --game-id 2026_01_NE_SEA
  python wopr_chart.py --season 2026 --week 1 --teams NE,SEA
  python wopr_chart.py --season 2026 --week 1 --game-id 2026_01_NE_SEA --out /path.png
"""
from __future__ import annotations

import argparse
import io
import re
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
BAR_COLOR = "#0B1F3A"
GAME_ID_RE = re.compile(r"^(\d{4})_(\d{2})_([A-Z0-9]+)_([A-Z0-9]+)$")


def _hex(c: str | None, fallback: str) -> str:
    if not c:
        return fallback
    c = str(c).strip()
    if not c.startswith("#"):
        c = "#" + c
    if len(c) == 7:
        return c
    return fallback


def load_team_meta() -> pl.DataFrame:
    teams = nfl.load_teams()
    return teams.select(
        [
            "team_abbr",
            "team_nick",
            "team_color",
            "team_color2",
            "team_logo_espn",
            "team_logo_squared",
        ]
    )


def download_logos(teams: pl.DataFrame, abbrs: list[str] | None = None) -> dict[str, Path]:
    LOGO_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    wanted = set(abbrs) if abbrs else None
    for row in teams.iter_rows(named=True):
        abbr = row["team_abbr"]
        if wanted is not None and abbr not in wanted:
            continue
        dest = LOGO_DIR / f"{abbr}.png"
        if dest.exists() and dest.stat().st_size > 0:
            paths[abbr] = dest
            continue
        urls = [row.get("team_logo_espn"), row.get("team_logo_squared")]
        ok = False
        for url in urls:
            if not url:
                continue
            try:
                r = requests.get(url, timeout=20)
                r.raise_for_status()
                dest.write_bytes(r.content)
                Image.open(io.BytesIO(r.content)).verify()
                paths[abbr] = dest
                ok = True
                break
            except Exception:
                continue
        if not ok:
            print(f"logo miss {abbr}", file=sys.stderr)
    return paths


def parse_game_id(game_id: str) -> tuple[int, int, str, str]:
    m = GAME_ID_RE.match(game_id)
    if not m:
        raise ValueError(
            f"game_id must look like 2026_01_NE_SEA, got {game_id!r}"
        )
    season, week, away, home = m.groups()
    return int(season), int(week), away, home


def resolve_selection(
    game_id: str | None,
    season: int | None,
    week: int | None,
    teams: list[str] | None,
) -> tuple[int, int | None, list[str] | None, str | None]:
    """Return season, week, team list, game_id."""
    if game_id:
        s, w, away, home = parse_game_id(game_id)
        return s, w, [away, home], game_id
    if season is None:
        raise ValueError("Provide --game-id or --season (with --week / --teams)")
    return season, week, teams, None


def _wr_positions() -> list[str]:
    return ["WR"]


def compute_wopr(
    season: int,
    week: int | None = None,
    teams: list[str] | None = None,
    game_id: str | None = None,
) -> tuple[pl.DataFrame, dict]:
    pbp = nfl.load_pbp(seasons=season)

    if week is not None and game_id is None:
        pbp = pbp.filter(pl.col("week") == week)

    # Resolve a single game_id when two teams are given (before dropping null-posteam plays)
    if game_id is None and teams and len(teams) == 2:
        ids = (
            pbp.filter(
                pl.col("posteam").is_in(teams) & pl.col("defteam").is_in(teams)
            )
            .select("game_id")
            .unique()
            .to_series()
            .to_list()
        )
        if len(ids) == 1:
            game_id = ids[0]

    if game_id:
        pbp = pbp.filter(pl.col("game_id") == game_id)
    elif teams:
        pbp = pbp.filter(
            pl.col("posteam").is_in(teams) | pl.col("defteam").is_in(teams)
        )

    if pbp.height == 0:
        raise RuntimeError("No plays found for the given selection")

    # Restrict offense rows to selected teams when filtering a matchup
    if teams:
        offense = pbp.filter(pl.col("posteam").is_in(teams))
    else:
        offense = pbp.filter(pl.col("posteam").is_not_null())

    passes = offense.filter(pl.col("pass_attempt") == 1)
    targeted = passes.filter(pl.col("receiver_player_id").is_not_null())

    # Team denominators: targets = targeted passes; air yards = all pass attempts
    team_targets = (
        targeted.group_by("posteam")
        .agg(pl.len().alias("team_targets"))
        .rename({"posteam": "team"})
    )
    team_air = (
        passes.group_by("posteam")
        .agg(pl.col("air_yards").fill_null(0).sum().alias("team_air_yards"))
        .rename({"posteam": "team"})
    )
    team_tot = team_targets.join(team_air, on="team", how="full", coalesce=True)

    player = targeted.group_by(
        ["receiver_player_id", "receiver_player_name", "posteam"]
    ).agg(
        pl.len().alias("targets"),
        pl.col("air_yards").fill_null(0).sum().alias("air_yards"),
    ).rename({"posteam": "team", "receiver_player_id": "player_id"})

    player = player.join(team_tot, on="team", how="left")
    player = player.with_columns(
        (pl.col("targets") / pl.col("team_targets")).alias("target_share"),
        pl.when(pl.col("team_air_yards") == 0)
        .then(0.0)
        .otherwise(pl.col("air_yards") / pl.col("team_air_yards"))
        .alias("air_yards_share"),
    )
    player = player.with_columns(
        (1.5 * pl.col("target_share") + 0.7 * pl.col("air_yards_share")).alias("wopr")
    )

    # Positions + display names
    players = nfl.load_players().select(
        [
            pl.col("gsis_id").alias("player_id"),
            pl.col("display_name").alias("player"),
            "position",
        ]
    )
    player = player.join(players, on="player_id", how="left")
    # Prefer roster position for the season/week when available
    try:
        rosters = nfl.load_rosters(seasons=season)
        rcols = ["gsis_id", "position", "full_name"]
        if "week" in rosters.columns and week is not None:
            rosters = rosters.filter(pl.col("week") == week)
        if teams:
            rosters = rosters.filter(pl.col("team").is_in(teams))
        rost = rosters.select(
            [
                pl.col("gsis_id").alias("player_id"),
                pl.col("position").alias("roster_position"),
                pl.col("full_name").alias("roster_name"),
                pl.col("team").alias("roster_team"),
            ]
        ).unique(subset=["player_id"], keep="first")
        player = player.join(rost, on="player_id", how="left")
        player = player.with_columns(
            pl.coalesce(["roster_position", "position"]).alias("position"),
            pl.coalesce(["player", "roster_name", "receiver_player_name"]).alias(
                "player"
            ),
        )
    except Exception:
        player = player.with_columns(
            pl.coalesce(["player", "receiver_player_name"]).alias("player")
        )

    wrs = player.filter(pl.col("position").is_in(_wr_positions()))

    # Add zero-target WRs who appear in weekly player_stats for this game/week
    try:
        ps = nfl.load_player_stats(seasons=season)
        ps = ps.filter(pl.col("position") == "WR")
        if week is not None:
            ps = ps.filter(pl.col("week") == week)
        if teams:
            ps = ps.filter(pl.col("team").is_in(teams))
        if game_id:
            # further limit to opponents in this game when possible
            _, _, away, home = parse_game_id(game_id)
            ps = ps.filter(pl.col("team").is_in([away, home]))
            if "opponent_team" in ps.columns:
                ps = ps.filter(
                    (
                        (pl.col("team") == away) & (pl.col("opponent_team") == home)
                    )
                    | (
                        (pl.col("team") == home) & (pl.col("opponent_team") == away)
                    )
                )
        id_col = "player_id" if "player_id" in ps.columns else None
        if id_col is None:
            for c in ("gsis_id", "player_gsis_id"):
                if c in ps.columns:
                    id_col = c
                    break
        name_col = (
            "player_display_name"
            if "player_display_name" in ps.columns
            else "player_name"
        )
        if id_col:
            have = set(wrs["player_id"].to_list())
            extras = ps.filter(~pl.col(id_col).is_in(list(have)))
            if extras.height:
                # attach team denominators for share calc (0/0 -> 0)
                extras = extras.select(
                    [
                        pl.col(id_col).alias("player_id"),
                        pl.col(name_col).alias("player"),
                        pl.col("team"),
                        pl.lit(0).cast(pl.UInt32).alias("targets"),
                        pl.lit(0.0).alias("air_yards"),
                        pl.lit("WR").alias("position"),
                    ]
                ).join(team_tot, on="team", how="left")
                extras = extras.with_columns(
                    pl.lit(0.0).alias("target_share"),
                    pl.lit(0.0).alias("air_yards_share"),
                    pl.lit(0.0).alias("wopr"),
                    pl.lit(None).cast(pl.Utf8).alias("receiver_player_name"),
                )
                # align columns
                for c in wrs.columns:
                    if c not in extras.columns:
                        extras = extras.with_columns(
                            pl.lit(None).alias(c).cast(wrs.schema[c])
                        )
                extras = extras.select(wrs.columns)
                wrs = pl.concat([wrs, extras], how="vertical_relaxed")
    except Exception as e:
        print(f"note: could not add zero-target WRs from player_stats: {e}", file=sys.stderr)

    wrs = wrs.sort("wopr", descending=True)

    # Infer week/season from pbp if needed
    weeks = pbp.select(pl.col("week").unique()).to_series().to_list()
    seasons = pbp.select(pl.col("season").unique()).to_series().to_list()
    game_ids = pbp.select(pl.col("game_id").unique()).to_series().to_list()
    meta = {
        "season": int(seasons[0]) if len(seasons) == 1 else season,
        "week": int(weeks[0]) if len(weeks) == 1 else week,
        "game_id": game_id or (game_ids[0] if len(game_ids) == 1 else None),
        "teams": teams
        or sorted(wrs["team"].unique().to_list()),
        "n_plays": int(pbp.height),
        "n_pass_attempts": int(passes.height),
        "n_targets": int(targeted.height),
    }
    return wrs, meta


def plot_wopr(
    df: pl.DataFrame,
    logos: dict[str, Path],
    meta: dict,
    out_path: Path,
) -> None:
    rows = df.to_dicts()
    if not rows:
        raise RuntimeError("No WR rows to plot")

    names = [r["player"] for r in rows]
    values = [float(r["wopr"] or 0.0) for r in rows]
    teams = [r["team"] for r in rows]
    n = len(rows)

    # Figure size scales with roster depth
    fig_h = max(4.5, 0.55 * n + 1.8)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y = np.arange(n)[::-1]  # highest WOPR at top
    ax.barh(y, values, color=BAR_COLOR, height=0.62, zorder=3, edgecolor="none")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=11, fontweight="bold", color="#111")
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", labelsize=10, colors="#333")

    xmax = max(1.05, max(values) * 1.12 if values else 1.05)
    # snap xmax up to next 0.25
    xmax = float(np.ceil(xmax * 4) / 4)
    ax.set_xlim(0, xmax)
    ax.set_xticks(np.arange(0, xmax + 0.001, 0.25))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="x", color="#D0D0D0", linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)

    ax.set_xlabel(
        "Weighted Opportunity Rating: Wide Receivers",
        fontsize=11,
        color="#222",
        labelpad=10,
    )

    season = meta.get("season")
    # Main title: no week/window wording
    title = "Leaders in Weighted Opportunity Rating (WOPR)"
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98, color="#111")
    ax.set_title(
        f"{season} Season | WOPR = 1.5 x Target Share + 0.7 x Air Yards Share",
        fontsize=10,
        fontweight="regular",
        pad=10,
        color="#444",
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#222")
        spine.set_linewidth(0.8)

    # Team logos at end of each bar
    for yi, val, team in zip(y, values, teams):
        path = logos.get(team)
        if not path:
            continue
        try:
            img = Image.open(path).convert("RGBA")
            # Keep enough pixels for dpi=160 (28px+zoom1 stretched ~62px → blurry)
            img.thumbnail((80, 80), Image.Resampling.LANCZOS)
            im = OffsetImage(np.asarray(img), zoom=18.0 / img.size[1])
            # place just past the bar tip; for near-zero, still near 0
            x = max(val, 0.0) + xmax * 0.018
            ab = AnnotationBbox(
                im,
                (x, yi),
                frameon=False,
                box_alignment=(0.0, 0.5),
                zorder=4,
            )
            ax.add_artist(ab)
        except Exception:
            ax.text(val + xmax * 0.01, yi, team, va="center", fontsize=8, color="#333")

    fig.text(
        0.99,
        0.01,
        "Chart: Numbers | Data: nflverse / nflfastR",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#666",
        transform=fig.transFigure,
    )

    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor="white")
    plt.close(fig)


def print_table(df: pl.DataFrame) -> None:
    cols = [
        "player",
        "team",
        "targets",
        "air_yards",
        "target_share",
        "air_yards_share",
        "wopr",
    ]
    show = df.select([c for c in cols if c in df.columns])
    # pretty print
    rows = show.to_dicts()
    headers = ["player", "team", "targets", "air_yards", "target_share", "air_yards_share", "wopr"]
    widths = {h: max(len(h), max((len(f"{r.get(h)}") if r.get(h) is not None else 0) for r in rows) if rows else 0) for h in headers}
    # widen numeric formatting
    def fmt(h, v):
        if v is None:
            return ""
        if h in ("target_share", "air_yards_share", "wopr"):
            return f"{float(v):.6f}"
        if h == "air_yards":
            return f"{float(v):.1f}"
        return str(v)

    widths = {
        h: max(len(h), max(len(fmt(h, r.get(h))) for r in rows) if rows else 0)
        for h in headers
    }
    line = "  ".join(h.ljust(widths[h]) for h in headers)
    print(line)
    print("  ".join("-" * widths[h] for h in headers))
    for r in rows:
        print("  ".join(fmt(h, r.get(h)).ljust(widths[h]) for h in headers))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WOPR WR chart for an NFL game (nflverse)")
    p.add_argument("--game-id", type=str, default=None, help="e.g. 2026_01_NE_SEA")
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--week", type=int, default=None)
    p.add_argument(
        "--teams",
        type=str,
        default=None,
        help="Comma-separated team abbrs, e.g. NE,SEA",
    )
    p.add_argument("--out", type=Path, default=None, help="Output PNG path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    teams = (
        [t.strip().upper() for t in args.teams.split(",") if t.strip()]
        if args.teams
        else None
    )
    season, week, teams, game_id = resolve_selection(
        args.game_id, args.season, args.week, teams
    )
    # If only game-id given, season/week already filled
    if args.season is not None and game_id is None:
        season = args.season
    if args.week is not None and week is None:
        week = args.week

    df, meta = compute_wopr(
        season=season, week=week, teams=teams, game_id=game_id
    )

    # Default output next to script
    if args.out:
        out_path = args.out
    else:
        gid = meta.get("game_id") or (
            f"{season}_{week:02d}_{'_'.join(meta.get('teams') or [])}"
            if week is not None
            else f"{season}_wopr"
        )
        out_path = HERE / f"wopr_{gid}.png"

    team_meta = load_team_meta()
    logos = download_logos(team_meta, abbrs=meta.get("teams"))
    plot_wopr(df, logos, meta, out_path)

    print_table(df)
    print()
    print("wrote", out_path)
    print("meta", meta)


if __name__ == "__main__":
    main()
