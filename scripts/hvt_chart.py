#!/usr/bin/env python
"""Running back High Value Touches (HVT) scatter chart from nflverse PBP.

HVT           = carries inside the opponent's 10 (yardline_100 <= 10) + receptions
Opportunities = rush attempts (carries) + targets
X             = HVTs / games played
Y             = HVT / opportunities (as %)

Filters: minimum 10 opportunities per game.
RB only (position == RB; FB excluded).

Carries/targets/receptions aggregated from load_pbp for consistency with
inside-10 counts (player_stats.rushing_10 is 10+ yard rushes, NOT inside-10).
Kneels (play_type == qb_kneel) and two-point attempts excluded so totals
match nflverse player_stats carries/targets/receptions.

Usage:
  python hvt_chart.py --season 2025 --weeks 8-11
  python hvt_chart.py --season 2026 --weeks 1-1
  python hvt_chart.py --season 2026 --through-week 4
  python hvt_chart.py --season 2025 --weeks 8-11 --out /path.png
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

try:
    from adjustText import adjust_text

    HAS_ADJUST = True
except ImportError:
    HAS_ADJUST = False


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


def parse_weeks(weeks: str | None, through_week: int | None) -> tuple[list[int], str]:
    """Return week list and a window token for filenames/meta (not used in chart title)."""
    if through_week is not None and weeks is not None:
        raise ValueError("Pass either --weeks or --through-week, not both")
    if through_week is not None:
        if through_week < 1:
            raise ValueError("--through-week must be >= 1")
        wlist = list(range(1, through_week + 1))
        if through_week == 1:
            return wlist, "Week 1"
        return wlist, f"through Week {through_week}"
    if weeks is None:
        raise ValueError("Provide --weeks START-END (or N-N) or --through-week N")
    parts = weeks.replace(" ", "").split("-")
    if len(parts) != 2:
        raise ValueError(f"--weeks must look like 8-11 or 1-1, got {weeks!r}")
    start, end = int(parts[0]), int(parts[1])
    if start < 1 or end < start:
        raise ValueError(f"invalid week range {weeks!r}")
    wlist = list(range(start, end + 1))
    if start == end:
        return wlist, f"Week {start}"
    return wlist, f"Weeks {start}-{end}"


def compute_hvt(
    season: int,
    weeks: list[int],
    min_opp_per_game: float = 10.0,
    min_opp_total: float = 0.0,
) -> tuple[pl.DataFrame, dict]:
    pbp = nfl.load_pbp(seasons=season)
    if pbp.height == 0:
        raise RuntimeError(f"No PBP rows for season {season}")

    available = sorted(pbp["week"].unique().to_list())
    pbp = pbp.filter(pl.col("week").is_in(weeks))
    if pbp.height == 0:
        raise RuntimeError(
            f"No PBP for season {season} weeks {weeks}; available weeks: {available}"
        )

    # Carries: rush_attempt with rusher; exclude kneels + 2PT (matches player_stats)
    rushes = pbp.filter(
        (pl.col("rush_attempt") == 1)
        & pl.col("rusher_player_id").is_not_null()
        & (pl.col("play_type") == "run")
        & (pl.col("two_point_attempt") != 1)
    )
    inside10 = rushes.filter(pl.col("yardline_100") <= 10)

    # Targets / receptions (exclude 2PT)
    targets = pbp.filter(
        (pl.col("pass_attempt") == 1)
        & pl.col("receiver_player_id").is_not_null()
        & (pl.col("two_point_attempt") != 1)
    )
    receptions = pbp.filter(
        (pl.col("complete_pass") == 1)
        & pl.col("receiver_player_id").is_not_null()
        & (pl.col("two_point_attempt") != 1)
    )

    carries_df = (
        rushes.group_by("rusher_player_id")
        .agg(
            pl.len().alias("carries"),
            pl.col("rusher_player_name").first().alias("rush_name"),
            pl.col("posteam").last().alias("rush_team"),
        )
        .rename({"rusher_player_id": "player_id"})
    )
    i10_df = (
        inside10.group_by("rusher_player_id")
        .agg(pl.len().alias("inside10_carries"))
        .rename({"rusher_player_id": "player_id"})
    )
    tgt_df = (
        targets.group_by("receiver_player_id")
        .agg(
            pl.len().alias("targets"),
            pl.col("receiver_player_name").first().alias("rec_name"),
            pl.col("posteam").last().alias("rec_team"),
        )
        .rename({"receiver_player_id": "player_id"})
    )
    rec_df = (
        receptions.group_by("receiver_player_id")
        .agg(pl.len().alias("receptions"))
        .rename({"receiver_player_id": "player_id"})
    )

    # Games played = distinct games with a carry or target
    rush_g = rushes.select(
        pl.col("rusher_player_id").alias("player_id"), "game_id"
    )
    tgt_g = targets.select(
        pl.col("receiver_player_id").alias("player_id"), "game_id"
    )
    games_df = (
        pl.concat([rush_g, tgt_g])
        .unique()
        .group_by("player_id")
        .agg(pl.len().alias("games"))
    )

    df = carries_df.join(i10_df, on="player_id", how="full", coalesce=True)
    df = df.join(tgt_df, on="player_id", how="full", coalesce=True)
    df = df.join(rec_df, on="player_id", how="full", coalesce=True)
    df = df.join(games_df, on="player_id", how="left")
    for c in ("carries", "targets", "receptions", "inside10_carries", "games"):
        df = df.with_columns(pl.col(c).fill_null(0))

    df = df.with_columns(
        [
            (pl.col("carries") + pl.col("targets")).alias("opportunities"),
            (pl.col("inside10_carries") + pl.col("receptions")).alias("hvt"),
            pl.coalesce(["rush_name", "rec_name"]).alias("name_short"),
            pl.coalesce(["rush_team", "rec_team"]).alias("team_pbp"),
        ]
    )

    # RB filter + display names via weekly player_stats (authoritative position)
    ps = nfl.load_player_stats(seasons=season)
    ps = ps.filter(
        (pl.col("position") == "RB") & pl.col("week").is_in(weeks)
    )
    if ps.height == 0:
        # Fall back to season rosters if player_stats empty for window
        rost = nfl.load_rosters(seasons=season).filter(pl.col("position") == "RB")
        id_map = rost.select(
            [
                pl.col("gsis_id").alias("player_id"),
                pl.col("full_name").alias("player"),
                pl.col("team").alias("team"),
                pl.lit("RB").alias("position"),
            ]
        ).unique(subset=["player_id"], keep="last")
    else:
        id_map = (
            ps.select(
                [
                    "player_id",
                    pl.col("player_display_name").alias("player"),
                    "team",
                    "position",
                ]
            )
            .unique(subset=["player_id"], keep="last")
        )

    df = df.join(id_map, on="player_id", how="inner")
    df = df.with_columns(
        [
            pl.coalesce(["player", "name_short"]).alias("player"),
            pl.coalesce(["team", "team_pbp"]).alias("team"),
        ]
    )

    df = df.with_columns(
        [
            (pl.col("hvt") / pl.col("games")).alias("hvt_per_game"),
            (100.0 * pl.col("hvt") / pl.col("opportunities")).alias("hvt_pct"),
            (pl.col("opportunities") / pl.col("games")).alias("opp_per_game"),
        ]
    )

    # Filters
    filtered = df.filter(
        (pl.col("opp_per_game") >= min_opp_per_game)
        & (pl.col("opportunities") >= min_opp_total)
    ).sort("hvt_per_game", descending=True)

    meta = {
        "season": season,
        "weeks": weeks,
        "available_weeks": available,
        "n_pbp": int(pbp.height),
        "n_rb_raw": int(df.height),
        "n_pass_filter": int(filtered.height),
        "min_opp_per_game": min_opp_per_game,
        "min_opp_total": min_opp_total,
    }
    return filtered, meta


def _short_name(full: str) -> str:
    """First initial + last name when possible."""
    if not full:
        return ""
    parts = str(full).strip().split()
    if len(parts) == 1:
        return parts[0]
    # Keep suffix with last token (Jr., II, etc.)
    return f"{parts[0][0]}. {' '.join(parts[1:])}"


def plot_hvt(
    df: pl.DataFrame,
    logos: dict[str, Path],
    meta: dict,
    window_label: str,
    out_path: Path,
) -> None:
    rows = df.to_dicts()
    season = meta["season"]

    fig, ax = plt.subplots(figsize=(12.5, 9.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    xs = [float(r["hvt_per_game"] or 0.0) for r in rows]
    ys = [float(r["hvt_pct"] or 0.0) for r in rows]
    teams = [r["team"] for r in rows]
    names = [_short_name(r["player"]) for r in rows]

    # Axis limits
    xmax = max(xs) if xs else 1.0
    ymax = max(ys) if ys else 10.0
    x_hi = max(3.0, float(np.ceil(xmax + 0.5)))
    y_hi = max(50.0, float(np.ceil((ymax + 5) / 5.0) * 5.0))
    ax.set_xlim(-0.15, x_hi)
    ax.set_ylim(0, y_hi)

    ax.set_xticks(range(0, int(x_hi) + 1))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))

    ax.grid(True, color="#D8D8D8", linewidth=0.85, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#444")
        spine.set_linewidth(0.9)

    ax.set_xlabel("HVTs/Game", fontsize=12, color="#222", labelpad=8)
    ax.set_ylabel(
        "HVT as a % of total opportunities",
        fontsize=12,
        color="#222",
        labelpad=8,
    )
    ax.tick_params(colors="#333", labelsize=10)

    # Title / subtitle (top-left style via fig text)
    title = "Running back high value touches"
    min_opp = meta.get("min_opp_per_game", 10.0)
    min_tot = meta.get("min_opp_total", 0.0)
    filt_bits = [f"Minimum {min_opp:g} opportunities per game"]
    if min_tot and min_tot > 0:
        filt_bits.append(f"{min_tot:g} opportunities total")
    subtitle = (
        ", ".join(filt_bits)
        + " | High value touches (HVT) include carries inside the ten and receptions"
    )
    fig.text(
        0.06,
        0.965,
        title,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color="#111",
    )
    fig.text(
        0.06,
        0.925,
        subtitle,
        ha="left",
        va="top",
        fontsize=9,
        color="#555",
    )

    # Place logos
    logo_artists = []
    for x, y, team in zip(xs, ys, teams):
        path = logos.get(team)
        if not path:
            ax.plot(x, y, "o", color="#888", markersize=6, zorder=3)
            continue
        try:
            img = Image.open(path).convert("RGBA")
            # Keep enough pixels for dpi=160 (do NOT thumbnail to 28px)
            img.thumbnail((80, 80), Image.Resampling.LANCZOS)
            zoom = 17.0 / img.size[1]
            im = OffsetImage(np.asarray(img), zoom=zoom)
            ab = AnnotationBbox(
                im,
                (x, y),
                frameon=False,
                box_alignment=(0.5, 0.5),
                zorder=4,
            )
            ax.add_artist(ab)
            logo_artists.append(ab)
        except Exception:
            ax.plot(x, y, "o", color="#888", markersize=6, zorder=3)

    # Player name labels
    texts = []
    for x, y, name in zip(xs, ys, names):
        # default offset above-right of logo
        t = ax.annotate(
            name,
            xy=(x, y),
            xytext=(x + 0.12, y + 1.6),
            fontsize=8,
            color="#222",
            ha="left",
            va="bottom",
            zorder=5,
        )
        texts.append(t)

    if texts and HAS_ADJUST:
        adjust_text(
            texts,
            x=xs,
            y=ys,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="#999", lw=0.5),
            force_text=(0.4, 0.6),
            force_points=(0.3, 0.4),
            expand_text=(1.15, 1.25),
            expand_points=(1.2, 1.3),
        )
    elif texts:
        # Simple collision nudge + thin leader lines
        used: list[tuple[float, float]] = []
        trans = ax.transData
        for i, (x, y, t) in enumerate(zip(xs, ys, texts)):
            tx, ty = x + 0.14, y + 1.8
            # stagger based on rank to reduce overlap
            ty += (i % 5) * 0.35 - 0.7
            tx += ((i % 3) - 1) * 0.08
            # push away from neighbors in display space
            for ux, uy in used:
                if abs(tx - ux) < 0.55 and abs(ty - uy) < 2.8:
                    ty = uy + 2.6 if ty >= uy else uy - 2.6
            used.append((tx, ty))
            t.set_position((tx, ty))
            ax.annotate(
                "",
                xy=(x, y),
                xytext=(tx, ty),
                arrowprops=dict(
                    arrowstyle="-",
                    color="#B0B0B0",
                    lw=0.55,
                    shrinkA=2,
                    shrinkB=8,
                ),
                zorder=2,
            )

    fig.text(
        0.99,
        0.012,
        "Chart: Numbers | Data: nflverse / nflfastR",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#666",
        transform=fig.transFigure,
    )
    # small season tag bottom-left
    fig.text(
        0.06,
        0.012,
        f"{season} season",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#888",
        transform=fig.transFigure,
    )

    fig.tight_layout(rect=[0.04, 0.04, 0.98, 0.90])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor="white")
    plt.close(fig)


def print_table(df: pl.DataFrame) -> None:
    cols = [
        "player",
        "team",
        "games",
        "carries",
        "targets",
        "opportunities",
        "inside10_carries",
        "receptions",
        "hvt",
        "hvt_per_game",
        "hvt_pct",
    ]
    show = df.select([c for c in cols if c in df.columns]).sort(
        "hvt_per_game", descending=True
    )
    rows = show.to_dicts()

    def fmt(h: str, v) -> str:
        if v is None:
            return ""
        if h == "hvt_per_game":
            return f"{float(v):.3f}"
        if h == "hvt_pct":
            return f"{float(v):.1f}"
        return str(v)

    widths = {
        h: max(len(h), max((len(fmt(h, r.get(h))) for r in rows), default=0))
        for h in cols
    }
    print("  ".join(h.ljust(widths[h]) for h in cols))
    print("  ".join("-" * widths[h] for h in cols))
    for r in rows:
        print("  ".join(fmt(h, r.get(h)).ljust(widths[h]) for h in cols))


def default_out_name(season: int, weeks: list[int], through: bool) -> str:
    if through:
        return f"hvt_{season}_through_w{weeks[-1]:02d}.png"
    return f"hvt_{season}_w{weeks[0]:02d}-{weeks[-1]:02d}.png"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RB High Value Touches scatter (nflverse PBP)"
    )
    p.add_argument("--season", type=int, required=True)
    p.add_argument(
        "--weeks",
        type=str,
        default=None,
        help="Week range START-END, e.g. 8-11 or 1-1",
    )
    p.add_argument(
        "--through-week",
        type=int,
        default=None,
        help="Include weeks 1..N (inclusive)",
    )
    p.add_argument("--out", type=Path, default=None, help="Output PNG path")
    p.add_argument("--min-opp-per-game", type=float, default=10.0)
    p.add_argument("--min-opp-total", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    weeks, window_label = parse_weeks(args.weeks, args.through_week)
    through = args.through_week is not None

    df, meta = compute_hvt(
        args.season, weeks,
        min_opp_per_game=args.min_opp_per_game,
        min_opp_total=args.min_opp_total,
    )
    meta["window_label"] = window_label

    if args.out:
        out_path = args.out
    else:
        out_path = HERE / default_out_name(args.season, weeks, through)

    team_meta = load_team_meta()
    abbrs = sorted({r["team"] for r in df.to_dicts() if r.get("team")})
    logos = download_logos(team_meta, abbrs=abbrs if abbrs else None)

    plot_hvt(df, logos, meta, window_label, out_path)

    print_table(df)
    print()
    print(f"n_pass_filter={meta['n_pass_filter']} n_rb_raw={meta['n_rb_raw']}")
    print("wrote", out_path)
    print("meta", meta)


if __name__ == "__main__":
    main()
