#!/usr/bin/env python
"""Pass Rate Over Expectation heatmap table (all 32 NFL teams).

Matches Josh's reference layout:
  TEAM | OVERALL PASS RATE (%) | PROE (%) OVERALL | NEUTRAL | LAST 4 | RED ZONE

Definitions (nflverse load_pbp, REG only):
  Universe: xpass not null, pass not null, posteam not null
  Overall pass rate = mean(pass) * 100
  PROE = mean(pass_oe)  — pass_oe is already (pass - xpass) * 100 (percentage points)
  NEUTRAL: abs(score_differential) <= 7
  LAST 4: last 4 REG games by (week, game_id) per team
  RED ZONE: yardline_100 <= 20

Usage:
  python proe_table.py --season 2025
  python proe_table.py --season 2025 --through-week 14
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import nflreadpy as nfl
import numpy as np
import polars as pl
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

HERE = Path(__file__).resolve().parent
LOGO_DIR = HERE / "logos"

# Division order (for TEAM_ORDER completeness); table sorts alphabetically by full name
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

NEUTRAL_DEF = "abs(score_differential) <= 7"
PASS_OE_NOTE = (
    "pass_oe is already (pass - xpass) * 100 (percentage points); "
    "PROE reported as mean(pass_oe) with no extra *100"
)


def load_team_meta() -> pl.DataFrame:
    teams = nfl.load_teams().filter(pl.col("team_abbr").is_in(TEAM_ORDER))
    return teams.select(["team_abbr", "team_name", "team_nick"]).unique(
        subset=["team_abbr"], keep="last"
    )


def ensure_logos() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for abbr in TEAM_ORDER:
        dest = LOGO_DIR / f"{abbr}.png"
        if dest.exists() and dest.stat().st_size > 0:
            paths[abbr] = dest
            continue
        # LAR alias
        alt = LOGO_DIR / "LAR.png"
        if abbr == "LA" and alt.exists():
            paths[abbr] = alt
    return paths


def _week_filter(pbp: pl.DataFrame, through_week: int | None) -> pl.DataFrame:
    if through_week is None:
        return pbp
    return pbp.filter(pl.col("week") <= through_week)


def compute(season: int, through_week: int | None) -> tuple[pl.DataFrame, dict]:
    pbp = nfl.load_pbp(seasons=season)
    pbp = pbp.filter(pl.col("season_type") == "REG")
    pbp = _week_filter(pbp, through_week)

    plays = pbp.filter(
        pl.col("xpass").is_not_null()
        & pl.col("pass").is_not_null()
        & pl.col("posteam").is_not_null()
    )

    # Per-team last-4 game_ids (chronological by week, then game_id)
    game_keys = (
        plays.select(["posteam", "game_id", "week"])
        .unique()
        .sort(["posteam", "week", "game_id"])
    )
    last4_rows = []
    for team in TEAM_ORDER:
        tg = game_keys.filter(pl.col("posteam") == team)
        ids = tg.tail(4)["game_id"].to_list()
        for gid in ids:
            last4_rows.append({"posteam": team, "game_id": gid})
    last4_df = pl.DataFrame(last4_rows) if last4_rows else pl.DataFrame(
        schema={"posteam": pl.Utf8, "game_id": pl.Utf8}
    )
    plays_l4 = plays.join(last4_df, on=["posteam", "game_id"], how="inner")

    overall = plays.group_by("posteam").agg(
        (100.0 * pl.col("pass").mean()).alias("pass_rate"),
        pl.col("pass_oe").mean().alias("proe_overall"),
        pl.len().alias("plays"),
        pl.col("game_id").n_unique().alias("games"),
    )

    neutral = (
        plays.filter(pl.col("score_differential").abs() <= 7)
        .group_by("posteam")
        .agg(pl.col("pass_oe").mean().alias("proe_neutral"))
    )

    redzone = (
        plays.filter(pl.col("yardline_100") <= 20)
        .group_by("posteam")
        .agg(pl.col("pass_oe").mean().alias("proe_redzone"))
    )

    last4 = plays_l4.group_by("posteam").agg(
        pl.col("pass_oe").mean().alias("proe_last4"),
        pl.col("game_id").n_unique().alias("last4_games"),
    )

    meta_teams = load_team_meta()
    df = (
        overall.join(neutral, on="posteam", how="left")
        .join(redzone, on="posteam", how="left")
        .join(last4, on="posteam", how="left")
        .join(meta_teams, left_on="posteam", right_on="team_abbr", how="left")
        .rename({"posteam": "team_abbr"})
    )

    # Ensure all 32 teams present
    base = pl.DataFrame({"team_abbr": TEAM_ORDER}).join(
        meta_teams, on="team_abbr", how="left"
    )
    df = base.join(df.drop(["team_name", "team_nick"], strict=False), on="team_abbr", how="left")
    # re-join names if dropped
    if "team_name" not in df.columns:
        df = df.join(meta_teams, on="team_abbr", how="left")

    df = df.sort("team_name")

    max_week = int(plays["week"].max()) if plays.height else 0
    missing = df.filter(pl.col("plays").is_null() | (pl.col("plays") == 0))[
        "team_abbr"
    ].to_list()

    meta = {
        "season": season,
        "season_type": "REG",
        "through_week": through_week if through_week is not None else max_week,
        "max_week_in_data": max_week,
        "plays": int(plays.height),
        "neutral_definition": NEUTRAL_DEF,
        "pass_oe_scale": PASS_OE_NOTE,
        "universe": "xpass not null & pass not null & posteam not null",
        "missing_teams": missing,
        "source": f"nflverse load_pbp(seasons={season}) REG",
    }
    return df, meta


def _diverging_rgba(val: float | None, vmax: float = 12.0) -> tuple[float, float, float, float]:
    """Green positive / purple negative / white near zero. Soften near zero."""
    if val is None or (isinstance(val, float) and val != val):
        return (1.0, 1.0, 1.0, 1.0)
    # Normalize to [-1, 1]
    t = float(np.clip(val / vmax, -1.0, 1.0))
    # Soft dead-zone near 0
    if abs(t) < 0.04:
        return (1.0, 1.0, 1.0, 1.0)
    if t > 0:
        # white -> #1B7A3D green
        strength = t
        r = 1.0 - strength * (1.0 - 0.106)
        g = 1.0 - strength * (1.0 - 0.478)
        b = 1.0 - strength * (1.0 - 0.239)
    else:
        # white -> #6B2D8B purple
        strength = -t
        r = 1.0 - strength * (1.0 - 0.420)
        g = 1.0 - strength * (1.0 - 0.176)
        b = 1.0 - strength * (1.0 - 0.545)
    return (r, g, b, 1.0)


def _fmt(val: float | None, digits: int = 1, signed: bool = False) -> str:
    if val is None or (isinstance(val, float) and val != val):
        return "—"
    if signed:
        return f"{val:+.{digits}f}"
    return f"{val:.{digits}f}"


def write_csv(df: pl.DataFrame, path: Path) -> None:
    out = df.select(
        pl.col("team_name").alias("TEAM"),
        pl.col("team_abbr").alias("ABBR"),
        pl.col("pass_rate").round(1).alias("OVERALL_PASS_RATE_PCT"),
        pl.col("proe_overall").round(1).alias("PROE_OVERALL"),
        pl.col("proe_neutral").round(1).alias("PROE_NEUTRAL"),
        pl.col("proe_last4").round(1).alias("PROE_LAST4"),
        pl.col("proe_redzone").round(1).alias("PROE_RED_ZONE"),
        pl.col("plays"),
        pl.col("games"),
        pl.col("last4_games"),
    )
    out.write_csv(path)


def write_markdown(df: pl.DataFrame, path: Path, meta: dict) -> None:
    rows = df.to_dicts()
    lines = [
        f"# Pass Rate Over Expectation — {meta['season']} REG",
        "",
        f"Source: `{meta['source']}`. Universe: {meta['universe']}.",
        f"Neutral: **{meta['neutral_definition']}**.",
        f"pass_oe scale: {meta['pass_oe_scale']}.",
        f"LAST 4: last 4 REG games by (week, game_id) per team. RED ZONE: yardline_100 ≤ 20.",
        f"Plays in universe: {meta['plays']}. Sorted alphabetically by full team name.",
        "",
        "| TEAM | OVERALL PASS RATE (%) | PROE (%) OVERALL | PROE (%) NEUTRAL | PROE (%) LAST 4 | PROE (%) RED ZONE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {team} | {pr} | {o} | {n} | {l4} | {rz} |".format(
                team=r.get("team_name") or r.get("team_abbr"),
                pr=_fmt(r.get("pass_rate")),
                o=_fmt(r.get("proe_overall"), signed=True),
                n=_fmt(r.get("proe_neutral"), signed=True),
                l4=_fmt(r.get("proe_last4"), signed=True),
                rz=_fmt(r.get("proe_redzone"), signed=True),
            )
        )
    if meta.get("missing_teams"):
        lines += ["", f"**Missing data:** {', '.join(meta['missing_teams'])}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_table(
    df: pl.DataFrame,
    logos: dict[str, Path],
    meta: dict,
    out: Path,
) -> None:
    rows = df.to_dicts()
    n = len(rows)
    # Column layout (figure fraction / inches)
    # TEAM wide, then 5 numeric cols
    col_labels_top = ["", "", "PROE (%)", "", "", ""]
    col_labels = [
        "TEAM",
        "OVERALL\nPASS RATE (%)",
        "OVERALL",
        "NEUTRAL",
        "LAST 4",
        "RED ZONE",
    ]
    # Relative widths
    widths = [3.6, 1.35, 1.15, 1.15, 1.15, 1.15]
    total_w = sum(widths)
    fig_w = 11.5
    row_h = 0.38
    header_h = 0.85
    # Extra top pad so title + subtitle never collide with each other or the header
    title_pad = 1.35
    fig_h = header_h + n * row_h + 0.55 + title_pad

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, n + 3.4)
    ax.axis("off")

    # Title block (clear vertical gap: title → subtitle → header)
    ax.text(
        0.05,
        n + 3.15,
        "Pass Rate Over Expectation",
        fontsize=18,
        fontweight="bold",
        va="top",
        ha="left",
        color="#111",
        transform=ax.transData,
        clip_on=False,
    )
    sub = f"{meta['season']} Regular Season"
    ax.text(
        0.05,
        n + 2.55,
        sub,
        fontsize=10,
        color="#555",
        va="top",
        ha="left",
        clip_on=False,
    )

    # Column x positions
    xs = []
    x = 0.0
    for w in widths:
        xs.append(x)
        x += w

    # Header background bar
    header_y = n + 0.15
    ax.fill_between(
        [0, total_w], header_y, header_y + 0.95, color="#F3F3F3", zorder=0
    )

    # Group header PROE (%)
    proe_x0 = xs[2]
    proe_x1 = total_w
    ax.text(
        (proe_x0 + proe_x1) / 2,
        header_y + 0.72,
        "PROE (%)",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#333",
    )
    ax.plot(
        [proe_x0 + 0.08, proe_x1 - 0.08],
        [header_y + 0.48, header_y + 0.48],
        color="#BBBBBB",
        lw=0.8,
    )

    for i, lab in enumerate(col_labels):
        cx = xs[i] + widths[i] / 2
        if i == 0:
            ax.text(
                xs[i] + 0.12,
                header_y + 0.22,
                lab,
                ha="left",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="#444",
            )
        else:
            ax.text(
                cx,
                header_y + 0.22,
                lab,
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="bold",
                color="#444",
            )

    # Data rows (top to bottom = alphabetical already)
    proe_cols = ["proe_overall", "proe_neutral", "proe_last4", "proe_redzone"]
    # Shared color scale from data
    all_proe = []
    for r in rows:
        for c in proe_cols:
            v = r.get(c)
            if v is not None and v == v:
                all_proe.append(abs(float(v)))
    vmax = float(np.percentile(all_proe, 90)) if all_proe else 12.0
    vmax = max(vmax, 8.0)

    for ri, r in enumerate(rows):
        y = n - ri - 1  # top row = first team
        # Alternating subtle row tint under TEAM only
        if ri % 2 == 1:
            ax.fill_between([0, total_w], y, y + 1, color="#FAFAFA", zorder=0)

        # PROE cell heatmaps
        for ci, col in enumerate(proe_cols):
            xi = 2 + ci
            val = r.get(col)
            color = _diverging_rgba(val, vmax=vmax)
            ax.fill_between(
                [xs[xi] + 0.04, xs[xi] + widths[xi] - 0.04],
                y + 0.08,
                y + 0.92,
                color=color,
                zorder=1,
            )
            ax.text(
                xs[xi] + widths[xi] / 2,
                y + 0.5,
                _fmt(val, signed=True),
                ha="center",
                va="center",
                fontsize=9,
                color="#111",
                zorder=3,
                fontweight="normal",
            )

        # Pass rate (no heatmap)
        ax.text(
            xs[1] + widths[1] / 2,
            y + 0.5,
            _fmt(r.get("pass_rate")),
            ha="center",
            va="center",
            fontsize=9,
            color="#111",
            zorder=3,
        )

        # Team name + logo
        name = r.get("team_name") or r.get("team_abbr") or ""
        abbr = r.get("team_abbr")
        ax.text(
            xs[0] + 0.12,
            y + 0.5,
            name,
            ha="left",
            va="center",
            fontsize=9,
            color="#111",
            zorder=3,
            fontweight="normal",
        )
        logo_path = logos.get(abbr) if abbr else None
        if logo_path and logo_path.exists():
            try:
                img = Image.open(logo_path).convert("RGBA")
                img.thumbnail((80, 80), Image.Resampling.LANCZOS)
                # Wash out for background-style mark on the right of the name cell
                arr = np.asarray(img).astype(float)
                arr[..., 3] = arr[..., 3] * 0.55  # lightly washed watermark
                im = OffsetImage(arr.astype(np.uint8), zoom=26.0 / max(img.size[1], 1))
                ab = AnnotationBbox(
                    im,
                    (xs[0] + widths[0] - 0.55, y + 0.5),
                    frameon=False,
                    pad=0,
                    zorder=2,
                )
                ax.add_artist(ab)
            except Exception:
                pass

        # Row separator
        ax.plot([0, total_w], [y, y], color="#DDDDDD", lw=0.6, zorder=4)

    # Top/bottom borders
    ax.plot([0, total_w], [n, n], color="#AAAAAA", lw=1.0, zorder=5)
    ax.plot([0, total_w], [0, 0], color="#AAAAAA", lw=1.0, zorder=5)

    footer = (
        f"Neutral = {meta['neutral_definition']}  ·  "
        f"LAST 4 = last 4 REG games  ·  RED ZONE = yardline_100 ≤ 20  ·  "
        f"Data: {meta['source']}  ·  {PASS_OE_NOTE}"
    )
    ax.text(
        0.05,
        -0.35,
        footer,
        fontsize=6.5,
        color="#666",
        ha="left",
        va="top",
        wrap=True,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="PROE heatmap table for NFL teams")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument(
        "--through-week",
        type=int,
        default=None,
        help="Optional: limit REG weeks 1..N (default: all available)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=HERE,
        help="Output directory for csv/png/md",
    )
    args = ap.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"proe_table_{args.season}_reg"
    if args.through_week is not None:
        tag = f"proe_table_{args.season}_w{args.through_week:02d}"

    print(
        f"Building PROE table season={args.season} through_week={args.through_week}",
        file=sys.stderr,
    )
    df, meta = compute(args.season, args.through_week)
    logos = ensure_logos()

    csv_path = out_dir / f"{tag}.csv"
    png_path = out_dir / f"{tag}.png"
    md_path = out_dir / f"{tag}.md"

    # Force standard names when full season
    if args.through_week is None:
        csv_path = out_dir / f"proe_table_{args.season}_reg.csv"
        png_path = out_dir / f"proe_table_{args.season}_reg.png"
        md_path = out_dir / f"proe_table_{args.season}_reg.md"

    write_csv(df, csv_path)
    write_markdown(df, md_path, meta)
    plot_table(df, logos, meta, png_path)

    print(f"Wrote {csv_path}", file=sys.stderr)
    print(f"Wrote {png_path}", file=sys.stderr)
    print(f"Wrote {md_path}", file=sys.stderr)
    print(f"Neutral: {meta['neutral_definition']}", file=sys.stderr)
    print(f"pass_oe: {meta['pass_oe_scale']}", file=sys.stderr)
    if meta["missing_teams"]:
        print(f"Missing: {meta['missing_teams']}", file=sys.stderr)
    else:
        print("All 32 teams present", file=sys.stderr)

    # Print text table for parent agent
    print("\nTEAM | PASS% | PROE | NEUTRAL | LAST4 | REDZONE")
    for r in df.to_dicts():
        print(
            "{team} | {pr} | {o} | {n} | {l4} | {rz}".format(
                team=r.get("team_name") or r.get("team_abbr"),
                pr=_fmt(r.get("pass_rate")),
                o=_fmt(r.get("proe_overall"), signed=True),
                n=_fmt(r.get("proe_neutral"), signed=True),
                l4=_fmt(r.get("proe_last4"), signed=True),
                rz=_fmt(r.get("proe_redzone"), signed=True),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
