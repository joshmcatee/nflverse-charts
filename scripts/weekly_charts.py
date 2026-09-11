#!/usr/bin/env python
"""Weekly NFL team charts: implied 5/4/3 routes on dropbacks + pass rate over expected.

Data: nflverse via nflreadpy.
Route-runner panel is a PROXY: FTN n_offense_backfield on qb_dropbacks
  0 backfield -> 5, 1 -> 4, 2 -> 3.
Trumedia route-runner counts are not in nflverse. Do not label this as Trumedia.

PROE: mean(pass), mean(xpass), mean(pass_oe) from load_pbp on regular-season
plays with xpass not null.

Usage:
  python weekly_charts.py --season 2025
  python weekly_charts.py --season 2025 --through-week 14
"""
from __future__ import annotations

import argparse
import io
import json
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
DEFAULT_OUT = HERE
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
TEAM_ORDER = [t for _, teams in DIVISIONS for t in teams]

# backfield occupancy -> implied route runners (empty / 1-back / 2-back)
BACKFIELD_TO_ROUTES = {0: 5, 1: 4, 2: 3}
ROUTE_LEVELS = [5, 4, 3]


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
    # current 32 only
    teams = teams.filter(pl.col("team_abbr").is_in(TEAM_ORDER))
    return teams.select(
        [
            "team_abbr",
            "team_nick",
            "team_color",
            "team_color2",
            "team_color3",
            "team_logo_espn",
            "team_logo_squared",
        ]
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
        urls = [row.get("team_logo_espn"), row.get("team_logo_squared")]
        ok = False
        for url in urls:
            if not url:
                continue
            try:
                r = requests.get(url, timeout=20)
                r.raise_for_status()
                dest.write_bytes(r.content)
                # verify image
                Image.open(io.BytesIO(r.content)).verify()
                paths[abbr] = dest
                ok = True
                break
            except Exception:
                continue
        if not ok:
            print(f"logo miss {abbr}")
    return paths


def _week_filter(df: pl.DataFrame, through_week: int | None) -> pl.DataFrame:
    if through_week is None:
        return df
    return df.filter(pl.col("week") <= through_week)


def latest_reg_week(pbp: pl.DataFrame) -> int:
    w = (
        pbp.filter(pl.col("season_type") == "REG")
        .select(pl.col("week").max())
        .item()
    )
    return int(w)


def compute_route_rates(season: int, through_week: int | None) -> tuple[pl.DataFrame, dict]:
    pbp = nfl.load_pbp(seasons=season)
    ftn = nfl.load_ftn_charting(seasons=season)

    pbp = pbp.filter(pl.col("season_type") == "REG")
    pbp = _week_filter(pbp, through_week)
    pbp = pbp.with_columns(pl.col("play_id").cast(pl.Int64, strict=False))

    ftn = ftn.with_columns(pl.col("nflverse_play_id").cast(pl.Int64, strict=False))
    ftn_s = ftn.select(
        [
            "nflverse_game_id",
            "nflverse_play_id",
            "n_offense_backfield",
            "is_qb_sneak",
        ]
    )

    j = pbp.join(
        ftn_s,
        left_on=["game_id", "play_id"],
        right_on=["nflverse_game_id", "nflverse_play_id"],
        how="inner",
    )

    drops = j.filter(
        (pl.col("qb_dropback") == 1)
        & (pl.col("n_offense_backfield").is_not_null())
        & (pl.col("is_qb_sneak").fill_null(False) == False)  # noqa: E712
        & (pl.col("posteam").is_not_null())
    )
    # exclude kneels/spikes if columns exist
    for col, bad in (("qb_kneel", 1), ("qb_spike", 1)):
        if col in drops.columns:
            drops = drops.filter(pl.col(col).fill_null(0) != bad)

    drops = drops.with_columns(
        pl.col("n_offense_backfield")
        .replace_strict(BACKFIELD_TO_ROUTES, default=None)
        .alias("implied_routes")
    )

    tot = drops.group_by("posteam").len().rename({"len": "dropbacks", "posteam": "team"})
    by = (
        drops.filter(pl.col("implied_routes").is_not_null())
        .group_by(["posteam", "implied_routes"])
        .len()
        .rename({"len": "n", "posteam": "team"})
    )
    rates = tot.join(by, on="team", how="left")
    rates = rates.with_columns((pl.col("n") / pl.col("dropbacks")).alias("rate"))

    meta = {
        "load": ["load_pbp", "load_ftn_charting"],
        "season": season,
        "season_type": "REG",
        "through_week": through_week or latest_reg_week(pbp),
        "dropbacks": int(drops.height),
        "joined_plays": int(j.height),
        "pbp_plays": int(pbp.height),
        "note": "implied_routes from FTN n_offense_backfield: 0->5, 1->4, 2->3",
    }
    return rates, meta


def compute_proe(season: int, through_week: int | None) -> tuple[pl.DataFrame, dict]:
    pbp = nfl.load_pbp(seasons=season)
    pbp = pbp.filter(pl.col("season_type") == "REG")
    pbp = _week_filter(pbp, through_week)
    plays = pbp.filter(
        pl.col("xpass").is_not_null()
        & pl.col("pass").is_not_null()
        & pl.col("posteam").is_not_null()
    )
    out = (
        plays.group_by("posteam")
        .agg(
            pl.len().alias("plays"),
            pl.col("pass").mean().alias("actual_pass"),
            pl.col("xpass").mean().alias("expected_pass"),
            pl.col("pass_oe").mean().alias("proe"),
        )
        .rename({"posteam": "team"})
    )
    # pass_oe is already (pass - xpass) * 100
    meta = {
        "load": ["load_pbp"],
        "season": season,
        "season_type": "REG",
        "through_week": through_week or latest_reg_week(pbp),
        "plays": int(plays.height),
        "columns": ["pass", "xpass", "pass_oe", "posteam", "week", "season_type"],
    }
    return out, meta


def rank_desc(values: dict[str, float]) -> dict[str, int]:
    ordered = sorted(values.items(), key=lambda kv: kv[1], reverse=True)
    return {team: i + 1 for i, (team, _) in enumerate(ordered)}


def plot_route_runners(
    rates: pl.DataFrame,
    teams: pl.DataFrame,
    meta: dict,
    out_path: Path,
) -> None:
    # team -> {5: rate, 4: rate, 3: rate}
    wide: dict[str, dict[int, float]] = {t: {5: 0.0, 4: 0.0, 3: 0.0} for t in TEAM_ORDER}
    dropbacks: dict[str, int] = {}
    for row in rates.iter_rows(named=True):
        team = row["team"]
        if team not in wide:
            continue
        dropbacks[team] = int(row["dropbacks"])
        lvl = row["implied_routes"]
        if lvl in wide[team] and row["rate"] is not None:
            wide[team][int(lvl)] = float(row["rate"])

    ranks = {
        lvl: rank_desc({t: wide[t][lvl] for t in TEAM_ORDER})
        for lvl in ROUTE_LEVELS
    }

    colors = {
        r["team_abbr"]: (
            _hex(r["team_color"], "#333333"),
            _hex(r["team_color2"], "#888888"),
            _hex(r["team_color3"], "#555555"),
            r["team_nick"] or r["team_abbr"],
        )
        for r in teams.iter_rows(named=True)
    }

    fig, axes = plt.subplots(4, 8, figsize=(22, 11), sharey=True)
    fig.patch.set_facecolor("#f4f4f4")
    week = meta["through_week"]
    fig.suptitle(
        "How often teams send out X implied route runners on QB dropbacks\n"
        "proxy: FTN n_offense_backfield 0/1/2 → 5/4/3  |  not Trumedia",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    for i, team in enumerate(TEAM_ORDER):
        ax = axes[i // 8][i % 8]
        ax.set_facecolor("#f4f4f4")
        c1, c2, c3, nick = colors.get(team, ("#333", "#888", "#555", team))
        bar_colors = [c1, c2, c3]
        vals = [wide[team][lvl] for lvl in ROUTE_LEVELS]
        xs = [0, 1, 2]
        ax.bar(xs, vals, color=bar_colors, width=0.72, zorder=3)
        ax.set_xticks(xs, ["5", "4", "3"], fontsize=8)
        ax.set_ylim(0, 0.90)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", color="white", linewidth=1.2, zorder=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(nick.upper(), color=c1, fontsize=10, fontweight="bold", pad=8)
        for x, v, lvl in zip(xs, vals, ROUTE_LEVELS):
            rk = ranks[lvl][team]
            ax.text(
                x,
                v + 0.02,
                f"{100*v:.1f}%\n(#{rk})",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#222",
            )
        if i % 8 == 0:
            ax.set_ylabel("Rate of dropbacks", fontsize=8)

    fig.text(
        0.01,
        0.01,
        "Data: nflverse load_pbp + load_ftn_charting (CC-BY-SA). "
        "Implied routes ≠ Trumedia route runners. Denominator = all charted dropbacks.",
        fontsize=8,
        color="#555",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)

    summary = {
        "meta": meta,
        "teams": {
            t: {
                "rate_5": wide[t][5],
                "rate_4": wide[t][4],
                "rate_3": wide[t][3],
                "rank_5": ranks[5][t],
                "rank_4": ranks[4][t],
                "rank_3": ranks[3][t],
                "dropbacks": dropbacks.get(t),
            }
            for t in TEAM_ORDER
        },
    }
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2))


def plot_proe(
    proe: pl.DataFrame,
    teams: pl.DataFrame,
    logos: dict[str, Path],
    meta: dict,
    out_path: Path,
) -> None:
    rows = {r["team"]: r for r in proe.iter_rows(named=True) if r["team"] in TEAM_ORDER}
    week = meta["through_week"]

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor("#f7f7f7")
    ax.set_facecolor("#f7f7f7")
    ax.set_title(
        "Pass rate over expected",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Expected pass rate", fontsize=11)
    ax.set_ylabel("Actual pass rate", fontsize=11)

    xs = np.array([rows[t]["expected_pass"] for t in TEAM_ORDER if t in rows])
    ys = np.array([rows[t]["actual_pass"] for t in TEAM_ORDER if t in rows])
    lo = float(min(xs.min(), ys.min()) - 0.02)
    hi = float(max(xs.max(), ys.max()) + 0.02)
    grid = np.linspace(lo, hi, 200)
    ax.plot(grid, grid, color="black", linewidth=1.4, zorder=1)
    for d in (-0.08, -0.04, 0.04, 0.08):
        ax.plot(grid, grid + d, color="#cccccc", linewidth=0.8, zorder=0)

    ax.text(lo + 0.005, hi - 0.01, "Throw more than expected", fontsize=9, color="#444")
    ax.text(hi - 0.20, lo + 0.012, "Throw less than expected", fontsize=9, color="#444")

    for t in TEAM_ORDER:
        if t not in rows:
            continue
        x = rows[t]["expected_pass"]
        y = rows[t]["actual_pass"]
        if t in logos:
            try:
                img = Image.open(logos[t]).convert("RGBA")
                img.thumbnail((36, 36), Image.Resampling.LANCZOS)
                im = OffsetImage(np.asarray(img), zoom=1.0)
                ab = AnnotationBbox(im, (x, y), frameon=False, zorder=3)
                ax.add_artist(ab)
            except Exception:
                ax.text(x, y, t, ha="center", va="center", fontsize=8, zorder=3)
        else:
            ax.text(x, y, t, ha="center", va="center", fontsize=8, zorder=3)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.set_aspect("equal", adjustable="box")
    for sp in ax.spines.values():
        sp.set_color("#dddddd")
    fig.text(
        0.01,
        0.01,
        "Team on black line has PROE of 0%.  "
        "Expected pass rate = nflfastR xpass.  "
        f"Data: nflverse load_pbp {meta['season']} REG through week {week}  |  {meta['plays']} plays",
        fontsize=8,
        color="#555",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)

    summary = {
        "meta": meta,
        "teams": {
            t: {
                "actual_pass": rows[t]["actual_pass"],
                "expected_pass": rows[t]["expected_pass"],
                "proe": rows[t]["proe"],
                "plays": rows[t]["plays"],
            }
            for t in TEAM_ORDER
            if t in rows
        },
    }
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2025)
    p.add_argument("--through-week", type=int, default=None)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--skip-routes", action="store_true")
    p.add_argument("--skip-proe", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    teams = load_team_meta()
    logos = download_logos(teams)

    week_tag = f"w{args.through_week}" if args.through_week else "reg"
    if not args.skip_routes:
        rates, meta = compute_route_rates(args.season, args.through_week)
        path = outdir / f"route_runners_{args.season}_{week_tag}.png"
        plot_route_runners(rates, teams, meta, path)
        print("wrote", path)
        print("routes_meta", json.dumps(meta))
    if not args.skip_proe:
        proe, meta = compute_proe(args.season, args.through_week)
        path = outdir / f"proe_{args.season}_{week_tag}.png"
        plot_proe(proe, teams, logos, meta, path)
        print("wrote", path)
        print("proe_meta", json.dumps(meta))


if __name__ == "__main__":
    main()
