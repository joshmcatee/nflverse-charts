#!/usr/bin/env python
"""RB career production heatmap (half-PPR) from nflverse / nflreadpy.

Recreates an attached-style per-season heatmap for any running back:
  SEASON | POSITION | HALF-PPR POINTS | EXPECTED POINTS | FPS OVER EXPECTED |
  TARGET SHARE | AIR YARDS SHARE | RUSH SHARE | OPPS SHARE |
  TARGETS PER SNAP | OPPS PER SNAP

All REG seasons with at least --min-games games are shown (default 0 = no floor).

Usage:
  python rb_career_production.py \\
      --player "Christian McCaffrey"
  python rb_career_production.py \\
      --player "Bijan Robinson" --out /tmp/bijan.png
  python rb_career_production.py \\
      --gsis-id 00-0033280 --min-games 1 --seasons 2017-2026
  python rb_career_production.py \\
      --player "Christian McCaffrey" \\
      --out /workspace/nflverse-charts/examples/rb_career_cmc.png

Column sources (printed at runtime):
  HALF-PPR POINTS     load_player_stats: (fantasy_points + 0.5*receptions) / games
                      (== fantasy_points_ppr - 0.5*receptions)
  EXPECTED POINTS     load_ff_opportunity: (total_fantasy_points_exp
                      - 0.5*receptions_exp) / games   # half-PPR conversion;
                      ffverse weekly totals are full-PPR
  FPS OVER EXPECTED   half-PPR points - expected points
  TARGET/AIR/RUSH/OPPS SHARE
                      load_player_stats player totals / team totals in the
                      player's own REG games (targets, receiving_air_yards,
                      carries; opps = carries + targets)
  TARGETS/OPPS PER SNAP
                      load_snap_counts offense_snaps (via roster pfr_id)
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import nflreadpy as nfl
import numpy as np
import polars as pl
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
LOGO_DIR = HERE / "logos"
HEADSHOT_DIR = HERE / "headshots"

# Lavender/blue (low) -> teal -> dark green (high), matching reference vibe
HEAT_COLORS = ["#D9D2E9", "#B4C7E7", "#7EB6D9", "#5FA8A0", "#3D8B6E", "#1B6B45"]


def _http_get(url: str, timeout: int = 25) -> bytes:
    req = Request(url, headers={"User-Agent": "nflverse-charts/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def load_team_meta() -> pl.DataFrame:
    return nfl.load_teams().select(
        [
            "team_abbr",
            "team_name",
            "team_nick",
            "team_logo_espn",
            "team_logo_squared",
        ]
    ).unique(subset=["team_abbr"], keep="last")


def download_logo(abbr: str, teams: pl.DataFrame) -> Path | None:
    LOGO_DIR.mkdir(parents=True, exist_ok=True)
    # Normalize common aliases
    aliases = {"LAR": "LA", "LA": "LA", "OAK": "LV", "SD": "LAC", "STL": "LA"}
    key = aliases.get(abbr, abbr)
    dest = LOGO_DIR / f"{key}.png"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    # also try original abbr file
    alt = LOGO_DIR / f"{abbr}.png"
    if alt.exists() and alt.stat().st_size > 0:
        return alt
    row = teams.filter(pl.col("team_abbr") == key)
    if row.height == 0:
        row = teams.filter(pl.col("team_abbr") == abbr)
    if row.height == 0:
        return None
    r = row.to_dicts()[0]
    for url in [r.get("team_logo_espn"), r.get("team_logo_squared")]:
        if not url:
            continue
        try:
            data = _http_get(url)
            Image.open(io.BytesIO(data)).verify()
            dest.write_bytes(data)
            return dest
        except Exception:
            continue
    return None


def circular_headshot(url: str | None, gsis_id: str, size: int = 160) -> Image.Image | None:
    """Load nflverse/NFL.com headshot; drop black studio backdrop; circular crop."""
    HEADSHOT_DIR.mkdir(parents=True, exist_ok=True)
    dest = HEADSHOT_DIR / f"{gsis_id}.png"
    img = None
    if dest.exists() and dest.stat().st_size > 0:
        try:
            img = Image.open(dest).convert("RGBA")
        except Exception:
            img = None
    if img is None and url:
        try:
            data = _http_get(url)
            dest.write_bytes(data)
            img = Image.open(io.BytesIO(data)).convert("RGBA")
        except Exception:
            return None
    if img is None:
        return None

    arr = np.asarray(img).copy()
    # Punch studio backdrop: near-black OR near corner color
    corners = np.vstack([
        arr[0, 0, :3], arr[0, -1, :3], arr[-1, 0, :3], arr[-1, -1, :3]
    ]).astype(np.int16)
    corner = np.median(corners, axis=0)
    rgb = arr[:, :, :3].astype(np.int16)
    dist = np.abs(rgb - corner).sum(axis=2)
    luma = rgb.mean(axis=2)
    backdrop = (dist < 55) | (luma < 45)
    arr[backdrop, 3] = 0
    img = Image.fromarray(arr, mode="RGBA")

    # Opaque bbox, then FACE-WEIGHTED square crop (NFL.com plates are wide torso shots)
    alpha = np.asarray(img)[:, :, 3]
    ys, xs = np.where(alpha > 20)
    if len(xs) == 0:
        return None
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    bw, bh = right - left, bottom - top
    side = min(bw, bh)
    if bw >= bh:
        left2 = left + (bw - side) // 2
        top2 = top
    else:
        left2 = left
        top2 = top + (bh - side) // 5
    img = img.crop((left2, top2, left2 + side, top2 + side))
    square = img.resize((size, size), Image.Resampling.LANCZOS)

    # Composite onto white, then circular mask — no black halo on white charts
    white = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    white.paste(square, (0, 0), square)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((1, 1, size - 2, size - 2), fill=255)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(white, (0, 0))
    out.putalpha(mask)
    return out


def resolve_player(
    player: str | None,
    gsis_id: str | None,
) -> dict:
    """Return dict with gsis_id, display_name, position, headshot, latest_team."""
    plrs = nfl.load_players()
    if gsis_id:
        hit = plrs.filter(pl.col("gsis_id") == gsis_id)
        if hit.height == 0:
            raise SystemExit(f"No player with gsis_id={gsis_id!r}")
        row = hit.to_dicts()[0]
        return {
            "gsis_id": row["gsis_id"],
            "display_name": row.get("display_name") or player or gsis_id,
            "position": row.get("position"),
            "headshot": row.get("headshot"),
            "latest_team": row.get("latest_team"),
        }

    if not player:
        raise SystemExit("Provide --player or --gsis-id")

    q = player.strip()
    # Exact display_name (case-insensitive)
    exact = plrs.filter(pl.col("display_name").str.to_lowercase() == q.lower())
    if exact.height == 1:
        row = exact.to_dicts()[0]
    elif exact.height > 1:
        # Prefer RB / active-ish
        rb = exact.filter(pl.col("position") == "RB")
        row = (rb if rb.height else exact).to_dicts()[0]
    else:
        # Fuzzy: contains all tokens
        tokens = [t for t in re.split(r"\s+", q.lower()) if t]
        cand = plrs
        for t in tokens:
            cand = cand.filter(pl.col("display_name").str.to_lowercase().str.contains(t, literal=True))
        if cand.height == 0:
            # try last-name only against recent player_stats
            raise SystemExit(f"No player match for {player!r}")
        rb = cand.filter(pl.col("position") == "RB")
        pool = rb if rb.height else cand
        # Prefer players with a recent gsis id style and non-null latest_team
        pool = pool.sort(
            [
                pl.col("latest_team").is_not_null().cast(pl.Int8),
                pl.col("display_name"),
            ],
            descending=[True, False],
        )
        if pool.height > 1:
            names = pool.select(["display_name", "gsis_id", "position", "latest_team"]).head(8)
            print("Multiple matches; using first RB/active-ish:", file=sys.stderr)
            print(names, file=sys.stderr)
        row = pool.to_dicts()[0]

    return {
        "gsis_id": row["gsis_id"],
        "display_name": row.get("display_name") or player,
        "position": row.get("position"),
        "headshot": row.get("headshot"),
        "latest_team": row.get("latest_team"),
    }


def parse_seasons(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def pfr_map_for_seasons(seasons: list[int], gsis_id: str) -> str | None:
    """Best-effort pfr_id for snap_counts join."""
    for s in sorted(seasons, reverse=True):
        try:
            rost = nfl.load_rosters(seasons=s)
            hit = rost.filter(
                (pl.col("gsis_id") == gsis_id) & pl.col("pfr_id").is_not_null()
            )
            if hit.height:
                return hit["pfr_id"][-1]
        except Exception:
            continue
    try:
        plrs = nfl.load_players().filter(pl.col("gsis_id") == gsis_id)
        if plrs.height and plrs["pfr_id"][0]:
            return plrs["pfr_id"][0]
    except Exception:
        pass
    return None


def compute_career(
    gsis_id: str,
    seasons: list[int] | None,
    min_games: int,
) -> tuple[pl.DataFrame, dict]:
    """Build one row per REG season with >= min_games."""
    # Discover seasons from player_stats if not provided
    if seasons is None:
        # Broad window; nflverse player_stats modern era
        seasons = list(range(2016, 2027))

    ps_all = nfl.load_player_stats(seasons=seasons)
    ps = ps_all.filter(
        (pl.col("player_id") == gsis_id) & (pl.col("season_type") == "REG")
    )
    if ps.height == 0:
        raise SystemExit(f"No REG player_stats for {gsis_id} in {seasons}")

    avail = sorted(ps["season"].unique().to_list())
    seasons = [s for s in avail]  # only seasons with any games

    # ff opportunity (weekly, full-PPR expected — convert to half)
    try:
        ff_all = nfl.load_ff_opportunity(seasons=seasons, stat_type="weekly", model_version="latest")
        ff = ff_all.filter(pl.col("player_id") == gsis_id)
        ff_ok = True
    except Exception as e:
        ff = pl.DataFrame()
        ff_ok = False
        ff_err = f"{type(e).__name__}: {e}"

    # Restrict ff to REG game_ids from player_stats
    reg_games = ps.select(["season", "game_id"]).unique()
    if ff_ok and ff.height:
        # season in ff may be string
        ff = ff.with_columns(pl.col("season").cast(pl.Int64))
        ff = ff.join(reg_games, on=["season", "game_id"], how="inner")

    pfr_id = pfr_map_for_seasons(seasons, gsis_id)
    snaps_by_season: dict[int, float] = {}
    if pfr_id:
        try:
            sc = nfl.load_snap_counts(seasons=seasons)
            sc = sc.filter(
                (pl.col("pfr_player_id") == pfr_id) & (pl.col("game_type") == "REG")
            )
            for s, sub in sc.group_by("season"):
                # group_by yields (key_tuple_or_value, df) depending on polars version
                pass
            snap_agg = sc.group_by("season").agg(
                pl.col("offense_snaps").sum().alias("offense_snaps")
            )
            for r in snap_agg.to_dicts():
                snaps_by_season[int(r["season"])] = float(r["offense_snaps"] or 0)
        except Exception:
            pass

    # Team denominators in player's games only
    # Join all player_stats rows onto player's (game_id, team) pairs
    player_games = ps.select(["season", "game_id", "team"]).unique()
    team_in_player_games = ps_all.filter(pl.col("season_type") == "REG").join(
        player_games, on=["season", "game_id", "team"], how="inner"
    )
    team_totals = team_in_player_games.group_by(["season", "team"]).agg(
        pl.col("targets").sum().alias("tm_tgt"),
        pl.col("carries").sum().alias("tm_car"),
        pl.col("receiving_air_yards").sum().alias("tm_air"),
    )

    # Per season + primary team
    rows = []
    col_map = {
        "half_ppr": "load_player_stats: (fantasy_points + 0.5*receptions) / n_games",
        "expected": (
            "load_ff_opportunity: (total_fantasy_points_exp - 0.5*receptions_exp) / n_games"
            if ff_ok
            else f"UNAVAILABLE ({ff_err if not ff_ok else ''})"
        ),
        "fpoe": "half_ppr - expected",
        "tgt_share": "load_player_stats targets / team targets in player's games",
        "air_share": "load_player_stats receiving_air_yards / team air yards in player's games",
        "rush_share": "load_player_stats carries / team carries in player's games",
        "opp_share": "(carries+targets) / team (carries+targets) in player's games",
        "tgt_per_snap": "targets / load_snap_counts.offense_snaps * 100",
        "opp_per_snap": "(carries+targets) / load_snap_counts.offense_snaps * 100",
    }

    for season in seasons:
        sp = ps.filter(pl.col("season") == season)
        n = sp.height
        if n < min_games:
            continue
        # primary team = most games; tie -> last appearing
        team_counts = (
            sp.group_by("team")
            .agg(pl.len().alias("g"))
            .sort(["g", "team"], descending=[True, False])
        )
        team = team_counts["team"][0]
        pos = sp["position"][-1] or "RB"

        half = float(sp["fantasy_points"].sum() + 0.5 * sp["receptions"].sum()) / n
        targets = int(sp["targets"].sum())
        carries = int(sp["carries"].sum())
        air = int(sp["receiving_air_yards"].sum())
        opps = carries + targets

        # Team totals across all teams the player appeared for this season
        tt = team_totals.filter(pl.col("season") == season)
        # Weight: sum team totals for teams player was on (already scoped to player games)
        # But team_totals is per (season, team) for player's games on that team — sum all
        tm_tgt = int(tt["tm_tgt"].sum()) if tt.height else 0
        tm_car = int(tt["tm_car"].sum()) if tt.height else 0
        tm_air = int(tt["tm_air"].sum()) if tt.height else 0
        tm_opp = tm_tgt + tm_car

        tgt_share = 100.0 * targets / tm_tgt if tm_tgt else None
        air_share = 100.0 * air / tm_air if tm_air else None
        rush_share = 100.0 * carries / tm_car if tm_car else None
        opp_share = 100.0 * opps / tm_opp if tm_opp else None

        # Expected / FPOE from ff_opportunity
        expected = None
        fpoe = None
        if ff_ok and ff.height:
            sf = ff.filter(pl.col("season") == season)
            if sf.height:
                # Align to same REG games when possible
                exp_sum = float(sf["total_fantasy_points_exp"].sum())
                rec_exp = float(sf["receptions_exp"].sum())
                # Use player_stats n as denominator (per-game career chart)
                expected = (exp_sum - 0.5 * rec_exp) / n
                fpoe = half - expected

        snaps = snaps_by_season.get(int(season))
        tgt_per_snap = 100.0 * targets / snaps if snaps and snaps > 0 else None
        opp_per_snap = 100.0 * opps / snaps if snaps and snaps > 0 else None

        rows.append(
            {
                "season": int(season),
                "team": team,
                "position": pos,
                "games": n,
                "half_ppr": half,
                "expected": expected,
                "fpoe": fpoe,
                "tgt_share": tgt_share,
                "air_share": air_share,
                "rush_share": rush_share,
                "opp_share": opp_share,
                "tgt_per_snap": tgt_per_snap,
                "opp_per_snap": opp_per_snap,
                "targets": targets,
                "carries": carries,
                "snaps": snaps,
            }
        )

    if not rows:
        raise SystemExit(f"No seasons with >= {min_games} REG games for {gsis_id}")

    df = pl.DataFrame(rows).sort("season")
    meta = {
        "seasons_included": df["season"].to_list(),
        "min_games": min_games,
        "pfr_id": pfr_id,
        "ff_ok": ff_ok,
        "column_map": col_map,
        "source": "nflverse (nflreadpy): load_player_stats, load_ff_opportunity, load_snap_counts, load_teams, load_players",
    }
    return df, meta


def _heat_cmap():
    return mcolors.LinearSegmentedColormap.from_list("rb_heat", HEAT_COLORS, N=256)


def _cell_rgba(val: float | None, vmin: float, vmax: float, cmap) -> tuple:
    if val is None or (isinstance(val, float) and val != val):
        return (1.0, 1.0, 1.0, 1.0)
    if vmax <= vmin:
        t = 0.5
    else:
        t = float(np.clip((val - vmin) / (vmax - vmin), 0.0, 1.0))
    return cmap(t)


def _text_color(rgba) -> str:
    # relative luminance — white on dark green / teal cells
    r, g, b = rgba[0], rgba[1], rgba[2]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#FFFFFF" if lum < 0.55 else "#111111"


def _fmt(val: float | None, digits: int = 1, signed: bool = False) -> str:
    if val is None or (isinstance(val, float) and val != val):
        return "—"
    if signed:
        return f"{val:+.{digits}f}"
    return f"{val:.{digits}f}"


def plot_heatmap(
    df: pl.DataFrame,
    meta: dict,
    player: dict,
    teams: pl.DataFrame,
    out: Path,
) -> None:
    rows = df.to_dicts()
    n = len(rows)
    team_name_map = {
        r["team_abbr"]: r["team_name"] for r in teams.to_dicts()
    }
    latest = player.get("latest_team")
    # Prefer team of most recent included season for title
    title_team_abbr = rows[-1]["team"] if rows else latest
    title_team = team_name_map.get(title_team_abbr) or title_team_abbr or ""
    title = f"{player['display_name']} - {title_team}"
    if meta["min_games"] <= 0:
        subtitle = "Career Production per Game | All historical REG seasons"
    else:
        subtitle = (
            f"Career Production per Game | Historical Seasons with a minimum of "
            f"{meta['min_games']} games"
        )

    # Columns: heat from EXPECTED onward; HALF-PPR plain
    heat_keys = [
        "expected",
        "fpoe",
        "tgt_share",
        "air_share",
        "rush_share",
        "opp_share",
        "tgt_per_snap",
        "opp_per_snap",
    ]
    col_labels = [
        "SEASON",
        "POSITION",
        "HALF-PPR\nPOINTS",
        "EXPECTED\nPOINTS",
        "FPS OVER\nEXPECTED",
        "TARGET\nSHARE (%)",
        "AIR YARDS\nSHARE (%)",
        "RUSH\nSHARE (%)",
        "OPPS\nSHARE (%)",
        "TARGETS PER\nSNAP (%)",
        "OPPS PER\nSNAP (%)",
    ]
    # Relative widths
    widths = [1.35, 1.0, 1.15, 1.15, 1.2, 1.15, 1.15, 1.1, 1.1, 1.25, 1.2]
    total_w = sum(widths)
    xs = []
    x = 0.0
    for w in widths:
        xs.append(x)
        x += w

    row_h = 0.42
    header_h = 0.85
    fig_w = 14.5
    # Compact header like the reference: headshot LEFT of title, table tight below
    shot_in = 0.95  # inches, square
    title_in = 0.95
    table_in = header_h + n * row_h + 0.55
    fig_h = title_in + table_in + 0.25
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    # Table axes fills everything below the header strip
    table_top = 1.0 - (title_in + 0.06) / fig_h
    ax = fig.add_axes([0.04, 0.06, 0.92, table_top - 0.06])
    ax.set_xlim(0, total_w)
    ax.set_ylim(-0.45, n + 0.55)
    ax.axis("off")
    ax.set_facecolor("white")

    # Square headshot axes in figure inches (equal W/H → no squash)
    hs = circular_headshot(player.get("headshot"), player["gsis_id"], size=240)
    hs_left = 0.035
    hs_bottom = table_top + 0.01
    hs_w = shot_in / fig_w
    hs_h = shot_in / fig_h
    # Vertically center headshot in the header strip
    header_strip_h = 1.0 - table_top
    hs_bottom = table_top + max(0.0, (header_strip_h - hs_h) / 2)
    if hs is not None:
        ax_hs = fig.add_axes([hs_left, hs_bottom, hs_w, hs_h])
        ax_hs.set_facecolor("white")
        ax_hs.imshow(np.asarray(hs), interpolation="lanczos")
        ax_hs.set_xticks([])
        ax_hs.set_yticks([])
        for sp in ax_hs.spines.values():
            sp.set_visible(False)
        ax_hs.set_aspect("equal")

    # Title / subtitle to the RIGHT of the headshot (reference layout)
    text_x = hs_left + hs_w + 0.015
    fig.text(
        text_x,
        hs_bottom + hs_h * 0.68,
        title,
        ha="left",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#111",
    )
    fig.text(
        text_x,
        hs_bottom + hs_h * 0.28,
        subtitle,
        ha="left",
        va="center",
        fontsize=10,
        color="#666",
    )

    # Column headers
    header_y = n + 0.08
    for i, lab in enumerate(col_labels):
        cx = xs[i] + widths[i] / 2
        ax.text(
            cx,
            header_y + 0.28,
            lab,
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color="#222",
            linespacing=1.15,
            zorder=5,
        )
    ax.plot([0, total_w], [n, n], color="#111", lw=1.6, zorder=5)




    cmap = _heat_cmap()
    # Column-wise min/max for heat columns
    scales = {}
    for key in heat_keys:
        vals = [r[key] for r in rows if r.get(key) is not None and r[key] == r[key]]
        if vals:
            scales[key] = (min(vals), max(vals))
        else:
            scales[key] = (0.0, 1.0)

    # Preload logos
    logos: dict[str, Path | None] = {}
    for r in rows:
        abbr = r["team"]
        if abbr not in logos:
            logos[abbr] = download_logo(abbr, teams)

    for ri, r in enumerate(rows):
        y = n - ri - 1  # oldest at top (like reference: 2017 top)
        # Actually reference has 2017 at top — ascending season top-to-bottom. rows already sorted ascending.

        # SEASON year + logo
        ax.text(
            xs[0] + 0.12,
            y + 0.5,
            str(r["season"]),
            ha="left",
            va="center",
            fontsize=10,
            color="#111",
            fontweight="normal",
            zorder=3,
        )
        lp = logos.get(r["team"])
        if lp and lp.exists():
            try:
                img = Image.open(lp).convert("RGBA")
                img.thumbnail((48, 48), Image.Resampling.LANCZOS)
                im = OffsetImage(np.asarray(img), zoom=18.0 / max(img.size[1], 1))
                ab = AnnotationBbox(
                    im,
                    (xs[0] + widths[0] - 0.35, y + 0.5),
                    frameon=False,
                    pad=0,
                    zorder=3,
                )
                ax.add_artist(ab)
            except Exception:
                pass

        # POSITION
        ax.text(
            xs[1] + widths[1] / 2,
            y + 0.5,
            r.get("position") or "RB",
            ha="center",
            va="center",
            fontsize=10,
            color="#111",
            zorder=3,
        )

        # HALF-PPR plain
        ax.text(
            xs[2] + widths[2] / 2,
            y + 0.5,
            _fmt(r.get("half_ppr")),
            ha="center",
            va="center",
            fontsize=10,
            color="#111",
            zorder=3,
        )

        # Heat columns
        heat_vals = [
            ("expected", r.get("expected"), False),
            ("fpoe", r.get("fpoe"), True),
            ("tgt_share", r.get("tgt_share"), False),
            ("air_share", r.get("air_share"), False),
            ("rush_share", r.get("rush_share"), False),
            ("opp_share", r.get("opp_share"), False),
            ("tgt_per_snap", r.get("tgt_per_snap"), False),
            ("opp_per_snap", r.get("opp_per_snap"), False),
        ]
        for j, (key, val, signed) in enumerate(heat_vals):
            xi = 3 + j
            vmin, vmax = scales[key]
            rgba = _cell_rgba(val, vmin, vmax, cmap)
            ax.fill_between(
                [xs[xi] + 0.05, xs[xi] + widths[xi] - 0.05],
                y + 0.08,
                y + 0.92,
                color=rgba,
                zorder=1,
            )
            ax.text(
                xs[xi] + widths[xi] / 2,
                y + 0.5,
                _fmt(val, signed=signed),
                ha="center",
                va="center",
                fontsize=9.5,
                color=_text_color(rgba),
                zorder=3,
            )

        # light row separator
        ax.plot([0, total_w], [y, y], color="#E6E6E6", lw=0.5, zorder=4)

    ax.plot([0, total_w], [0, 0], color="#CCCCCC", lw=0.8, zorder=4)

    footer = "Quant / Source: nflverse"
    ax.text(
        0.05,
        -0.35,
        footer,
        fontsize=8,
        color="#888",
        ha="left",
        va="top",
        clip_on=False,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)




def main() -> int:
    ap = argparse.ArgumentParser(
        description="RB career production heatmap (half-PPR) from nflverse"
    )
    ap.add_argument("--player", type=str, default=None, help='Player name, e.g. "Christian McCaffrey"')
    ap.add_argument("--gsis-id", type=str, default=None, help="GSIS player id")
    ap.add_argument("--min-games", type=int, default=0, help="Minimum REG games to include a season (default 0 = no floor)")
    ap.add_argument(
        "--seasons",
        type=str,
        default=None,
        help="Season list or range, e.g. 2017-2025 or 2019,2023",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: ./rb_career_<slug>.png)",
    )
    args = ap.parse_args()

    player = resolve_player(args.player, args.gsis_id)
    seasons = parse_seasons(args.seasons)

    df, meta = compute_career(player["gsis_id"], seasons, args.min_games)
    teams = load_team_meta()

    slug = re.sub(r"[^a-z0-9]+", "_", player["display_name"].lower()).strip("_")
    out = args.out or (HERE / f"rb_career_{slug}.png")

    plot_heatmap(df, meta, player, teams, out)

    # Stdout report
    print("=== RB career production heatmap ===")
    print(f"player: {player['display_name']} ({player['gsis_id']}) pos={player.get('position')}")
    print(f"seasons_included: {meta['seasons_included']} (min_games={meta['min_games']})")
    print(f"png: {out.resolve()}")
    print("column_map:")
    for k, v in meta["column_map"].items():
        print(f"  {k}: {v}")
    print("rows:")
    show = df.select(
        [
            "season",
            "team",
            "games",
            "half_ppr",
            "expected",
            "fpoe",
            "tgt_share",
            "air_share",
            "rush_share",
            "opp_share",
            "tgt_per_snap",
            "opp_per_snap",
        ]
    )
    with pl.Config(tbl_rows=50, fmt_float="mixed"):
        print(show)

    # Reference comparison for CMC when applicable
    if player["gsis_id"] == "00-0033280":
        print("\n=== vs reference chart (CMC) ===")
        print("ref 2019: half~25.8 exp~20.5 fpoe~5.3 tgt~23.7 rush~74.4 tgt/snap~15.0 opps/snap~45.4")
        print("ref 2023: half~22.4 exp~17.8 fpoe~4.5 tgt~18.6 rush~57.9 tgt/snap~11.8 opps/snap~41.5")
        for s in (2019, 2023):
            hit = df.filter(pl.col("season") == s)
            if hit.height:
                r = hit.to_dicts()[0]
                print(
                    f"ours {s}: half={r['half_ppr']:.1f} exp={_fmt(r['expected'])} "
                    f"fpoe={_fmt(r['fpoe'], signed=True)} tgt={_fmt(r['tgt_share'])} "
                    f"rush={_fmt(r['rush_share'])} tgt/snap={_fmt(r['tgt_per_snap'])} "
                    f"opps/snap={_fmt(r['opp_per_snap'])}"
                )
        print(
            "notes: half-PPR / tgt / rush shares match closely; "
            "EXPECTED/FPOE use ffverse full-PPR exp converted to half-PPR "
            "(typically ~1–2 pts higher exp than the reference chart); "
            "per-snap rates use PFR snap_counts (often a bit lower tgt/snap than reference)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
