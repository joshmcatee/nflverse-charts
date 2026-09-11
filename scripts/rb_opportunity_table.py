#!/usr/bin/env python
"""RB opportunity / snap share table from nflverse (nflreadpy).

Columns:
  Player | Team | Pos | GMs | Snaps | Snap% | Tgt | Tgt Shr | Att | % Tm Att | Opps | % Tm Opps

Sources (cited):
  - load_snap_counts(seasons=SEASON): offense_snaps, offense_pct (fraction 0-1)
  - load_player_stats(seasons=SEASON): carries, targets, team, position (GSIS player_id)
  - load_team_stats(seasons=SEASON): team carries + targets denominators (all positions)
  - load_rosters(seasons=SEASON): pfr_id <-> gsis_id join

Regular season filter: game_type/season_type REG and weeks in --weeks (default 1-18).
Pos: RB only (snap_counts HB treated as RB; FB excluded).
Team shares use all-team rush attempts and all-team targets (not RB-only).
GMs: distinct games with offense_snaps > 0 and/or (carries + targets) > 0.
Snap%: 100 * sum(offense_snaps) / sum(team offense snaps in those snap-rows),
        where team snaps inferred as offense_snaps / offense_pct when pct > 0.

Usage:
  python rb_opportunity_table.py --season 2025 --weeks 1-18
  python rb_opportunity_table.py --season 2026 --weeks 1-18
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nflreadpy as nfl
import polars as pl

HERE = Path(__file__).resolve().parent


def parse_weeks(weeks: str) -> list[int]:
    parts = weeks.replace(" ", "").split("-")
    if len(parts) != 2:
        raise ValueError(f"--weeks must look like 1-18, got {weeks!r}")
    start, end = int(parts[0]), int(parts[1])
    if start < 1 or end < start:
        raise ValueError(f"invalid week range {weeks!r}")
    return list(range(start, end + 1))


def build_pfr_gsis_map(season: int) -> pl.DataFrame:
    """Map pfr_player_id -> gsis_id via season rosters (+ players fallback)."""
    rost = nfl.load_rosters(seasons=season)
    m = (
        rost.filter(pl.col("pfr_id").is_not_null() & pl.col("gsis_id").is_not_null())
        .select(
            pl.col("pfr_id").alias("pfr_player_id"),
            pl.col("gsis_id").alias("player_id"),
            pl.col("full_name").alias("roster_name"),
        )
        .unique(subset=["pfr_player_id"], keep="last")
    )
    # players table fill for any pfr missing on roster
    try:
        plrs = nfl.load_players()
        extra = (
            plrs.filter(pl.col("pfr_id").is_not_null() & pl.col("gsis_id").is_not_null())
            .select(
                pl.col("pfr_id").alias("pfr_player_id"),
                pl.col("gsis_id").alias("player_id"),
                pl.col("display_name").alias("roster_name"),
            )
            .unique(subset=["pfr_player_id"], keep="last")
        )
        known = m["pfr_player_id"].implode()
        m = pl.concat([m, extra.filter(~pl.col("pfr_player_id").is_in(known))])
    except Exception:
        pass
    return m


def load_snaps(season: int, weeks: list[int]) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Aggregate RB/HB snap counts. Returns (player agg, game participation, meta)."""
    meta: dict = {"snap_source": f"load_snap_counts(seasons={season})", "snap_ok": False}
    try:
        sc = nfl.load_snap_counts(seasons=season)
    except Exception as e:
        meta["snap_error"] = f"{type(e).__name__}: {e}"
        empty = pl.DataFrame(
            schema={
                "pfr_player_id": pl.Utf8,
                "snap_player": pl.Utf8,
                "snap_team": pl.Utf8,
                "snap_pos": pl.Utf8,
                "Snaps": pl.Float64,
                "Snap_pct_raw": pl.Float64,
                "snap_gms": pl.UInt32,
            }
        )
        empty_g = pl.DataFrame(
            schema={"pfr_player_id": pl.Utf8, "game_id": pl.Utf8}
        )
        return empty, empty_g, meta

    meta["snap_ok"] = True
    meta["snap_weeks_available"] = sorted(sc["week"].unique().to_list())
    sc = sc.filter(
        (pl.col("game_type") == "REG")
        & pl.col("week").is_in(weeks)
        & pl.col("position").is_in(["RB", "HB"])
    )
    meta["snap_rows"] = int(sc.height)

    # Team snaps for Snap% denominator (only when offense_pct > 0)
    sc = sc.with_columns(
        pl.when(pl.col("offense_pct") > 0)
        .then(pl.col("offense_snaps") / pl.col("offense_pct"))
        .otherwise(None)
        .alias("team_off_snaps")
    )

    games = (
        sc.filter(pl.col("offense_snaps") > 0)
        .select("pfr_player_id", "game_id")
        .unique()
    )

    agg = sc.group_by("pfr_player_id").agg(
        pl.col("player").last().alias("snap_player"),
        pl.col("team").last().alias("snap_team"),
        pl.col("position").last().alias("snap_pos"),
        pl.col("offense_snaps").sum().alias("Snaps"),
        pl.col("team_off_snaps").sum().alias("team_off_snaps_sum"),
        pl.col("game_id")
        .filter(pl.col("offense_snaps") > 0)
        .n_unique()
        .alias("snap_gms"),
    )
    agg = agg.with_columns(
        pl.when(pl.col("team_off_snaps_sum") > 0)
        .then(100.0 * pl.col("Snaps") / pl.col("team_off_snaps_sum"))
        .otherwise(None)
        .alias("Snap_pct_raw")
    ).drop("team_off_snaps_sum")
    return agg, games, meta


def load_usage(season: int, weeks: list[int]) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict]:
    """Player Att/Tgt, opp games, and team denominators from player_stats / team_stats."""
    meta: dict = {
        "stats_source": f"load_player_stats(seasons={season})",
        "team_source": f"load_team_stats(seasons={season})",
    }
    ps = nfl.load_player_stats(seasons=season)
    meta["ps_weeks_available"] = sorted(ps["week"].unique().to_list())
    ps = ps.filter(
        (pl.col("season_type") == "REG") & pl.col("week").is_in(weeks)
    )

    # Team denominators: all positions (standard opportunity share)
    try:
        ts = nfl.load_team_stats(seasons=season)
        ts = ts.filter(
            (pl.col("season_type") == "REG") & pl.col("week").is_in(weeks)
        )
        team_tot = ts.group_by("team").agg(
            pl.col("carries").sum().alias("tm_att"),
            pl.col("targets").sum().alias("tm_tgt"),
        )
        meta["team_denom"] = "load_team_stats carries/targets"
    except Exception:
        team_tot = ps.group_by("team").agg(
            pl.col("carries").sum().alias("tm_att"),
            pl.col("targets").sum().alias("tm_tgt"),
        )
        meta["team_denom"] = "sum load_player_stats carries/targets (fallback)"

    # RB usage (authoritative position from player_stats)
    rb = ps.filter(pl.col("position") == "RB")
    opp_games = (
        rb.filter((pl.col("carries") + pl.col("targets")) > 0)
        .select(pl.col("player_id"), "game_id")
        .unique()
    )

    usage = rb.group_by("player_id").agg(
        pl.col("player_display_name").last().alias("ps_player"),
        pl.col("team").last().alias("ps_team"),
        pl.col("position").last().alias("ps_pos"),
        pl.col("carries").sum().alias("Att"),
        pl.col("targets").sum().alias("Tgt"),
        pl.col("game_id")
        .filter((pl.col("carries") + pl.col("targets")) > 0)
        .n_unique()
        .alias("opp_gms"),
    )
    meta["rb_usage_players"] = int(usage.height)
    return usage, opp_games, team_tot, meta


def compute_table(season: int, weeks: list[int]) -> tuple[pl.DataFrame, dict]:
    id_map = build_pfr_gsis_map(season)
    snaps, snap_games, snap_meta = load_snaps(season, weeks)
    usage, opp_games, team_tot, usage_meta = load_usage(season, weeks)

    meta = {
        "season": season,
        "weeks": weeks,
        **snap_meta,
        **usage_meta,
        "id_map_rows": int(id_map.height),
    }

    # Attach gsis to snap aggregates
    snaps_j = snaps.join(id_map, on="pfr_player_id", how="left")

    # Full outer on player_id between usage and snaps (snaps without gsis kept via synthetic key)
    # For snap-only rows missing gsis, use pfr as synthetic player_id so they remain in table.
    snaps_j = snaps_j.with_columns(
        pl.coalesce([pl.col("player_id"), pl.col("pfr_player_id")]).alias("join_id")
    )
    usage_j = usage.with_columns(pl.col("player_id").alias("join_id"))

    df = usage_j.join(
        snaps_j.select(
            "join_id",
            "pfr_player_id",
            "snap_player",
            "snap_team",
            "snap_pos",
            "Snaps",
            "Snap_pct_raw",
            "snap_gms",
            pl.col("player_id").alias("snap_gsis"),
        ),
        on="join_id",
        how="full",
        coalesce=True,
    )

    # GMs: union of snap games and opp games via gsis / pfr
    snap_games_m = snap_games.join(id_map, on="pfr_player_id", how="left").with_columns(
        pl.coalesce([pl.col("player_id"), pl.col("pfr_player_id")]).alias("join_id")
    )
    opp_games_m = opp_games.with_columns(pl.col("player_id").alias("join_id"))
    all_games = (
        pl.concat(
            [
                snap_games_m.select("join_id", "game_id"),
                opp_games_m.select("join_id", "game_id"),
            ]
        )
        .unique()
        .group_by("join_id")
        .agg(pl.len().alias("GMs"))
    )
    df = df.join(all_games, on="join_id", how="left")

    df = df.with_columns(
        [
            pl.coalesce(["ps_player", "snap_player"]).alias("Player"),
            pl.coalesce(["ps_team", "snap_team"]).alias("Team"),
            pl.lit("RB").alias("Pos"),
            pl.col("Att").fill_null(0),
            pl.col("Tgt").fill_null(0),
            pl.col("Snaps").fill_null(0.0),
            pl.col("GMs").fill_null(0),
            pl.col("snap_gms").fill_null(0),
            pl.col("opp_gms").fill_null(0),
        ]
    )
    df = df.with_columns(
        (pl.col("Att") + pl.col("Tgt")).alias("Opps")
    )

    # Keep RBs with >=1 game and (snaps or opps)
    df = df.filter(
        (pl.col("GMs") >= 1)
        & ((pl.col("Snaps") > 0) | (pl.col("Opps") > 0))
    )

    # Team share vs season team totals for displayed team
    df = df.join(team_tot, left_on="Team", right_on="team", how="left")
    df = df.with_columns(
        [
            pl.when(pl.col("tm_tgt") > 0)
            .then(100.0 * pl.col("Tgt") / pl.col("tm_tgt"))
            .otherwise(None)
            .alias("Tgt_Shr"),
            pl.when(pl.col("tm_att") > 0)
            .then(100.0 * pl.col("Att") / pl.col("tm_att"))
            .otherwise(None)
            .alias("Pct_Tm_Att"),
            pl.when((pl.col("tm_att") + pl.col("tm_tgt")) > 0)
            .then(100.0 * pl.col("Opps") / (pl.col("tm_att") + pl.col("tm_tgt")))
            .otherwise(None)
            .alias("Pct_Tm_Opps"),
            pl.col("Snap_pct_raw").alias("Snap_pct"),
        ]
    )

    # ID join quirks
    missing_gsis = (
        snaps_j.filter(pl.col("player_id").is_null())
        .select("pfr_player_id", "snap_player", "snap_team", "Snaps")
        .sort("Snaps", descending=True)
    )
    meta["snap_without_gsis"] = missing_gsis.to_dicts()
    meta["n_rows"] = int(df.height)

    out = (
        df.select(
            [
                "Player",
                "Team",
                "Pos",
                pl.col("GMs").cast(pl.Int64),
                pl.col("Snaps").cast(pl.Int64),
                pl.col("Snap_pct").alias("Snap%"),
                pl.col("Tgt").cast(pl.Int64),
                pl.col("Tgt_Shr").alias("Tgt Shr"),
                pl.col("Att").cast(pl.Int64),
                pl.col("Pct_Tm_Att").alias("% Tm Att"),
                pl.col("Opps").cast(pl.Int64),
                pl.col("Pct_Tm_Opps").alias("% Tm Opps"),
                "join_id",
                "pfr_player_id",
            ]
        )
        .sort("Opps", descending=True)
    )
    return out, meta


def format_pct(x: float | None, digits: int = 1) -> str:
    if x is None:
        return ""
    try:
        if x != x:  # NaN
            return ""
        return f"{float(x):.{digits}f}%"
    except Exception:
        return ""


def write_markdown(df: pl.DataFrame, path: Path, season: int, weeks: list[int], meta: dict) -> None:
    rows = df.drop(["join_id", "pfr_player_id"], strict=False).to_dicts()
    lines = [
        f"# RB Opportunity Share — {season} REG weeks {weeks[0]}–{weeks[-1]}",
        "",
        f"Sources: `{meta.get('snap_source')}`; `{meta.get('stats_source')}`; "
        f"team denoms via `{meta.get('team_denom')}`.",
        f"Snap%: offense_snaps / inferred team offense snaps (from offense_pct). "
        f"Pos=RB (HB from snap_counts mapped to RB; FB excluded).",
        f"Rows: {len(rows)}. Sorted by Opps desc.",
        "",
        "| Player | Team | Pos | GMs | Snaps | Snap% | Tgt | Tgt Shr | Att | % Tm Att | Opps | % Tm Opps |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {Player} | {Team} | {Pos} | {GMs} | {Snaps} | {snap} | {Tgt} | {tgt_shr} | {Att} | {tm_att} | {Opps} | {tm_opps} |".format(
                Player=r["Player"],
                Team=r["Team"],
                Pos=r["Pos"],
                GMs=r["GMs"],
                Snaps=r["Snaps"],
                snap=format_pct(r.get("Snap%")),
                Tgt=r["Tgt"],
                tgt_shr=format_pct(r.get("Tgt Shr")),
                Att=r["Att"],
                tm_att=format_pct(r.get("% Tm Att")),
                Opps=r["Opps"],
                tm_opps=format_pct(r.get("% Tm Opps")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="RB opportunity / snap share table")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--weeks", type=str, default="1-18", help="START-END inclusive")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=HERE,
        help="Directory for csv/md outputs",
    )
    ap.add_argument("--prefix", type=str, default=None, help="Filename prefix override")
    args = ap.parse_args()
    weeks = parse_weeks(args.weeks)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or f"rb_opportunity_{args.season}_reg"

    print(
        f"Building RB opportunity table: season={args.season} weeks={weeks[0]}-{weeks[-1]}",
        file=sys.stderr,
    )
    try:
        df, meta = compute_table(args.season, weeks)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        if "snap" in str(e).lower() or "404" in str(e):
            print("Hint: snap_counts may be unpublished for this season.", file=sys.stderr)
        return 1

    if not meta.get("snap_ok"):
        print(
            f"WARNING: snap_counts unavailable for {args.season}: {meta.get('snap_error')}",
            file=sys.stderr,
        )

    csv_path = out_dir / f"{prefix}.csv"
    md_path = out_dir / f"{prefix}.md"

    # CSV with rounded percents for readability (still numeric)
    csv_df = df.drop(["join_id", "pfr_player_id"], strict=False).with_columns(
        [
            pl.col("Snap%").round(1),
            pl.col("Tgt Shr").round(1),
            pl.col("% Tm Att").round(1),
            pl.col("% Tm Opps").round(1),
        ]
    )
    csv_df.write_csv(csv_path)
    write_markdown(df, md_path, args.season, weeks, meta)

    print(f"Wrote {csv_path} ({meta.get('n_rows')} rows)", file=sys.stderr)
    print(f"Wrote {md_path}", file=sys.stderr)
    if meta.get("snap_without_gsis"):
        print("Snap rows without gsis_id:", file=sys.stderr)
        for row in meta["snap_without_gsis"]:
            print(f"  {row}", file=sys.stderr)

    # Print top 15 for CLI convenience
    top = csv_df.head(15)
    print("\nTop 15 by Opps:", file=sys.stderr)
    print(top, file=sys.stderr)
    print(f"META_JSON_KEYS season={meta['season']} snap_ok={meta.get('snap_ok')} n_rows={meta.get('n_rows')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
