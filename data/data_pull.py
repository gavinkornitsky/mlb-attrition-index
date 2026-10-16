"""
    pull Statcast pitch-level data and collapse it to one row per plate appearance.
    require: pip install pybaseball pandas pyarrow
"""

# imports
import pandas as pd
from pybaseball import statcast, cache

cache.enable()

START, END = "2025-03-27", "2025-09-28"  # 2025 mlb season

# PA outcome events
OUT_EVENTS = {
    "field_out", "strikeout", "grounded_into_double_play", "force_out",
    "double_play", "fielders_choice_out", "strikeout_double_play",
    "triple_play", "sac_fly", "sac_bunt", "sac_fly_double_play",
    "sac_bunt_double_play",
}

# events that don't reflect a complete, batter-driven PA
DROP_EVENTS = {"truncated_pa", "intent_walk", "None"}

# ---------------------------------------------------------------------------
# 1. pull raw pitches from Statcast. slow the first time, cached after
# ---------------------------------------------------------------------------
pitches = statcast(start_dt=START, end_dt=END)
pitches = pitches[pitches["game_type"] == "R"].copy()
pitches.to_parquet("statcast_raw.parquet")

# ---------------------------------------------------------------------------
# 2. collapse to one row per PA
# ---------------------------------------------------------------------------
pitches = pitches.sort_values(["game_pk", "at_bat_number", "pitch_number"])
for base in ("on_1b", "on_2b", "on_3b"):
    pitches[base] = pitches[base].notna()

pa = (
    pitches.groupby(["game_pk", "at_bat_number"])
    .agg(
        game_date=("game_date", "first"),
        batter=("batter", "first"),
        pitcher=("pitcher", "first"),
        stand=("stand", "first"),
        p_throws=("p_throws", "first"),
        inning=("inning", "first"),
        outs=("outs_when_up", "first"),
        on_1b=("on_1b", "first"),
        on_2b=("on_2b", "first"),
        on_3b=("on_3b", "first"),
        n_pitches=("pitch_number", "max"),
        event=("events", "last"),  # events is only populated on the final pitch
    )
    .reset_index()
)
print(f"{len(pitches):,} pitches -> {len(pa):,} plate appearances")

# 3. event audit, then clean
print(pa["event"].value_counts(dropna=False))

before = len(pa)
pa = pa[pa["event"].notna() & ~pa["event"].isin(DROP_EVENTS)].copy()
print(f"Cleaned: {before:,} -> {len(pa):,} PAs")

pa["is_out"] = pa["event"].isin(OUT_EVENTS)

# ---------------------------------------------------------------------------
# 4. summary stats (cleaned)
# ---------------------------------------------------------------------------
print(pa["n_pitches"].describe())
print(pa.groupby("is_out")["n_pitches"].mean())

pa.to_parquet("pa_table.parquet")

# ---------------------------------------------------------------------------
# 5. split-half reliability of raw P/PA (cleaned)
# ---------------------------------------------------------------------------
pa = pa.sort_values(["game_date", "game_pk", "at_bat_number"])
pa["half"] = pa.groupby("batter").cumcount() % 2

b = pa.groupby(["batter", "half"])["n_pitches"].agg(["mean", "size"]).unstack()
b = b[b[("size", 0)] + b[("size", 1)] >= 300]
print(len(b), "batters")
print("split-half r:", b[("mean", 0)].corr(b[("mean", 1)]))