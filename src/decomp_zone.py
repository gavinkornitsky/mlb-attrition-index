"""
    zone-level decomposition of P/PA.

    takes are split by zone: good takes (outside the zone) vs. passive takes (in the zone),
    plus the two umpire-call buckets. whiffs and fouls (the only non-terminal contact swings)
    are split by zone and count. components still sum exactly to P/PA.

    reads statcast_raw.parquet, pa_table.parquet, batter_baseline.csv. run from the project root.
"""

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

MIN_PA = 300
KEYS = ["game_pk", "at_bat_number"]

BALL = {"ball", "blocked_ball", "automatic_ball", "pitchout", "intent_ball"}
CALLED = {"called_strike", "automatic_strike"}
WHIFF = {"swinging_strike", "swinging_strike_blocked", "missed_bunt", "foul_tip", "bunt_foul_tip"}
FOUL = {"foul", "foul_bunt"}
MISS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
SWING = WHIFF | FOUL | {"hit_into_play"}
K_EVENTS = {"strikeout", "strikeout_double_play"}

# component -> (group, label)
COMP = {
    "take_out_ball":   ("good take",         "take, outside zone, called ball"),
    "take_in_strike":  ("passive take",      "take, in zone, called strike"),
    "take_in_ball":    ("umpire call",       "take, in zone, called ball"),
    "take_out_strike": ("umpire call",       "take, outside zone, called strike"),
    "whiff_in":        ("whiff",             "whiff, in zone"),
    "whiff_out":       ("whiff",             "whiff, outside zone"),
    "foul_in_early":   ("foul in zone",      "foul, in zone, 0-1 strikes"),
    "foul_in_2k":      ("foul in zone",      "foul, in zone, 2 strikes"),
    "foul_out_early":  ("foul outside zone", "foul, outside zone, 0-1 strikes"),
    "foul_out_2k":     ("foul outside zone", "foul, outside zone, 2 strikes"),
    "no_zone":         ("no zone data",      "non-terminal, no zone data"),
}
C = list(COMP)

# ---------------------------------------------------------------------------
# 1. load pitches for cleaned PAs only
# ---------------------------------------------------------------------------
pa = pd.read_parquet("pa_table.parquet")
pa["batter"] = pa["batter"].astype(int)
pa = pa.sort_values(["game_date", "game_pk", "at_bat_number"])
pa["half"] = pa.groupby("batter").cumcount() % 2

raw = pd.read_parquet(
    "statcast_raw.parquet",
    columns=KEYS + ["pitch_number", "description", "strikes", "zone"],
)
p = raw.merge(pa[KEYS + ["batter", "half"]], on=KEYS, how="inner")
p = p.sort_values(KEYS + ["pitch_number"])

# ---------------------------------------------------------------------------
# 2. classify every pitch by action x zone x count
# ---------------------------------------------------------------------------
d = p["description"].astype(str)
zone = p["zone"].fillna(0)
zin, zout = zone.between(1, 9), zone.between(11, 14)
znone = ~(zin | zout)
nt = ~(p["pitch_number"] == p.groupby(KEYS)["pitch_number"].transform("max"))
two_k = p["strikes"] == 2

is_ball, is_called = d.isin(BALL), d.isin(CALLED)
is_whiff, is_foul = d.isin(WHIFF), d.isin(FOUL)
is_take = is_ball | is_called
is_swing, is_miss = d.isin(SWING), d.isin(MISS)

p["take_out_ball"] = nt & zout & is_ball
p["take_in_strike"] = nt & zin & is_called
p["take_in_ball"] = nt & zin & is_ball
p["take_out_strike"] = nt & zout & is_called
p["whiff_in"] = nt & zin & is_whiff
p["whiff_out"] = nt & zout & is_whiff
p["foul_in_early"] = nt & zin & is_foul & ~two_k
p["foul_in_2k"] = nt & zin & is_foul & two_k
p["foul_out_early"] = nt & zout & is_foul & ~two_k
p["foul_out_2k"] = nt & zout & is_foul & two_k
p["no_zone"] = nt & znone

# rate flags (all pitches, terminal included)
p["zin"], p["zout"] = zin, zout
p["swing_in"], p["swing_out"] = is_swing & zin, is_swing & zout
p["miss_in"], p["miss_out"] = is_miss & zin, is_miss & zout
p["foul_any"] = is_foul
p["inplay"] = d == "hit_into_play"
p["swing_2k"] = is_swing & two_k
p["foul_2k_any"] = is_foul & two_k
p["take_in"], p["take_out"] = is_take & zin, is_take & zout
p["called_out"] = is_called & zout
p["ball_in"] = is_ball & zin
FLAGS = ["zin", "zout", "swing_in", "swing_out", "miss_in", "miss_out", "foul_any", "inplay",
         "swing_2k", "foul_2k_any", "take_in", "take_out", "called_out", "ball_in"]


def rates(s):
    return pd.DataFrame({
        "z_contact": 1 - s["miss_in"] / s["swing_in"],
        "o_contact": 1 - s["miss_out"] / s["swing_out"],
        "foul_share_of_contact": s["foul_any"] / (s["foul_any"] + s["inplay"]),
        "two_k_foul_rate": s["foul_2k_any"] / s["swing_2k"],
        "good_take_rate": s["take_out"] / s["zout"],
        "passive_take_rate": s["take_in"] / s["zin"],
        "ump_strike_outside": s["called_out"] / s["take_out"],
        "ump_ball_inside": s["ball_in"] / s["take_in"],
    })


# ---------------------------------------------------------------------------
# 3. batter-level components and rates, full season and split halves
# ---------------------------------------------------------------------------
per_pa = p.groupby(KEYS + ["batter", "half"])[C].sum().reset_index()
per_pa["ppa"] = per_pa[C].sum(axis=1) + 1
per_pa = per_pa.merge(pa[KEYS + ["event"]], on=KEYS)
per_pa["k"] = per_pa["event"].isin(K_EVENTS)
per_pa["bb"] = per_pa["event"] == "walk"
VALS = C + ["ppa", "k", "bb"]


def batter_table(by):
    comps = per_pa.groupby(by)[VALS].mean()
    comps["pa"] = per_pa.groupby(by).size()
    r = rates(p.groupby(by)[FLAGS].sum().astype(float))
    return comps.join(r).astype(float)


full = batter_table("batter")
full = full[full["pa"] >= MIN_PA].copy()

halves = batter_table(["batter", "half"])
halves = halves[halves.index.get_level_values("batter").isin(full.index)]
h0, h1 = halves.xs(0, level="half"), halves.xs(1, level="half")

chk = (per_pa.merge(pa[KEYS + ["n_pitches"]], on=KEYS).eval("ppa - n_pitches").abs().max())
print(f"PAs: {len(pa):,} | pitches: {len(p):,} | max component-sum error: {chk}")

X = np.column_stack([np.ones(len(full)), full["k"], full["bb"]])
beta, *_ = np.linalg.lstsq(X, full["ppa"].to_numpy(), rcond=None)
full["resid"] = full["ppa"] - X @ beta
print(f"K/BB model: P/PA = {beta[0]:.3f} + {beta[1]:.3f}*K% + {beta[2]:.3f}*BB%")

# ---------------------------------------------------------------------------
# 4. component table and grouped shares
# ---------------------------------------------------------------------------
lg_comp = p[C].sum().astype(float) / len(pa)
var_ppa, var_res = full["ppa"].var(), full["resid"].var()
dec = pd.DataFrame([
    {
        "group": COMP[c][0],
        "component": COMP[c][1],
        "league_per_pa": lg_comp[c],
        "batter_std": full[c].std(),
        "split_half_r": h0[c].corr(h1[c]),
        "share_ppa_var": full[c].cov(full["ppa"]) / var_ppa,
        "share_resid_var": full[c].cov(full["resid"]) / var_res,
    }
    for c in C
])
print(f"\nZone decomposition ({len(full)} batters, {MIN_PA}+ PA)")
print(dec.drop(columns="group").round(3).to_string(index=False))
print(f"terminal pitch: 1.000 | league total: {lg_comp.sum() + 1:.3f}")

grp = dec.groupby("group", sort=False)[["league_per_pa", "share_ppa_var", "share_resid_var"]].sum()
print("\nGrouped shares")
print(grp.round(3))
print(f"sums: ppa {grp['share_ppa_var'].sum():.3f} | resid {grp['share_resid_var'].sum():.3f}")

# ---------------------------------------------------------------------------
# 5. take and contact rates
# ---------------------------------------------------------------------------
rate_cols = ["good_take_rate", "passive_take_rate", "ump_strike_outside", "ump_ball_inside",
             "z_contact", "o_contact", "foul_share_of_contact", "two_k_foul_rate"]
rt = pd.DataFrame({
    "qualified_mean": [full[c].mean() for c in rate_cols],
    "corr_ppa": [full[c].corr(full["ppa"]) for c in rate_cols],
    "corr_resid": [full[c].corr(full["resid"]) for c in rate_cols],
    "corr_bb": [full[c].corr(full["bb"]) for c in rate_cols],
    "split_half_r": [h0[c].corr(h1[c]) for c in rate_cols],
}, index=rate_cols)
print("\nTake and contact rates")
print(rt.round(3))

# ---------------------------------------------------------------------------
# 6. leaderboards
# ---------------------------------------------------------------------------
names = pd.read_csv("batter_baseline.csv", usecols=["batter", "name"]).set_index("batter")
full = full.join(names)


def board(col, title):
    print(f"\nTop 10: {title}")
    cols = ["name", "pa", "ppa", col, "resid"]
    print(full.sort_values(col, ascending=False)[cols].head(10).round(3).to_string(index=False))


board("take_out_ball", "good takes per PA (balls taken outside the zone)")
board("take_in_strike", "passive takes per PA (called strikes in the zone)")
board("foul_out_2k", "two-strike fouls outside the zone per PA (spoiling chase pitches)")
board("o_contact", "contact rate on swings outside the zone")

full.reset_index().to_csv("batter_zone_decomposition.csv", index=False)