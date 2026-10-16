"""
    decompose P/PA into additive pitch components and find what drives attrition beyond K% and BB%.

    every PA = 1 terminal pitch + non-terminal balls + called strikes + whiffs + fouls,
    so the components add up exactly to P/PA and their variance shares sum to 1.

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
SWING = WHIFF | FOUL | {"hit_into_play"}
MISS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
K_EVENTS = {"strikeout", "strikeout_double_play"}
NON_BIP = K_EVENTS | {"walk", "hit_by_pitch", "catcher_interf"}

COMP = {
    "balls": "balls taken",
    "called": "called strikes",
    "whiffs": "whiffs / foul tips",
    "fouls_early": "fouls, 0-1 strikes",
    "fouls_2k": "fouls, 2 strikes",
    "other": "unclassified",
}

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

rows = p.groupby(KEYS).size().rename("rows").reset_index()
chk = pa.merge(rows, on=KEYS)
print(f"PAs: {len(pa):,} | pitches: {len(p):,}")
print(f"PAs where pitch rows != max pitch_number: {(chk['rows'] != chk['n_pitches']).sum():,}")

# ---------------------------------------------------------------------------
# 2. classify every pitch
# ---------------------------------------------------------------------------
d = p["description"].astype(str)
p["terminal"] = p["pitch_number"] == p.groupby(KEYS)["pitch_number"].transform("max")
nt = ~p["terminal"]
two_k = p["strikes"] == 2

p["balls"] = nt & d.isin(BALL)
p["called"] = nt & d.isin(CALLED)
p["whiffs"] = nt & d.isin(WHIFF)
p["fouls_early"] = nt & d.isin(FOUL) & ~two_k
p["fouls_2k"] = nt & d.isin(FOUL) & two_k
p["other"] = nt & ~d.isin(BALL | CALLED | WHIFF | FOUL)

zone = p["zone"].fillna(0)
p["swing"] = d.isin(SWING)
p["miss"] = d.isin(MISS)
p["in_zone"] = zone.between(1, 9)
p["out_zone"] = zone.between(11, 14)
p["chase"] = p["swing"] & p["out_zone"]
p["z_swing"] = p["swing"] & p["in_zone"]
p["first"] = p["pitch_number"] == 1
p["fp_swing"] = p["swing"] & p["first"]
p["swing_2k"] = p["swing"] & two_k
p["foul_2k_any"] = d.isin(FOUL) & two_k

print("\nunclassified descriptions:", sorted(d[p["other"]].unique()))

# ---------------------------------------------------------------------------
# 3. league reference: pitches per PA by outcome, and league decomposition
# ---------------------------------------------------------------------------
pa["outcome"] = np.select(
    [pa["event"].isin(K_EVENTS), pa["event"] == "walk", pa["event"] == "hit_by_pitch"],
    ["strikeout", "walk", "hbp"],
    default="in play / other",
)
lg = pa.groupby("outcome")["n_pitches"].agg(n="size", pitches_per_pa="mean")
lg["share_of_pa"] = lg["n"] / len(pa)
print("\nLeague pitches per PA by outcome")
print(lg[["share_of_pa", "pitches_per_pa"]].astype(float).round(3))

lg_comp = p[list(COMP)].sum().astype(float) / len(pa)
print("\nLeague decomposition (per PA)")
print(lg_comp.round(3).to_string())
print(f"terminal pitch: 1.000\ntotal: {lg_comp.sum() + 1:.3f}")

# ---------------------------------------------------------------------------
# 4. batter-level components, full season and split halves
# ---------------------------------------------------------------------------
per_pa = p.groupby(KEYS + ["batter", "half"])[list(COMP)].sum().reset_index()
per_pa["ppa"] = per_pa[list(COMP)].sum(axis=1) + 1
per_pa = per_pa.merge(pa[KEYS + ["event"]], on=KEYS)
per_pa["k"] = per_pa["event"].isin(K_EVENTS)
per_pa["bb"] = per_pa["event"] == "walk"
per_pa["bip_pitches"] = per_pa["ppa"].where(~per_pa["event"].isin(NON_BIP))

VALS = list(COMP) + ["ppa", "k", "bb", "bip_pitches"]


def batter_rates(df, by):
    out = df.groupby(by)[VALS].mean()
    out["pa"] = df.groupby(by).size()
    return out.astype(float)



full = batter_rates(per_pa, "batter")
full = full[full["pa"] >= MIN_PA].copy()

halves = batter_rates(per_pa, ["batter", "half"])
halves = halves[halves.index.get_level_values("batter").isin(full.index)]
h0, h1 = halves.xs(0, level="half"), halves.xs(1, level="half")

# plate discipline rates
flags = ["swing", "miss", "in_zone", "out_zone", "chase", "z_swing",
         "first", "fp_swing", "swing_2k", "foul_2k_any"]
s = p.groupby("batter")[flags].sum().astype(float)
n = p.groupby("batter").size().astype(float)
disc = pd.DataFrame({
    "swing_rate": s["swing"] / n,
    "zone_rate": s["in_zone"] / n,
    "chase_rate": s["chase"] / s["out_zone"],
    "zone_swing_rate": s["z_swing"] / s["in_zone"],
    "contact_rate": 1 - s["miss"] / s["swing"],
    "first_pitch_swing": s["fp_swing"] / s["first"],
    "two_k_foul_rate": s["foul_2k_any"] / s["swing_2k"],
})
full = full.join(disc)

# K/BB residual (same model as reliability.py)
X = np.column_stack([np.ones(len(full)), full["k"], full["bb"]])
beta, *_ = np.linalg.lstsq(X, full["ppa"].to_numpy(), rcond=None)
full["resid"] = full["ppa"] - X @ beta
print(f"\nK/BB model: P/PA = {beta[0]:.3f} + {beta[1]:.3f}*K% + {beta[2]:.3f}*BB%")

# ---------------------------------------------------------------------------
# 5. decomposition table: what drives P/PA and the residual?
# ---------------------------------------------------------------------------
var_ppa, var_res = full["ppa"].var(), full["resid"].var()
dec = pd.DataFrame([
    {
        "component": label,
        "league_per_pa": lg_comp[c],
        "batter_std": full[c].std(),
        "split_half_r": h0[c].corr(h1[c]),
        "share_ppa_var": full[c].cov(full["ppa"]) / var_ppa,
        "share_resid_var": full[c].cov(full["resid"]) / var_res,
    }
    for c, label in COMP.items()
])
print(f"\nDecomposition ({len(full)} batters, {MIN_PA}+ PA)")
print(dec.round(3).to_string(index=False))
print(f"shares sum to: ppa {dec['share_ppa_var'].sum():.3f} | resid {dec['share_resid_var'].sum():.3f}")

# ---------------------------------------------------------------------------
# 6. plate discipline: which approach traits go with attrition?
# ---------------------------------------------------------------------------
drivers = ["bip_pitches"] + list(disc.columns)
corr = pd.DataFrame({
    "qualified_mean": [full[c].mean() for c in drivers],
    "corr_ppa": [full[c].corr(full["ppa"]) for c in drivers],
    "corr_resid": [full[c].corr(full["resid"]) for c in drivers],
    "corr_bb": [full[c].corr(full["bb"]) for c in drivers],
    "split_half_r": [h0[c].corr(h1[c]) if c in h0 else np.nan for c in drivers],
}, index=drivers)
print("\nPlate discipline correlations")
print(corr.round(3))

# ---------------------------------------------------------------------------
# 7. leaderboards
# ---------------------------------------------------------------------------
names = pd.read_csv("batter_baseline.csv", usecols=["batter", "name"]).set_index("batter")
full = full.join(names)

cols = ["name", "pa", "ppa", "fouls_2k", "two_k_foul_rate", "bip_pitches", "resid"]
print("\nTop 10: two-strike fouls per PA")
print(full.sort_values("fouls_2k", ascending=False)[cols].head(10).round(3).to_string(index=False))
print("\nTop 10: pitches per ball-in-play PA")
print(full.sort_values("bip_pitches", ascending=False)[cols].head(10).round(3).to_string(index=False))

full.reset_index().to_csv("batter_decomposition.csv", index=False)