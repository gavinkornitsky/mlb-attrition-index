"""
    naive baseline: per-batter pitches per PA vs. league average.
    reads pa_table.parquet from data_pull.py. run from the project root.
"""

import pandas as pd
from pybaseball import playerid_reverse_lookup

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

MIN_PA = 300

pa = pd.read_parquet("pa_table.parquet")
lg_ppa = pa["n_pitches"].mean()
print(f"league P/PA: {lg_ppa:.3f}")

# ---------------------------------------------------------------------------
# per-batter totals and rates
# ---------------------------------------------------------------------------
bat = (
    pa.groupby("batter")
    .agg(pa=("n_pitches", "size"), pitches=("n_pitches", "sum"))
    .reset_index()
)
bat["batter"] = bat["batter"].astype(int)
bat["ppa"] = bat["pitches"] / bat["pa"]
bat["ppa_plus"] = 100 * bat["ppa"] / lg_ppa             # 100 = league average
bat["pitches_above_avg"] = bat["pitches"] - lg_ppa * bat["pa"]  # extra pitches forced over the season

bat = bat[bat["pa"] >= MIN_PA].copy()

# ---------------------------------------------------------------------------
# attach names (MLBAM ids)
# ---------------------------------------------------------------------------
names = playerid_reverse_lookup(bat["batter"].tolist(), key_type="mlbam")
names["name"] = names["name_first"].str.title() + " " + names["name_last"].str.title()
bat = bat.merge(
    names[["key_mlbam", "name"]], left_on="batter", right_on="key_mlbam", how="left"
).drop(columns="key_mlbam")

# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
cols = ["name", "pa", "ppa", "ppa_plus", "pitches_above_avg"]
bat = bat.sort_values("ppa", ascending=False)

print(f"\n{len(bat)} batters with {MIN_PA}+ PA")
print(bat["ppa"].describe())

print("\nTop 15 P/PA")
print(bat[cols].head(15).round(2).to_string(index=False))

print("\nBottom 15 P/PA")
print(bat[cols].tail(15).round(2).to_string(index=False))

bat.to_csv("batter_baseline.csv", index=False)

import statsmodels.formula.api as smf

pa["k"] = pa["event"].isin(["strikeout", "strikeout_double_play"])
pa["bb"] = pa["event"] == "walk"
rates = pa.groupby("batter")[["k", "bb"]].mean().reset_index()
rates["batter"] = rates["batter"].astype(int)

bat = bat.merge(rates, on="batter").astype({"ppa": float, "k": float, "bb": float})
print(bat[["ppa", "k", "bb"]].corr().round(3))
print("R² (ppa ~ k + bb):", round(smf.ols("ppa ~ k + bb", data=bat).fit().rsquared, 3))