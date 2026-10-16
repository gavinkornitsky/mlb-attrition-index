"""
    reliability check: is the part of P/PA not explained by K% and BB% a stable batter skill?
    reads pa_table.parquet and batter_baseline.csv. run from the project root.
"""

import pandas as pd
import statsmodels.formula.api as smf

pd.set_option("display.width", 200)

MIN_PA = 300

pa = pd.read_parquet("pa_table.parquet")
pa = pa.sort_values(["game_date", "game_pk", "at_bat_number"])
pa["half"] = pa.groupby("batter").cumcount() % 2
pa["k"] = pa["event"].isin(["strikeout", "strikeout_double_play"])
pa["bb"] = pa["event"] == "walk"


def spearman_brown(r):
    return 2 * r / (1 + r)


# ---------------------------------------------------------------------------
# 1. split-half: per-batter rates within each alternating half
# ---------------------------------------------------------------------------
halves = (
    pa.groupby(["batter", "half"])
    .agg(n=("n_pitches", "size"), ppa=("n_pitches", "mean"), k=("k", "mean"), bb=("bb", "mean"))
    .reset_index()
    .astype({"ppa": float, "k": float, "bb": float})
)

tot = halves.groupby("batter")["n"].sum()
halves = halves[halves["batter"].isin(tot[tot >= MIN_PA].index)].copy()

# residual within each half, using that half's own K% and BB%
half_fit = smf.ols("ppa ~ k + bb", data=halves).fit()
halves["resid"] = half_fit.resid

wide = halves.pivot(index="batter", columns="half", values=["ppa", "k", "bb", "resid"])

print(f"{len(wide)} batters with {MIN_PA}+ PA")
print(f"half-level R² (ppa ~ k + bb): {half_fit.rsquared:.3f}\n")
print(f"{'metric':>8} {'split-half r':>14} {'full-season':>13}")
for m in ["ppa", "k", "bb", "resid"]:
    r = wide[(m, 0)].corr(wide[(m, 1)])
    print(f"{m:>8} {r:>14.3f} {spearman_brown(r):>13.3f}")

# ---------------------------------------------------------------------------
# 2. full-season residual leaderboard: who forces pitches beyond K and BB?
# ---------------------------------------------------------------------------
full = (
    pa.groupby("batter")
    .agg(pa=("n_pitches", "size"), ppa=("n_pitches", "mean"), k=("k", "mean"), bb=("bb", "mean"))
    .reset_index()
)
full = full[full["pa"] >= MIN_PA].astype({"ppa": float, "k": float, "bb": float}).copy()
full_fit = smf.ols("ppa ~ k + bb", data=full).fit()
full["resid"] = full_fit.resid
full["batter"] = full["batter"].astype(int)

names = pd.read_csv("batter_baseline.csv", usecols=["batter", "name"])
full = full.merge(names, on="batter", how="left").sort_values("resid", ascending=False)

print("\nfull-season coefficients:")
print(full_fit.params.round(3))

cols = ["name", "pa", "ppa", "k", "bb", "resid"]
print("\nTop 10: most pitches beyond what K% and BB% predict")
print(full[cols].head(10).round(3).to_string(index=False))
print("\nBottom 10: fewest pitches beyond what K% and BB% predict")
print(full[cols].tail(10).round(3).to_string(index=False))

full.to_csv("batter_kbb_residual.csv", index=False)