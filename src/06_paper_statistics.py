"""Recompute the summary statistics quoted in the text of the paper.

Reads results/tables/ (produced by 01_run_pipeline.py) and writes
results/tables/paper_statistics.txt.
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
T = "results/tables"
r = pd.read_csv(f"{T}/results_all.csv")
dm = pd.read_csv(f"{T}/diebold_mariano.csv")
shift = pd.read_csv(f"{T}/covariate_shift.csv")
dp = pd.read_csv(f"{T}/daily_predictions.csv.gz", parse_dates=["date"])
out = []
w = out.append
key = ["City", "FeatureSet", "TestYear"]
sel = r[r.is_selected].copy()

w("== Mean of the selected model per predictor set and test year ==")
w(sel.groupby(["FeatureSet", "TestYear"])[["R2", "nRMSE_%", "MBE", "Skill_vs_Climatology", "Skill_vs_Persistence"]]
  .mean().round(3).to_string())

g = sel.pivot_table(index=["City", "TestYear"], columns="FeatureSet", values="Skill_vs_Climatology")
w(f"\nMedian skill gain F0->F1: {100 * (g.F1 - g.F0).median():.1f} pp; F1->F2: {100 * (g.F2 - g.F1).median():.1f} pp")
w("Skill gain F0->F1 (pp) by site and year:\n" + (100 * (g.F1 - g.F0)).round(1).unstack().to_string())

w("\n== Variance decomposition of skill (main effects, balanced design) ==")
x = r.assign(sk=100 * r.Skill_vs_Climatology)
tot = ((x.sk - x.sk.mean()) ** 2).sum()
for f in ["FeatureSet", "Model", "City", "TestYear"]:
    m = x.groupby(f).sk.transform("mean")
    w(f"eta2 {f}: {((m - x.sk.mean()) ** 2).sum() / tot:.3f}")

w("\n== Algorithms ==")
r["rank"] = r.groupby(key).RMSE.rank()
tab = pd.DataFrame({"mean_rank": r.groupby("Model")["rank"].mean(),
                    "test_optimal": r[r.is_oracle].Model.value_counts(),
                    "selected": r[r.is_selected].Model.value_counts(),
                    "last": r.loc[r.groupby(key).RMSE.idxmax()].Model.value_counts()}).fillna(0)
for fs in ["F0", "F1", "F2"]:
    tab[f"median_skill_{fs}_%"] = 100 * r[r.FeatureSet == fs].groupby("Model").Skill_vs_Climatology.median()
tab["beats_clim_raw"] = dm[dm.Reference == "clim"].groupby("Model").beats_5pct_raw.sum()
w(tab.round(2).to_string())
o = r[r.is_oracle].set_index(key).Skill_vs_Climatology
s_ = sel.set_index(key).Skill_vs_Climatology
loss = 100 * (o - s_)
w(f"Selected == test-optimal: {(sel.Model.values == sel.Oracle.values).sum()} / {len(sel)}; "
  f"skill loss median {loss.median():.2f} pp, max {loss.max():.2f} pp")
rho = [spearmanr(gg.CV_RMSE_train, gg.RMSE)[0] for _, gg in r.groupby(key)]
w(f"Median Spearman (CV RMSE vs test RMSE across models): {np.median(rho):.2f}")
w("Median test/CV RMSE ratio: " + str((r.RMSE / r.CV_RMSE_train).groupby(r.FeatureSet).median().round(3).to_dict()))
flex = r[r.Model.isin(["RF", "GBT", "SVM", "GAM"])].groupby(key).Skill_vs_Climatology.agg(lambda v: v.max() - v.min())
w("Median skill range among SVR/RF/GBT/GAM (pp): " + str((100 * flex.groupby("FeatureSet").median()).round(1).to_dict()))

w("\n== Diebold-Mariano, selected model, Holm 5% ==")
d = dm[dm.is_selected]
w(d.groupby(["Reference", "FeatureSet"]).beats_5pct_holm.sum().to_string())

w("\n== Bias and covariate shift (selected F1 model) ==")
f1 = sel[sel.FeatureSet == "F1"].merge(shift, on=["City", "Region", "TestYear"])
for v in ["d_RH2M", "d_DTR", "d_T2M", "d_ALLSKY_SFC_SW_DWN", "d_CLOUD_AMT", "d_PRECTOTCORR"]:
    w(f"Pearson r(MBE, {v}) = {np.corrcoef(f1[v], f1.MBE)[0, 1]:.2f}")
w(f"MBE > 0 in {(f1.MBE > 0).sum()} of {len(f1)} site-years; climatology MBE range "
  f"{sel.MBE_Climatology.min():.2f} to {sel.MBE_Climatology.max():.2f}")

w("\n== Error structure of the selected F1 model ==")
P = {k: gg.set_index("date")["value"] for k, gg in dp.groupby(["City", "FeatureSet", "Model", "TestYear"])}
for _, row in sel[sel.FeatureSet == "F1"].iterrows():
    obs = P[(row.City, "F1", "_obs", row.TestYear)]
    est = P[(row.City, "F1", row.Model, row.TestYear)]
    dark = obs < obs.quantile(0.10)
    w(f"{row.City:15s} {row.TestYear}  SD ratio {est.std() / obs.std():.2f}  "
      f"mean error darkest decile {(est[dark] - obs[dark]).mean():+.2f}")

open(f"{T}/paper_statistics.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
