from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)  # all paths below are relative to the repository root
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl
from matplotlib.patches import Rectangle
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "savefig.dpi": 600,
                     "axes.spines.top": False, "axes.spines.right": False})
O = "results/tables"
F = "results/figures"
os.makedirs(F, exist_ok=True)
res = pd.read_csv(f"{O}/results_all.csv")
dm = pd.read_csv(f"{O}/diebold_mariano.csv")
imp = pd.read_csv(f"{O}/grouped_permutation_importance_F1.csv")
shift = pd.read_csv(f"{O}/covariate_shift.csv")
CITY = ["Manaus", "Fortaleza", "Recife", "Brasilia", "Cuiaba", "Belo Horizonte",
        "Rio de Janeiro", "Sao Paulo", "Curitiba", "Porto Alegre"]
LAB = {"Brasilia": "Brasília", "Cuiaba": "Cuiabá", "Sao Paulo": "São Paulo"}
lab = lambda c: LAB.get(c, c)
MODELS = ["Linear", "CART", "SVM", "RF", "GBT", "GAM"]
MLAB = {"SVM": "SVR"}
FSN = {"F0": "F0: meteorological", "F1": "F1: + astronomical", "F2": "F2: + cloud amount (upper bound)"}
FCOL = {"F0": "#b8c2cc", "F1": "#3b7dd8", "F2": "#1b3f73"}
sel = res[res.is_selected]

# ---------- Fig 3: Spearman correlation, training period only ----------
cols = ["T2M", "DTR", "RH2M", "WS2M", "PS", "PRECTOTCORR", "TOA_SW_DWN", "CLOUD_AMT"]
M = {}
for c in CITY:
    d = pd.read_csv(f"data/raw/{c.replace(' ', '_')}.csv", index_col=0, parse_dates=True)
    d = d[d.index.year <= 2023]
    d["DTR"] = d.T2M_MAX - d.T2M_MIN
    M[lab(c)] = [d[[k, "ALLSKY_SFC_SW_DWN"]].corr("spearman").iloc[0, 1] for k in cols]
M = pd.DataFrame(M, index=cols).T
M.round(3).to_csv(f"{O}/spearman_train_2020_2023.csv")
fig, ax = plt.subplots(figsize=(3.45, 3.0))
im = ax.imshow(M.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=6)
ax.set_yticks(range(len(M))); ax.set_yticklabels(M.index, fontsize=6.5)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5,
                color="white" if abs(v) > 0.55 else "black")
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.ax.tick_params(labelsize=6)
cb.set_label("Spearman ρ", fontsize=6.5)
for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig(f"{F}/Fig2_correlation.png", bbox_inches="tight"); plt.close(fig)

# ---------- Fig 4: R2 heatmaps 2024 (and 2025 supplementary) ----------
for year in [2024, 2025]:
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.9), sharey=True)
    for ax, fs in zip(axes, ["F0", "F1", "F2"]):
        t = res[(res.FeatureSet == fs) & (res.TestYear == year)]
        pv = t.pivot(index="City", columns="Model", values="R2").reindex(index=CITY, columns=MODELS)
        im = ax.imshow(pv.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(6)); ax.set_xticklabels([MLAB.get(m, m) for m in MODELS], rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(10)); ax.set_yticklabels([lab(c) for c in CITY], fontsize=6.5)
        for i, c in enumerate(CITY):
            s = t[(t.City == c) & t.is_selected].Model.iloc[0]
            for j, m in enumerate(MODELS):
                v = pv.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.2,
                        color="black" if v > 0.6 else "white")
                if m == s:
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="#ff3b30", lw=1.1))
        ax.set_title(FSN[fs], fontsize=7)
        for sp in ax.spines.values(): sp.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.015); cb.ax.tick_params(labelsize=6)
    cb.set_label(f"R² (test year {year})", fontsize=6.5)
    fig.savefig((f"{F}/Fig4_R2_heatmap_2024.png" if year == 2024 else f"{F}/FigS1_R2_heatmap_2025.png"), bbox_inches="tight"); plt.close(fig)

# ---------- Fig 5: skill vs climatology, selected model, 2024 and 2025 ----------
dsel = dm[(dm.is_selected) & (dm.Reference == "clim")]
fig, axes = plt.subplots(2, 1, figsize=(7.16, 3.9), sharex=True)
for ax, year in zip(axes, [2024, 2025]):
    x = np.arange(len(CITY)); w = 0.26
    for k, fs in enumerate(["F0", "F1", "F2"]):
        t = sel[(sel.FeatureSet == fs) & (sel.TestYear == year)].set_index("City").reindex(CITY)
        sg = dsel[(dsel.FeatureSet == fs) & (dsel.TestYear == year)].set_index("City").reindex(CITY).beats_5pct_holm.values
        bars = ax.bar(x + (k - 1) * w, 100 * t.Skill_vs_Climatology.values, w, color=FCOL[fs], label=FSN[fs], ec="none")
        for b, ok in zip(bars, sg):
            if not ok:
                b.set_hatch("//////"); b.set_edgecolor("white"); b.set_alpha(0.55)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel(f"Skill vs. climatology (%)\n{year}", fontsize=6.5)
    ax.set_ylim(-5, 70); ax.yaxis.grid(True, lw=0.3, color="#cccccc"); ax.set_axisbelow(True)
axes[1].set_xticks(x); axes[1].set_xticklabels([lab(c) for c in CITY], rotation=30, ha="right")
h, l = axes[0].get_legend_handles_labels()
h.append(Rectangle((0, 0), 1, 1, fc="#9aa5b1", ec="white", hatch="//////", alpha=0.55)); l.append("not significant (DM, Holm 5%)")
axes[0].legend(h, l, fontsize=5.8, ncol=4, loc="upper left", frameon=False, bbox_to_anchor=(0, 1.22))
fig.tight_layout(); fig.savefig(f"{F}/Fig5_skill.png", bbox_inches="tight"); plt.close(fig)

# ---------- Fig 6: time series F1 2024, selected model ----------
_dp = pd.read_csv(f"{O}/daily_predictions.csv.gz", parse_dates=["date"])
P = {k: g.set_index("date")["value"] for k, g in _dp.groupby(["City", "FeatureSet", "Model", "TestYear"])}
for year in [2024, 2025]:
    fig, axes = plt.subplots(5, 2, figsize=(7.16, 7.6), sharex=True)
    for ax, c in zip(axes.T.ravel(), CITY):
        r = sel[(sel.City == c) & (sel.FeatureSet == "F1") & (sel.TestYear == year)].iloc[0]
        obs = P[(c, "F1", "_obs", year)]
        ax.plot(obs.index, obs.values, color="#222222", lw=0.55, label="NASA/POWER")
        ax.plot(obs.index, P[(c, "F1", r.Model, year)].values, color="#d8573b", lw=0.6, label="F1 model")
        ax.plot(obs.index, P[(c, "F1", "_clim", year)].values, color="#3b7dd8", lw=0.8, ls="--", label="Climatology")
        ax.set_title(f"{lab(c)}: {MLAB.get(r.Model, r.Model)}, R² = {r.R2:.2f}, MBE = {r.MBE:+.2f}", fontsize=6.5)
        ax.set_ylim(0, 9); ax.tick_params(labelsize=5.5)
    for ax in axes[:, 0]: ax.set_ylabel("kWh m$^{-2}$ day$^{-1}$", fontsize=6)
    for ax in axes[-1]:
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%b")); ax.xaxis.set_major_locator(mpl.dates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, fontsize=6.5, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig((f"{F}/Fig6_timeseries_F1_2024.png" if year == 2024 else f"{F}/FigS2_timeseries_F1_2025.png"), bbox_inches="tight"); plt.close(fig)

# ---------- Fig 7: grouped permutation importance (RF, F1), mean of 2024-2025 ----------
g = imp[imp.Which == "RF"].groupby(["City", "Group"]).dRMSE_mean.mean().unstack()
GORD = ["Precipitation", "Relative humidity", "Diurnal temperature range", "Temperature (T2M)",
        "Astronomical (TOA, DOY)", "Wind and pressure"]
g = g.reindex(index=CITY, columns=GORD).clip(lower=0)
share = g.div(g.sum(axis=1), axis=0)
share.round(4).to_csv(f"{O}/grouped_importance_share_RF_F1.csv")
fig, ax = plt.subplots(figsize=(3.45, 3.0))
im = ax.imshow(100 * share.values, cmap="YlOrBr", vmin=0, vmax=60, aspect="auto")
ax.set_xticks(range(len(GORD)))
ax.set_xticklabels(["Precipitation", "Rel. humidity", "DTR", "T2M", "Astronomical", "Wind + pressure"], rotation=40, ha="right", fontsize=6)
ax.set_yticks(range(10)); ax.set_yticklabels([lab(c) for c in CITY], fontsize=6.5)
for i in range(10):
    for j in range(len(GORD)):
        v = 100 * share.values[i, j]
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.5, color="white" if v > 40 else "black")
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.ax.tick_params(labelsize=6)
cb.set_label("Share of RMSE increase (%)", fontsize=6.5)
for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig(f"{F}/Fig8_importance.png", bbox_inches="tight"); plt.close(fig)

# ---------- Fig 8: covariate shift vs bias ----------
s1 = sel[sel.FeatureSet == "F1"][["City", "TestYear", "MBE", "MBE_Climatology"]].merge(shift, on=["City", "TestYear"])
s1.to_csv(f"{O}/shift_vs_bias_F1.csv", index=False, float_format="%.4f")
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6))
for ax, v, xl in zip(axes, ["d_RH2M", "d_DTR"], ["Test − training mean RH2M (percentage points)", "Test − training mean DTR (°C)"]):
    for y, mk in [(2024, "o"), (2025, "s")]:
        t = s1[s1.TestYear == y]
        ax.scatter(t[v], t.MBE, marker=mk, s=16, c="#d8573b" if y == 2024 else "#3b7dd8", label=str(y), zorder=3)
        for _, r in t.iterrows():
            ax.annotate(lab(r.City), (r[v], r.MBE), fontsize=4.6, xytext=(2, 2), textcoords="offset points")
    rr = np.corrcoef(s1[v], s1.MBE)[0, 1]
    b = np.polyfit(s1[v], s1.MBE, 1); xx = np.linspace(s1[v].min(), s1[v].max(), 10)
    ax.plot(xx, np.polyval(b, xx), color="#555555", lw=0.7, ls="--")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel(xl, fontsize=6.5); ax.set_ylabel("MBE of F1 model (kWh m$^{-2}$ day$^{-1}$)", fontsize=6.5)
    ax.text(0.98, 0.97, f"Pearson r = {rr:.2f} (n = 20)", transform=ax.transAxes, ha="right", va="top", fontsize=6)
axes[0].legend(fontsize=6, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(f"{F}/Fig7_shift_bias.png", bbox_inches="tight"); plt.close(fig)
print("figs ok")
