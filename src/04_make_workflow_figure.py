from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)  # all paths below are relative to the repository root
os.makedirs("results/figures", exist_ok=True)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 600})
fig, ax = plt.subplots(figsize=(7.16, 2.75))
ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
C = {"data": "#e8eef7", "feat": "#eaf4ea", "model": "#fdf1e0", "eval": "#f3e8f3"}


def box(x, y, w, h, title, body, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=c, ec="#444444", lw=0.7))
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=7, weight="bold")
    ax.text(x + w / 2, y + h - 5.2, body, ha="center", va="top", fontsize=5.6, linespacing=1.35)


def arrow(x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#333333"))


box(0.5, 6, 17, 31, "NASA/POWER",
    "Daily point data, RE community\n10 cities, 2020–2025\n\nTarget: ALLSKY_SFC_SW_DWN\n(kWh m$^{-2}$ day$^{-1}$)\n\nPredictors: T2M, T2M_MAX,\nT2M_MIN, RH2M, WS2M, PS,\nPRECTOTCORR, TOA_SW_DWN,\nCLOUD_AMT", C["data"])
box(21.5, 6, 18, 31, "Predictor sets",
    "F0  meteorological\nT2M, RH2M, WS2M, PS,\nPRECTOTCORR\n\nF1  F0 + astronomical\n+ TOA_SW_DWN, DTR,\nsin/cos(day of year)\n\nF2  F1 + CLOUD_AMT\n(upper-bound scenario)", C["feat"])
box(43.5, 21.5, 23, 15.5, "Training 2020–2023",
    "Linear, CART, SVR, RF, GBT, GAM\nGrid search with TimeSeriesSplit\n(4 folds); ex-ante model selection\nby cross-validated RMSE", C["model"])
box(43.5, 6, 23, 12.5, "Independent test years",
    "2024 (n = 366)   2025 (n = 365)\nno refitting, no tuning\non test data", C["model"])
box(70.5, 6, 29, 31, "Evaluation",
    "MAE, RMSE, nRMSE, MBE, R$^2$\n\nReferences:\nsmoothed climatology (training years)\npersistence (previous day)\n\nSkill = 1 − RMSE / RMSE$_{ref}$\n\nDiebold–Mariano test\n(HAC variance, HLN, Holm)\n\nGrouped permutation importance\nCovariate-shift diagnostics", C["eval"])
arrow(18.6, 21.5, 20.9, 21.5)
arrow(40.6, 29, 42.9, 29)
arrow(40.6, 12, 42.9, 12)
arrow(55, 21.0, 55, 19.0)
arrow(67.6, 12, 69.9, 12)
arrow(67.6, 29, 69.9, 29)
ax.text(50, 1.2, "Strict chronological separation: no information from 2024–2025 enters training, tuning or model selection.",
        ha="center", fontsize=5.8, style="italic", color="#333333")
fig.savefig("results/figures/Fig3_workflow.png", bbox_inches="tight")
fig.savefig("results/figures/Fig3_workflow.pdf", bbox_inches="tight")
print("ok")
