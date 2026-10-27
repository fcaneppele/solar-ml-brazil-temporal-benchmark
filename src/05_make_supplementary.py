"""Build the supplementary workbook (Tables S1-S5) from results/tables."""
from pathlib import Path
import os
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
T = "results/tables"
os.makedirs("results/supplementary", exist_ok=True)
ren = {"SVM": "SVR"}
r = pd.read_csv(f"{T}/results_all.csv").replace({"Model": ren, "Selected": ren, "Oracle": ren})
dm = pd.read_csv(f"{T}/diebold_mariano.csv").replace({"Model": ren})
imp = pd.read_csv(f"{T}/grouped_permutation_importance_F1.csv").replace({"Model": ren})
sh = pd.read_csv(f"{T}/covariate_shift.csv")
sp = pd.read_csv(f"{T}/spearman_train_2020_2023.csv")
readme = pd.DataFrame({
    "Sheet": ["S1_all_results", "S2_DM_tests", "S3_grouped_importance", "S4_covariate_shift", "S5_spearman_train"],
    "Content": [
        "Test-year metrics of all 360 runs (10 sites x 3 predictor sets x 6 models x 2 test years). CV_RMSE_train: time-series cross-validated RMSE on 2020-2023; is_selected: ex-ante selected model; is_oracle: test-optimal model. Energy in kWh m-2 day-1; skill as fraction.",
        "Diebold-Mariano tests (squared loss, Newey-West HAC variance, lag 5, HLN factor, t(n-1), two-sided) of every model vs climatology and vs persistence; p_holm_selected: Holm-adjusted p-value for the selected models (60 tests per reference).",
        "Grouped permutation importance (20 repeats) of RF and of the selected F1 model, per site and test year: mean and SD of the RMSE increase.",
        "Test-year minus training-period (2020-2023) mean of the target and predictors, per site and test year.",
        "Spearman correlation between predictors and daily solar energy, training period 2020-2023 (Fig. 2)."]})
with pd.ExcelWriter("results/supplementary/Supplementary_Tables_S1-S5.xlsx") as xw:
    for name, df in [("README", readme), ("S1_all_results", r), ("S2_DM_tests", dm), ("S3_grouped_importance", imp),
                     ("S4_covariate_shift", sh), ("S5_spearman_train", sp)]:
        df.to_excel(xw, sheet_name=name, index=False)
print("supplementary ok")
