# Predictor information content vs. algorithm choice in ML estimation of daily solar energy — Brazil, 10 cities

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

Code, data and results accompanying the manuscript

> F. de L. Caneppele, A. Dal Pai, C. P. Tieghi, C. E. C. Nogueira, J. A. C. Siqueira, M. M. Frigo, M. V. C. Calça, D. C. Malagueta, and E. Z. Godinho, "Predictor Information Content Versus Algorithm Choice in Machine Learning Estimation of Daily Solar Energy: A Temporally Validated Benchmark for Ten Brazilian Cities," submitted to *IEEE Access*, 2026.

The study benchmarks six regression models across three nested predictor sets for daily all-sky solar energy at ten Brazilian state capitals, using NASA/POWER data:

- **Models:** linear regression, CART, SVR, random forest, gradient boosted trees and an additive spline (GAM-type) model.
- **Predictor sets:** meteorological (F0); + astronomical (F1); + satellite cloud amount, as an upper bound (F2).
- **Validation:** strict temporal validation. Training, tuning and model selection use 2020–2023 only; 2024 and 2025 are independent test years.
- **Evaluation:** skill scores against climatology and persistence, and Diebold–Mariano tests with HAC variance and Holm correction.

## Repository structure

```
data/raw/                     NASA/POWER daily point data, 10 cities, 2020-01-01 to 2025-12-31 (retrieved 2026-10-07)
src/00_download_power.py      (optional) re-download the data from the POWER API
src/01_run_pipeline.py        training, tuning, ex-ante selection, test metrics, DM tests, importance, shift diagnostics
src/02_make_result_figures.py Figs. 2, 4, 5, 6, 7, 8 and supplementary Figs. S1–S2
src/03_make_map.py            Fig. 1 (downloads Natural Earth boundaries on first run)
src/04_make_workflow_figure.py Fig. 3
src/05_make_supplementary.py  supplementary workbook (Tables S1–S5)
src/06_paper_statistics.py    summary statistics quoted in the text (writes results/tables/paper_statistics.txt)
results/tables/               all result tables (CSV) and daily predictions (CSV.gz)
results/figures/              all figures (PNG, 600 dpi)
results/supplementary/        Supplementary_Tables_S1-S5.xlsx
legacy/                       first (Colab) version of the pipeline, kept for transparency
```

## Reproducing the results

Requires Python ≥ 3.10.

```bash
pip install -r requirements.txt
python src/01_run_pipeline.py          # ~15–20 min on 2 CPU cores
python src/02_make_result_figures.py
python src/03_make_map.py
python src/04_make_workflow_figure.py
python src/05_make_supplementary.py
python src/06_paper_statistics.py
```

All scripts use paths relative to the repository root, so they can be run from any folder. The random seed is fixed at 42. Because tree ensembles are built in parallel, a re-execution can differ from the stored results by a few thousandths of kWh·m⁻²·day⁻¹ in RMSE. Conclusions are unaffected.

## Data

- **Source:** NASA Langley Research Center POWER Project, daily point API, Renewable Energy community, local solar time (https://power.larc.nasa.gov).
- **Target:** `ALLSKY_SFC_SW_DWN`, in kWh·m⁻²·day⁻¹.
- **Predictors:** `T2M`, `T2M_MAX`, `T2M_MIN`, `RH2M`, `WS2M`, `PS`, `PRECTOTCORR`, `TOA_SW_DWN` and `CLOUD_AMT`.
- **Snapshot:** the files in `data/raw/` are the exact inputs used in the paper. POWER products may be reprocessed, so a fresh download (`src/00_download_power.py --force`) can differ slightly.

| City | Region | Lat (°) | Lon (°) |
|---|---|---|---|
| Manaus | North | −3.119 | −60.022 |
| Fortaleza | Northeast | −3.732 | −38.527 |
| Recife | Northeast | −8.054 | −34.881 |
| Brasília | Center-West | −15.794 | −47.882 |
| Cuiabá | Center-West | −15.601 | −56.097 |
| Belo Horizonte | Southeast | −19.917 | −43.935 |
| Rio de Janeiro | Southeast | −22.907 | −43.173 |
| São Paulo | Southeast | −23.550 | −46.633 |
| Curitiba | South | −25.428 | −49.273 |
| Porto Alegre | South | −30.035 | −51.218 |

## Main output files

| File | Content |
|---|---|
| `results/tables/results_all.csv` | MAE, RMSE, nRMSE, MBE, R² and skill vs. climatology/persistence for all 360 runs (10 sites × 3 predictor sets × 6 models × 2 test years). `is_selected` marks the ex-ante selected model and `is_oracle` the test-optimal one. |
| `results/tables/diebold_mariano.csv` | DM tests of every model vs. climatology and persistence, with Holm-adjusted p-values for the selected models. |
| `results/tables/grouped_permutation_importance_F1.csv` | Grouped permutation importance (RF and selected F1 model). |
| `results/tables/covariate_shift.csv` | Test-minus-training means of target and predictors. |
| `results/tables/daily_predictions.csv.gz` | Daily estimates of every model, observations (`_obs`), climatology (`_clim`) and persistence (`_pers`) for the test years. |

## Citation

If you use this material, please cite the article (once published) and this archive. See `CITATION.cff`, or use the "Cite this repository" button on GitHub.

## Funding

National Council for Scientific and Technological Development (CNPq), Brazil — Grant 401718/2023-9 (CNPq/MCTI Call No. 10/2023, Category A, Emerging Groups). NASA/POWER data are provided by the NASA Langley Research Center POWER Project, funded through the NASA Earth Science/Applied Science Program.

## License

- **Code** (`src/`, `legacy/`): MIT License (see `LICENSE`).
- **Results and figures** (`results/`): Creative Commons Attribution 4.0 International (CC BY 4.0).
- **NASA/POWER data** (`data/raw/`): redistributed under NASA's open data policy; please also acknowledge the POWER Project.

## Authors

Fernando de Lima Caneppele (USP, corresponding), Alexandre Dal Pai (UNESP), Camila Piacitelli Tieghi (USP), Carlos Eduardo Camargo Nogueira (UNIOESTE), Jair Antonio Cruz Siqueira (UNIOESTE), Murilo Miceno Frigo (IFMS), Marcus Vinícius Contes Calça (SENAI-SP), Diego Cunha Malagueta (UFRJ) and Emmanuel Zullo Godinho (USP).

## Contact

Fernando de Lima Caneppele — University of São Paulo (FZEA/USP) — caneppele@usp.br — ORCID [0000-0003-4498-8682](https://orcid.org/0000-0003-4498-8682)
