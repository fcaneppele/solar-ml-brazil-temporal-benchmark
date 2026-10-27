"""S2 pipeline v2 (revised for IEEE Access submission).

Changes relative to S2_pipeline_revisado.ipynb (v1):
  1. The "selected" model per city x predictor set is chosen ex ante, by the
     time-series cross-validated RMSE on the training years (2020-2023), not by
     test-year RMSE. Ex-post best (oracle) is still stored for reference.
  2. Diebold-Mariano test uses a Newey-West (Bartlett) HAC long-run variance with
     automatic lag L = floor(4 (n/100)^(2/9)), plus the HLN small-sample factor,
     t(n-1) reference. Holm-Bonferroni adjustment over the 60 tests.
     DM is run for every model vs climatology and vs persistence (supplementary).
  3. Grouped permutation importance (physically coherent predictor groups),
     on the CV-selected model and on RF, test 2024 and 2025.
  4. Covariate-shift diagnostics: test-minus-training mean of every predictor
     and of the target, per city and test year.
Data: the NASA/POWER daily CSVs in data/raw/ (retrieved 2026-10-07 with src/00_download_power.py).
Outputs: results/tables/*.csv and results/tables/daily_predictions.csv.gz
Runtime: about 15-20 min on a 2-core machine.
"""
import os, json, sys, time, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)  # all paths below are relative to the repository root

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
OUT = "results/tables"
os.makedirs(OUT, exist_ok=True)
SEED = 42
TRAIN_YEARS = [2020, 2021, 2022, 2023]
TEST_YEARS = [2024, 2025]
TARGET = "ALLSKY_SFC_SW_DWN"

CITIES = {
    "Manaus": (-3.119, -60.022, "North"),
    "Fortaleza": (-3.732, -38.527, "Northeast"),
    "Recife": (-8.054, -34.881, "Northeast"),
    "Brasilia": (-15.794, -47.882, "Center-West"),
    "Cuiaba": (-15.601, -56.097, "Center-West"),
    "Belo Horizonte": (-19.917, -43.935, "Southeast"),
    "Rio de Janeiro": (-22.907, -43.173, "Southeast"),
    "Sao Paulo": (-23.550, -46.633, "Southeast"),
    "Curitiba": (-25.428, -49.273, "South"),
    "Porto Alegre": (-30.035, -51.218, "South"),
}
F0 = ["T2M", "RH2M", "WS2M", "PS", "PRECTOTCORR"]
F1 = F0 + ["TOA_SW_DWN", "DTR", "DOY_SIN", "DOY_COS"]
F2 = F1 + ["CLOUD_AMT"]
FEATURE_SETS = {"F0": F0, "F1": F1, "F2": F2}
GROUPS = {
    "Temperature (T2M)": ["T2M"],
    "Diurnal temperature range": ["DTR"],
    "Relative humidity": ["RH2M"],
    "Precipitation": ["PRECTOTCORR"],
    "Wind and pressure": ["WS2M", "PS"],
    "Astronomical (TOA, DOY)": ["TOA_SW_DWN", "DOY_SIN", "DOY_COS"],
}


def build_features(df):
    d = df.copy()
    d["DTR"] = d["T2M_MAX"] - d["T2M_MIN"]
    doy = d.index.dayofyear
    d["DOY_SIN"] = np.sin(2 * np.pi * doy / 365.25)
    d["DOY_COS"] = np.cos(2 * np.pi * doy / 365.25)
    d["YEAR"] = d.index.year
    return d


def make_models():
    return {
        "Linear": (Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]), None),
        "CART": (DecisionTreeRegressor(random_state=SEED),
                 {"max_depth": [4, 6, 8, 12], "min_samples_leaf": [5, 10, 20]}),
        "SVM": (Pipeline([("sc", StandardScaler()), ("m", SVR())]),
                {"m__C": [1, 10, 30], "m__gamma": ["scale", 0.05], "m__epsilon": [0.1, 0.3]}),
        "RF": (RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=-1),
               {"max_depth": [None, 12], "min_samples_leaf": [1, 5], "max_features": [0.5, 1.0]}),
        "GBT": (GradientBoostingRegressor(random_state=SEED),
                {"n_estimators": [200, 500], "learning_rate": [0.03, 0.1], "max_depth": [2, 3]}),
        "GAM": (Pipeline([("sc", StandardScaler()), ("sp", SplineTransformer(n_knots=8, degree=3)),
                          ("m", Ridge())]), {"m__alpha": [0.1, 1, 10]}),
    }


def metrics(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    rmse = np.sqrt(mean_squared_error(y, yhat))
    return {"MAE": mean_absolute_error(y, yhat), "RMSE": rmse, "nRMSE_%": 100 * rmse / np.mean(y),
            "MBE": np.mean(yhat - y), "R2": r2_score(y, yhat)}


def climatology_ref(train, test):
    clim = train.groupby(train.index.dayofyear)[TARGET].mean().reindex(range(1, 367))
    clim = clim.interpolate(limit_direction="both")
    ext = pd.concat([clim.iloc[-7:], clim, clim.iloc[:7]])
    clim = ext.rolling(15, center=True).mean().iloc[7:-7]
    return clim.reindex(test.index.dayofyear).values


def persistence_ref(full, test):
    return full[TARGET].shift(1).reindex(test.index).values


def dm_test(e1, e2, h=1):
    """DM test, squared loss, Newey-West HAC variance + HLN factor.
    stat < 0 -> forecast 1 more accurate. Returns (stat, two-sided p, lag)."""
    d = np.asarray(e1) ** 2 - np.asarray(e2) ** 2
    d = d[~np.isnan(d)]
    n = len(d)
    dbar = d.mean()
    L = int(np.floor(4 * (n / 100) ** (2 / 9)))
    L = max(L, h - 1)
    dc = d - dbar
    lrv = np.sum(dc * dc) / n
    for k in range(1, L + 1):
        w = 1 - k / (L + 1)
        lrv += 2 * w * np.sum(dc[k:] * dc[:-k]) / n
    dm = dbar / np.sqrt(lrv / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    p = 2 * stats.t.sf(np.abs(stat), df=n - 1)
    return stat, p, L


def holm(p):
    p = np.asarray(p)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def grouped_perm_importance(model, X, y, groups, n_repeats=20, seed=SEED):
    rng = np.random.default_rng(seed)
    base = np.sqrt(mean_squared_error(y, model.predict(X)))
    out = {}
    for g, cols in groups.items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        inc = []
        for _ in range(n_repeats):
            Xp = X.copy()
            idx = rng.permutation(len(Xp))
            Xp[cols] = X[cols].values[idx]
            inc.append(np.sqrt(mean_squared_error(y, model.predict(Xp))) - base)
        out[g] = (np.mean(inc), np.std(inc))
    return base, out


def main():
    t0 = time.time()
    frames = {c: pd.read_csv(f"data/raw/{c.replace(' ', '_')}.csv", index_col=0, parse_dates=True)
              for c in CITIES}
    rows, preds, fitted, shift_rows = [], {}, {}, []
    for city, raw in frames.items():
        region = CITIES[city][2]
        d = build_features(raw)
        tr_all = d[d.YEAR.isin(TRAIN_YEARS)]
        for ty in TEST_YEARS:
            te_all = d[d.YEAR == ty]
            row = {"City": city, "Region": region, "TestYear": ty}
            for v in [TARGET] + F2 + ["T2M_MAX", "T2M_MIN"]:
                row[f"d_{v}"] = te_all[v].mean() - tr_all[v].mean()
            shift_rows.append(row)
        for fs, cols in FEATURE_SETS.items():
            dd = d.dropna(subset=cols + [TARGET])
            train = dd[dd.YEAR.isin(TRAIN_YEARS)]
            Xtr, ytr = train[cols], train[TARGET]
            tscv = TimeSeriesSplit(n_splits=4)
            for mname, (est, grid) in make_models().items():
                if grid:
                    gs = GridSearchCV(est, grid, cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1)
                    gs.fit(Xtr, ytr)
                    model, best, cv_rmse = gs.best_estimator_, gs.best_params_, -gs.best_score_
                else:
                    cv_rmse = -cross_val_score(est, Xtr, ytr, cv=tscv,
                                               scoring="neg_root_mean_squared_error").mean()
                    model, best = est.fit(Xtr, ytr), {}
                fitted[(city, fs, mname)] = model
                for ty in TEST_YEARS:
                    test = dd[dd.YEAR == ty]
                    yhat = model.predict(test[cols])
                    clim = climatology_ref(train, test)
                    pers = persistence_ref(dd, test)
                    m = metrics(test[TARGET], yhat)
                    yv = test[TARGET].values
                    rmse_c = np.sqrt(np.nanmean((yv - clim) ** 2))
                    rmse_p = np.sqrt(np.nanmean((yv - pers) ** 2))
                    mbe_c = np.nanmean(clim - yv)
                    m.update({"CV_RMSE_train": cv_rmse,
                              "Skill_vs_Climatology": 1 - m["RMSE"] / rmse_c,
                              "Skill_vs_Persistence": 1 - m["RMSE"] / rmse_p,
                              "RMSE_Climatology": rmse_c, "RMSE_Persistence": rmse_p,
                              "MBE_Climatology": mbe_c})
                    rows.append({"City": city, "Region": region, "FeatureSet": fs, "Model": mname,
                                 "TestYear": ty, "n_test": len(test), **m, "BestParams": json.dumps(best)})
                    preds[(city, fs, mname, ty)] = pd.Series(yhat, index=test.index)
                    preds[(city, fs, "_obs", ty)] = test[TARGET]
                    preds[(city, fs, "_clim", ty)] = pd.Series(clim, index=test.index)
                    preds[(city, fs, "_pers", ty)] = pd.Series(pers, index=test.index)
            print(f"OK {city:15s} {fs}  ({time.time() - t0:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    # ex-ante selection (min CV RMSE on training) and ex-post oracle
    sel = (res.sort_values("CV_RMSE_train").groupby(["City", "FeatureSet", "TestYear"]).first()
           .reset_index()[["City", "FeatureSet", "TestYear", "Model"]].rename(columns={"Model": "Selected"}))
    orc = (res.sort_values("RMSE").groupby(["City", "FeatureSet", "TestYear"]).first()
           .reset_index()[["City", "FeatureSet", "TestYear", "Model"]].rename(columns={"Model": "Oracle"}))
    res = res.merge(sel, on=["City", "FeatureSet", "TestYear"]).merge(orc, on=["City", "FeatureSet", "TestYear"])
    res["is_selected"] = res.Model == res.Selected
    res["is_oracle"] = res.Model == res.Oracle

    # DM for every model vs climatology and vs persistence
    dm_rows = []
    for _, r in res.iterrows():
        k = (r.City, r.FeatureSet)
        obs = preds[(*k, "_obs", r.TestYear)].values
        e_m = obs - preds[(*k, r.Model, r.TestYear)].values
        for ref in ["_clim", "_pers"]:
            e_r = obs - preds[(*k, ref, r.TestYear)].values
            mask = ~np.isnan(e_r)
            s, p, L = dm_test(e_m[mask], e_r[mask])
            dm_rows.append({"City": r.City, "FeatureSet": r.FeatureSet, "TestYear": r.TestYear,
                            "Model": r.Model, "Reference": ref[1:], "DM_stat": s, "p_value": p,
                            "NW_lag": L, "is_selected": r.is_selected, "is_oracle": r.is_oracle})
    dm = pd.DataFrame(dm_rows)
    for (ref, flag), g in dm.groupby(["Reference", "is_selected"]):
        if flag:
            dm.loc[g.index, "p_holm_selected"] = holm(g.p_value.values)
    dm["beats_5pct_raw"] = (dm.DM_stat < 0) & (dm.p_value < 0.05)
    dm["beats_5pct_holm"] = (dm.DM_stat < 0) & (dm.p_holm_selected < 0.05)

    # grouped permutation importance (F1), RF and selected model
    imp_rows = []
    for city in CITIES:
        d = build_features(frames[city]).dropna(subset=F1 + [TARGET])
        for ty in TEST_YEARS:
            test = d[d.YEAR == ty]
            selm = sel[(sel.City == city) & (sel.FeatureSet == "F1") & (sel.TestYear == ty)].Selected.iloc[0]
            for lab, mname in [("RF", "RF"), ("Selected", selm)]:
                base, imp = grouped_perm_importance(fitted[(city, "F1", mname)], test[F1], test[TARGET], GROUPS)
                for g, (mu, sd) in imp.items():
                    imp_rows.append({"City": city, "TestYear": ty, "Which": lab, "Model": mname,
                                     "Group": g, "dRMSE_mean": mu, "dRMSE_sd": sd, "base_RMSE": base})
    imp = pd.DataFrame(imp_rows)

    res.to_csv(f"{OUT}/results_all.csv", index=False, float_format="%.5f")
    dm.to_csv(f"{OUT}/diebold_mariano.csv", index=False, float_format="%.5f")
    imp.to_csv(f"{OUT}/grouped_permutation_importance_F1.csv", index=False, float_format="%.5f")
    pd.DataFrame(shift_rows).to_csv(f"{OUT}/covariate_shift.csv", index=False, float_format="%.4f")
    # daily predictions in long format (portable; used by the figure scripts)
    pr = []
    for (city, fs, model, ty), ser in preds.items():
        pr.append(pd.DataFrame({"date": ser.index, "City": city, "FeatureSet": fs, "Model": model,
                                "TestYear": ty, "value": ser.values}))
    pd.concat(pr).to_csv(f"{OUT}/daily_predictions.csv.gz", index=False, float_format="%.5f")
    print(f"DONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
