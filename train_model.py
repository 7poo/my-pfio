import pandas as pd
import numpy as np
import joblib
import warnings
import os
import sys
import traceback
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

warnings.filterwarnings("ignore")

# --- Configuration ---
DATA_PATH = Path(r"C:\Users\HUNG\Downloads\DATA-20260123T060319Z-1-001\DATA - Copy")
ARTIFACTS_DIR = Path(".")
LOG_FILE = ARTIFACTS_DIR / "training_log.txt"

def log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# --- Feature Engineering Functions ---

def parse_term_id(hk: str) -> int:
    try:
        t, years = str(hk).split(" ")
        y = int(years.split("-")[0])
        s = int(t.replace("HK", ""))
        return y * 10 + s
    except Exception:
        return 0

def safe_num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

def load_data():
    log(f"Loading data from {DATA_PATH}...")
    train = pd.read_csv(DATA_PATH / "academic_records.csv", dtype={"MA_SO_SV": "string"})
    adm = pd.read_csv(DATA_PATH / "admission.csv", dtype={"MA_SO_SV": "string"})
    if "MA_SO_SV" in adm.columns:
        adm = adm.drop_duplicates(subset=["MA_SO_SV"]) 
    if (DATA_PATH / "test.csv").exists():
        test = pd.read_csv(DATA_PATH / "test.csv", dtype={"MA_SO_SV": "string"})
    else:
        test = pd.DataFrame(columns=["MA_SO_SV", "HOC_KY"])
    return train, adm, test

def clean_and_merge(train, adm, test):
    log("Cleaning and merging...")
    train = train.drop_duplicates(subset=["MA_SO_SV", "HOC_KY"], keep="last").copy()
    train = train.merge(adm, on="MA_SO_SV", how="left")
    if not test.empty:
        test = test.merge(adm, on="MA_SO_SV", how="left")
    train = train[
        (train["TC_DANGKY"] >= 0) & (train["TC_HOANTHANH"] >= 0) & (train["TC_HOANTHANH"] <= train["TC_DANGKY"])
    ].copy()
    train.reset_index(drop=True, inplace=True)
    if not test.empty:
        test.reset_index(drop=True, inplace=True)
    return train, test

def add_basic_features(df):
    df = df.copy()
    if "HOC_KY" in df.columns:
        df["Term_ID"] = df["HOC_KY"].apply(parse_term_id)
    if "Term_ID" not in df.columns:
        df["Term_ID"] = 0
    df["ONLINE_COVID"] = df["Term_ID"].between(20201, 20212).astype(int)
    df["term_year"] = (df["Term_ID"] // 10).astype(int)
    df["term_sem"] = (df["Term_ID"] % 10).astype(int)
    df["PRE_COVID"] = (df["Term_ID"] < 20201).astype(int)
    df["POST_COVID"] = (df["Term_ID"] >= 20221).astype(int)
    return df

def add_score_features(train, test, use_score_norm=True):
    train = train.copy()
    test = test.copy()
    score_train = safe_num(train["DIEM_TRUNGTUYEN"])
    score_test = safe_num(test["DIEM_TRUNGTUYEN"])
    cutoff_train = safe_num(train["DIEM_CHUAN"])
    cutoff_test = safe_num(test["DIEM_CHUAN"])
    lo, hi = score_train.quantile(0.01), score_train.quantile(0.99)
    score_train = score_train.clip(lo, hi)
    score_test = score_test.clip(lo, hi)
    train["score_diff"] = score_train - cutoff_train
    test["score_diff"] = score_test - cutoff_test
    train["score_ratio_cut"] = score_train / (cutoff_train.replace(0, np.nan))
    test["score_ratio_cut"] = score_test / (cutoff_test.replace(0, np.nan))
    train["score_ratio_cut"] = train["score_ratio_cut"].fillna(0.0).clip(0, 10)
    test["score_ratio_cut"] = test["score_ratio_cut"].fillna(0.0).clip(0, 10)
    if use_score_norm:
        year_stats = train.groupby("NAM_TUYENSINH")["DIEM_TRUNGTUYEN"].agg(["mean", "std"]).fillna(0.0)
        global_mean = score_train.mean()
        global_std = score_train.std() or 1.0
        def norm_row(row):
            year = row["NAM_TUYENSINH"]
            score = pd.to_numeric(row["DIEM_TRUNGTUYEN"], errors="coerce")
            if pd.isna(score): return 0.0
            score = float(np.clip(score, lo, hi))
            if year in year_stats.index:
                m = year_stats.loc[year, "mean"]
                s = year_stats.loc[year, "std"]
                return (score - m) / s if s > 1e-9 else 0.0
            return (score - global_mean) / global_std
        train["score_norm"] = train.apply(norm_row, axis=1)
        test["score_norm"] = test.apply(norm_row, axis=1)
    return train, test

def add_admission_group_norm(train, test):
    train = train.copy()
    test = test.copy()
    score_train = safe_num(train["DIEM_TRUNGTUYEN"])
    ptxt_stats = train.assign(_score=score_train).groupby("PTXT")["_score"].agg(["mean", "std"])
    combo_stats = train.assign(_score=score_train).groupby(["PTXT", "TOHOP_XT"])["_score"].agg(["mean", "std"])
    def z_by_ptxt(row):
        p = row["PTXT"]
        s = pd.to_numeric(row["DIEM_TRUNGTUYEN"], errors="coerce")
        if pd.isna(s) or p not in ptxt_stats.index: return 0.0
        m, sd = ptxt_stats.loc[p, "mean"], ptxt_stats.loc[p, "std"]
        return (s - m) / sd if sd > 1e-9 else 0.0
    def z_by_combo(row):
        key = (row["PTXT"], row["TOHOP_XT"])
        s = pd.to_numeric(row["DIEM_TRUNGTUYEN"], errors="coerce")
        if pd.isna(s): return z_by_ptxt(row)
        if key not in combo_stats.index: return z_by_ptxt(row)
        m, sd = combo_stats.loc[key, "mean"], combo_stats.loc[key, "std"]
        return (s - m) / sd if sd > 1e-9 else 0.0
    train["score_z_ptxt"] = train.apply(z_by_ptxt, axis=1)
    test["score_z_ptxt"] = test.apply(z_by_ptxt, axis=1)
    train["score_z_ptxt_tohop"] = train.apply(z_by_combo, axis=1)
    test["score_z_ptxt_tohop"] = test.apply(z_by_combo, axis=1)
    return train, test

def add_lag_features(train):
    log("Engineering lag features for training...")
    train = train.copy()
    train["ratio"] = (train["TC_HOANTHANH"] / (train["TC_DANGKY"] + 1e-9)).clip(0, 1)
    train = train.sort_values(["MA_SO_SV", "Term_ID"]).reset_index(drop=True)
    g = train.groupby("MA_SO_SV", sort=False)
    train["count_prev"] = g.cumcount()
    train["lag1_ratio"] = g["ratio"].shift(1).fillna(0.0)
    train["lag1_tc"] = g["TC_DANGKY"].shift(1).fillna(0.0)
    train["lag1_gpa"] = g["GPA"].shift(1).fillna(0.0)
    train["lag1_cpa"] = g["CPA"].shift(1).fillna(0.0)
    train["lag2_ratio"] = g["ratio"].shift(2).fillna(0.0)
    train["lag3_ratio"] = g["ratio"].shift(3).fillna(0.0)
    train["term_prev"] = g["Term_ID"].shift(1).fillna(0.0)
    train["gap_prev"] = (train["Term_ID"] - train["term_prev"]).fillna(0.0)
    train["gap_prev_norm"] = train["gap_prev"] / (train["count_prev"] + 1.0)
    train["term_idx"] = train["count_prev"] + 1
    train["cum_reg"] = g["TC_DANGKY"].cumsum() - train["TC_DANGKY"]
    train["cum_comp"] = g["TC_HOANTHANH"].cumsum() - train["TC_HOANTHANH"]
    train["cum_ratio"] = (train["cum_comp"] / (train["cum_reg"] + 1e-9)).fillna(0.0)
    train["zero_prev_count"] = g["ratio"].apply(lambda s: (s.shift(1) <= 1e-9).cumsum()).reset_index(level=0, drop=True).fillna(0.0)
    train["low_ratio_prev_count"] = g["ratio"].apply(lambda s: (s.shift(1) < 0.5).cumsum()).reset_index(level=0, drop=True).fillna(0.0)
    gpa_cum = g["GPA"].cumsum()
    cpa_cum = g["CPA"].cumsum()
    train["mean_gpa"] = ((gpa_cum - train["GPA"]) / (train["count_prev"].replace(0, np.nan))).fillna(0.0)
    train["mean_cpa"] = ((cpa_cum - train["CPA"]) / (train["count_prev"].replace(0, np.nan))).fillna(0.0)
    cumsum = g["ratio"].cumsum()
    cumsum2 = g["ratio"].apply(lambda s: (s * s).cumsum()).reset_index(level=0, drop=True)
    train["sum_prev"] = cumsum - train["ratio"]
    train["sumsq_prev"] = cumsum2 - train["ratio"] ** 2
    cnt = train["count_prev"].replace(0, np.nan)
    train["mean_ratio_prev"] = (train["sum_prev"] / cnt).fillna(0.0)
    mean_sq = (train["sumsq_prev"] / cnt).fillna(0.0)
    train["std_ratio_prev"] = np.sqrt(np.maximum(mean_sq - train["mean_ratio_prev"] ** 2, 0.0)).fillna(0.0)
    train["min_ratio_prev"] = g["ratio"].cummin().shift(1).fillna(0.0)
    train["load_stress"] = train["TC_DANGKY"] / (train["lag1_tc"] + 1e-9)
    train["trend_ratio"] = train["lag1_ratio"] - train["mean_ratio_prev"]
    train["gpa_x_tc"] = train["lag1_gpa"] * train["TC_DANGKY"]
    train["mean_ratio_last2"] = (train["lag1_ratio"] + train["lag2_ratio"]) / 2.0
    train["mean_ratio_last3"] = (train["lag1_ratio"] + train["lag2_ratio"] + train["lag3_ratio"]) / 3.0
    train["ratio_slope_k3"] = (train["lag1_ratio"] - train["lag3_ratio"]) / 2.0
    train["covid_terms_count"] = g["ONLINE_COVID"].cumsum().shift(1).fillna(0.0)
    train["ratio_covid"] = train["ratio"] * train["ONLINE_COVID"]
    train["covid_ratio_sum_prev"] = g["ratio_covid"].cumsum().shift(1).fillna(0.0)
    train["covid_ratio_mean"] = (train["covid_ratio_sum_prev"] / (train["covid_terms_count"] + 1e-9)).fillna(0.0)
    train["covid_exposure_ratio"] = train["covid_terms_count"] / (train["count_prev"] + 1.0)
    return train

def add_time_features(train, test):
    train = train.copy()
    test = test.copy()
    train_term_year = (train["Term_ID"] // 10).astype(int)
    test_term_year = (test["Term_ID"] // 10).astype(int)
    train["years_since_admission"] = train_term_year - train["NAM_TUYENSINH"].astype(int)
    test["years_since_admission"] = test_term_year - test["NAM_TUYENSINH"].astype(int)
    return train, test

def add_tc_bucket(df):
    df = df.copy()
    tc = safe_num(df["TC_DANGKY"])
    df["tc_bucket"] = pd.cut(tc, bins=[-1, 12, 20, 1000], labels=[0, 1, 2]).astype(int)
    return df

def build_test_features(train, test):
    log("Building inference features (latest state)...")
    def last_k_ratios(s, k):
        vals = s.values[-k:] if len(s) >= k else s.values
        out = [0.0] * (k - len(vals)) + list(vals)
        return out
    hist = train.groupby("MA_SO_SV").agg(
        count_prev=("ratio", "size"),
        lag1_ratio=("ratio", "last"),
        lag1_tc=("TC_DANGKY", "last"),
        lag1_gpa=("GPA", "last"),
        lag1_cpa=("CPA", "last"),
        mean_gpa=("GPA", "mean"),
        mean_cpa=("CPA", "mean"),
        mean_ratio_prev=("ratio", "mean"),
        std_ratio_prev=("ratio", "std"),
        min_ratio_prev=("ratio", "min"),
        term_prev=("Term_ID", "last"),
        covid_terms_count=("ONLINE_COVID", "sum"),
    ).reset_index()
    hist["std_ratio_prev"] = hist["std_ratio_prev"].fillna(0.0)
    last3 = train.groupby("MA_SO_SV")["ratio"].apply(lambda s: last_k_ratios(s, 3))
    last3 = pd.DataFrame(last3.tolist(), index=last3.index, columns=["lag3_ratio", "lag2_ratio", "lag1_ratio_dup"]).reset_index().rename(columns={"index": "MA_SO_SV"})
    hist = hist.merge(last3[["MA_SO_SV", "lag2_ratio", "lag3_ratio"]], on="MA_SO_SV", how="left")
    cum_stats = train.groupby("MA_SO_SV").agg(
        cum_reg=("TC_DANGKY", "sum"),
        cum_comp=("TC_HOANTHANH", "sum"),
        zero_prev_count=("ratio", lambda s: float((s <= 1e-9).sum())),
        low_ratio_prev_count=("ratio", lambda s: float((s < 0.5).sum())),
    ).reset_index()
    hist = hist.merge(cum_stats, on="MA_SO_SV", how="left")
    hist["cum_ratio"] = (hist["cum_comp"] / (hist["cum_reg"] + 1e-9)).fillna(0.0)
    covid_ratio_mean = (
        train.assign(ratio_covid=train["ratio"] * train["ONLINE_COVID"])
        .groupby("MA_SO_SV")["ratio_covid"].mean().reset_index().rename(columns={"ratio_covid": "covid_ratio_mean"})
    )
    hist = hist.merge(covid_ratio_mean, on="MA_SO_SV", how="left")
    test_feat = test.merge(hist, on="MA_SO_SV", how="left")
    fill_cols = [
        "count_prev", "lag1_ratio", "lag2_ratio", "lag3_ratio", "lag1_tc", "lag1_gpa", "lag1_cpa",
        "mean_gpa", "mean_cpa", "mean_ratio_prev", "std_ratio_prev", "min_ratio_prev", "term_prev",
        "covid_terms_count", "covid_ratio_mean", "cum_reg", "cum_comp", "cum_ratio",
        "zero_prev_count", "low_ratio_prev_count"
    ]
    for col in fill_cols:
        test_feat[col] = test_feat[col].fillna(0.0)
    test_feat["gap_prev"] = (test_feat["Term_ID"] - test_feat["term_prev"]).fillna(0.0)
    test_feat["gap_prev_norm"] = test_feat["gap_prev"] / (test_feat["count_prev"] + 1.0)
    test_feat["term_idx"] = test_feat["count_prev"] + 1
    test_feat["load_stress"] = test_feat["TC_DANGKY"] / (test_feat["lag1_tc"] + 1e-9)
    test_feat["trend_ratio"] = test_feat["lag1_ratio"] - test_feat["mean_ratio_prev"]
    test_feat["gpa_x_tc"] = test_feat["lag1_gpa"] * test_feat["TC_DANGKY"]
    test_feat["mean_ratio_last2"] = (test_feat["lag1_ratio"] + test_feat["lag2_ratio"]) / 2.0
    test_feat["mean_ratio_last3"] = (test_feat["lag1_ratio"] + test_feat["lag2_ratio"] + test_feat["lag3_ratio"]) / 3.0
    test_feat["ratio_slope_k3"] = (test_feat["lag1_ratio"] - test_feat["lag3_ratio"]) / 2.0
    test_feat["covid_exposure_ratio"] = test_feat["covid_terms_count"] / (test_feat["count_prev"] + 1.0)
    return test_feat

def build_anchors(train):
    anchor_global = float(train["ratio"].mean())
    anchor_ptxt = train.groupby("PTXT")["ratio"].mean().to_dict()
    anchor_combo = train.groupby(["PTXT", "TOHOP_XT"])["ratio"].mean().to_dict()
    return anchor_global, anchor_ptxt, anchor_combo

def add_anchor_feature(df, anchors):
    anchor_global, anchor_ptxt, anchor_combo = anchors
    def get_anchor(ptxt, tohop):
        key = (str(ptxt), str(tohop))
        return float(anchor_combo.get(key, anchor_ptxt.get(str(ptxt), anchor_global)))
    df = df.copy()
    df["anchor_ratio"] = [get_anchor(p, t) for p, t in zip(df["PTXT"].astype(str), df["TOHOP_XT"].astype(str))]
    return df

def get_feature_cols(use_score_norm):
    base = [
        "TC_DANGKY", "ONLINE_COVID", "PRE_COVID", "POST_COVID", "term_sem", "tc_bucket",
        "count_prev", "gap_prev", "gap_prev_norm", "lag1_ratio", "lag2_ratio", "lag3_ratio",
        "lag1_tc", "lag1_gpa", "lag1_cpa", "mean_gpa", "mean_cpa", "mean_ratio_prev",
        "std_ratio_prev", "min_ratio_prev", "load_stress", "trend_ratio", "gpa_x_tc",
        "mean_ratio_last2", "mean_ratio_last3", "ratio_slope_k3", "cum_reg", "cum_comp",
        "cum_ratio", "zero_prev_count", "low_ratio_prev_count", "score_diff", "score_ratio_cut",
        "covid_terms_count", "covid_ratio_mean", "covid_exposure_ratio", "years_since_admission",
        "term_idx", "anchor_ratio", "score_z_ptxt", "score_z_ptxt_tohop",
    ]
    if use_score_norm:
        base += ["score_norm", "score_year_interact"]
    return base

# --- Main Training Flow ---

def main():
    try:
        if LOG_FILE.exists():
            try:
                os.remove(LOG_FILE)
            except: 
                pass
        
        use_score_norm = True
        train, adm, test_orig = load_data()
        train, test_orig = clean_and_merge(train, adm, test_orig)
        train = add_basic_features(train)
        test_orig = add_basic_features(test_orig)
        train, test_orig = add_score_features(train, test_orig, use_score_norm)
        train, test_orig = add_time_features(train, test_orig)
        train, test_orig = add_admission_group_norm(train, test_orig)
        train = add_tc_bucket(train)
        test_orig = add_tc_bucket(test_orig)
        train = add_lag_features(train)
        anchors_full = build_anchors(train)
        train = add_anchor_feature(train, anchors_full)
        if use_score_norm:
            train["score_year_interact"] = train["score_norm"] * train["NAM_TUYENSINH"]
        
        log("Training model (Pipeline)...")
        num_cols = get_feature_cols(use_score_norm)
        
        preprocessor = ColumnTransformer(
            transformers=[("num", "passthrough", num_cols)],
            remainder="drop",
            verbose_feature_names_out=False
        )
        
        model = Pipeline([
            ("prep", preprocessor),
            ("m", HistGradientBoostingRegressor(
                max_iter=1000, learning_rate=0.03, max_depth=12,
                l2_regularization=0.05, random_state=42
            ))
        ])
        
        y = train["ratio"].values.astype(float)
        model.fit(train, y)
        
        joblib.dump(model, ARTIFACTS_DIR / "student_credit_model.pkl")
        log(f"Saved student_credit_model.pkl")
        
        log("Generating latest state for all students...")
        all_students = train[["MA_SO_SV", "NAM_TUYENSINH", "PTXT", "TOHOP_XT", "DIEM_TRUNGTUYEN", "DIEM_CHUAN"]].drop_duplicates("MA_SO_SV")
        
        last_term = train.groupby("MA_SO_SV")["Term_ID"].max().reset_index()
        last_term["Next_Term_ID"] = last_term["Term_ID"].apply(lambda t: t + 1 if t % 10 == 1 else t + 9)
        last_term = last_term.drop(columns=["Term_ID"]).rename(columns={"Next_Term_ID": "Term_ID"})
        
        all_test = all_students.merge(last_term, on="MA_SO_SV")
        
        def term_to_str(t):
            y = t // 10
            s = t % 10
            return f"HK{s} {y}-{y+1}"
        
        all_test["HOC_KY"] = all_test["Term_ID"].apply(term_to_str)
        all_test["TC_DANGKY"] = 15 
        all_test["TC_HOANTHANH"] = 0 
        
        all_test = add_basic_features(all_test)
        all_test, _ = add_score_features(all_test, all_test, use_score_norm) 
        all_test, _ = add_time_features(all_test, all_test)
        all_test, _ = add_admission_group_norm(all_test, all_test)
        all_test = add_tc_bucket(all_test)
        
        all_test_feat = build_test_features(train, all_test)
        all_test_feat = add_anchor_feature(all_test_feat, anchors_full)
        
        if use_score_norm:
            all_test_feat["score_year_interact"] = all_test_feat["score_norm"] * all_test_feat["NAM_TUYENSINH"]

        joblib.dump(all_test_feat, ARTIFACTS_DIR / "latest_student_state.pkl")
        log("Saved latest_student_state.pkl")
        joblib.dump(train, ARTIFACTS_DIR / "full_history_processed.pkl")
        log("Saved full_history_processed.pkl")
        log("DONE")
    except Exception as e:
        log(f"ERROR: {e}")
        log(traceback.format_exc())

if __name__ == "__main__":
    main()
