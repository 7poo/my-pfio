import re
import numpy as np
import pandas as pd

def parse_term(hk: str):
    """Parses 'HK1 2020-2021' into semester, start_year, and a linear term_idx."""
    s = str(hk).strip()
    m = re.search(r"HK\s*([12]).*?(\d{4})\s*-\s*(\d{4})", s)
    if not m:
        return np.nan, np.nan, np.nan
    sem = int(m.group(1))
    start_year = int(m.group(2))
    # Term index: 2020*2 + 0 (HK1) or 1 (HK2)
    # Normalized so we can compare timeline
    term_idx = start_year * 2 + (sem - 1)
    return sem, start_year, term_idx

def load_and_merge_data(academic_path, admission_path):
    """
    Loads raw CSVs and performs initial merge and Year of Study calculation.
    """
    ac = pd.read_csv(academic_path)
    ad = pd.read_csv(admission_path)
    
    # 1. Parse Terms
    sems, years, idxs = [], [], []
    for hk in ac["HOC_KY"]:
        sem, yr, idx = parse_term(hk)
        sems.append(sem); years.append(yr); idxs.append(idx)
    ac["SEM"] = sems
    ac["START_YEAR"] = years
    ac["TERM_IDX"] = idxs
    
    # Filter valid terms
    ac = ac[ac["TERM_IDX"].notna()].copy()
    
    # 2. Admission Preprocessing
    ad["PTXT"] = ad["PTXT"].astype(str)
    ad["TOHOP_XT"] = ad["TOHOP_XT"].astype(str)
    ad["Score_Surplus"] = ad["DIEM_TRUNGTUYEN"] - ad["DIEM_CHUAN"]
    ad["Score_Ratio_Cut"] = ad["DIEM_TRUNGTUYEN"] / (ad["DIEM_CHUAN"] + 1e-9)
    
    # Z-score within PTXT (Method)
    grp = ad.groupby("PTXT")["DIEM_TRUNGTUYEN"]
    mu = grp.transform("mean")
    sd = grp.transform("std").replace(0, 1.0)
    ad["DIEM_Z_PTXT"] = (ad["DIEM_TRUNGTUYEN"] - mu) / sd
    
    # 3. Merge
    df = ac.merge(ad, on="MA_SO_SV", how="left")
    
    # Year of Study
    df["Year_of_Study"] = df["START_YEAR"] - df["NAM_TUYENSINH"]
    
    # Ratio Calculation
    df = df[(df["TC_DANGKY"] > 0)].copy() # Drop zero reg
    df["ratio"] = (df["TC_HOANTHANH"].fillna(0) / df["TC_DANGKY"]).clip(0, 1)
    
    return df

def feature_engineering_history(df):
    """
    Computes lag/rolling features. 
    IMPORTANT: Data must be sorted by Student and Term.
    """
    df = df.sort_values(["MA_SO_SV", "TERM_IDX"]).copy()
    
    # Lag 1
    df["lag1_ratio"] = df.groupby("MA_SO_SV")["ratio"].shift(1)
    df["lag1_GPA"]   = df.groupby("MA_SO_SV")["GPA"].shift(1)
    df["lag1_CPA"]   = df.groupby("MA_SO_SV")["CPA"].shift(1)
    
    # Expanding Mean/Std (Historical Performance)
    # shift(1) ensures we only see past data
    df["mean_ratio"] = df.groupby("MA_SO_SV")["ratio"].transform(lambda s: s.expanding().mean().shift(1))
    df["std_ratio"]  = df.groupby("MA_SO_SV")["ratio"].transform(lambda s: s.expanding().std().shift(1))
    
    # Cumulative Credits
    df["cum_reg"]  = df.groupby("MA_SO_SV")["TC_DANGKY"].transform(lambda s: s.fillna(0).cumsum().shift(1))
    df["cum_comp"] = df.groupby("MA_SO_SV")["TC_HOANTHANH"].transform(lambda s: s.fillna(0).cumsum().shift(1))
    df["cum_ratio"] = (df["cum_comp"] / (df["cum_reg"] + 1e-9)).clip(0, 1)
    
    # Semester Count (Experience)
    df["Count_Semester"] = df.groupby("MA_SO_SV").cumcount()
    
    return df
