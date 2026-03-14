import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import logging
from .utils import evaluate_and_plot

logger = logging.getLogger(__name__)

EPS = 1e-12
MIN_S_VALUE = 1e-6
MIN_R_CHAR = 1e-6

#PCA-Dubina here from BACalc
def zscore(x_array: np.ndarray):
    mu = np.nanmean(x_array, axis=0)
    sd = np.nanstd(x_array, axis=0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)
    z_values = (x_array - mu) / sd
    return z_values, mu, sd

def t_scale(bas: np.ndarray, ca: np.ndarray, ddof: int = 0):
    bas_mu = np.nanmean(bas)
    bas_sd = np.nanstd(bas, ddof=ddof) or 1.0
    ca_mu = np.nanmean(ca)
    ca_sd = np.nanstd(ca, ddof=ddof)
    return (bas - bas_mu) * (ca_sd / bas_sd) + ca_mu

def dubina_correct(BA: np.ndarray, CA: np.ndarray):
    ca_mean = np.nanmean(CA)
    var_ca = np.nanvar(CA, ddof=0) + EPS
    cov_ba_ca = np.nanmean((BA - np.nanmean(BA)) * (CA - ca_mean))
    b = cov_ba_ca / var_ca
    return (
        BA + (CA - ca_mean) * (1.0 - b),
        float(b),
        float(ca_mean),
        float(np.nanstd(CA, ddof=0)),
    )

#KDM here from BACalc
def train_kdm_params(df_sub, age_col, biom_cols, s2_ba_floor=0.1):
    X_age = df_sub[[age_col]].values
    k_vals, q_vals, s_vals, r_vals = [], [], [], []

    for biom in biom_cols:
        y = df_sub[[biom]].values
        if np.var(y) < 1e-10:
            k, q, s, r = 0.0, 0.0, 1.0, 0.0
        else:
            mdl = LinearRegression().fit(X_age, y)
            k = mdl.coef_[0][0]
            q = mdl.intercept_[0]
            y_pred = mdl.predict(X_age)
            s = np.sqrt(mean_squared_error(y, y_pred))
            r_stat, _ = stats.pearsonr(df_sub[age_col], df_sub[biom])
            r = r_stat if not np.isnan(r_stat) else 0.0

        k_vals.append(k)
        q_vals.append(q)
        s_vals.append(s if s > MIN_S_VALUE else 1.0)
        r_vals.append(r)

    num, den = 0, 0
    for r in r_vals:
        if abs(r) >= 1 or abs(r) < MIN_R_CHAR:
            continue
        sqrt_term = np.sqrt(1 - r**2)
        num += (r**2) / sqrt_term
        den += r / sqrt_term

    r_char = (num / den) if abs(den) > 1e-9 else 0.0
    r_char = np.clip(r_char, -0.9999, 0.9999)

    n = len(df_sub)
    m = len(biom_cols)
    ba_e_vals = []

    for i in range(n):
        num_kdm, den_kdm = 0, 0
        for j in range(m):
            val = df_sub[biom_cols[j]].iloc[i]
            if abs(k_vals[j]) < 1e-9:
                continue

            weight = (k_vals[j] / s_vals[j]) ** 2
            age_equiv = (val - q_vals[j]) / k_vals[j]

            num_kdm += age_equiv * weight
            den_kdm += weight

        ba_e_vals.append(num_kdm / den_kdm if den_kdm > 1e-9 else np.nan)

    ba_e_arr = np.array(ba_e_vals)
    ca_arr = df_sub[age_col].values

    diff = ba_e_arr - ca_arr
    diff = diff[~np.isnan(diff)]
    term1 = np.var(diff, ddof=0) if len(diff) > 1 else 10.0

    ca_range = np.max(ca_arr) - np.min(ca_arr)
    var_ca_approx = ((ca_range**2) / 12.0) if ca_range > 1e-9 else 0.0

    try:
        if abs(r_char) > MIN_R_CHAR:
            factor_A = (1.0 - r_char**2) / (r_char**2)
            factor_B = var_ca_approx / m
            term2 = factor_A * factor_B
        else:
            term2 = 0
    except:
        term2 = 0

    s2_ba = max(s2_ba_floor, term1 - term2)

    return {
        "k": k_vals, "q": q_vals, "s": s_vals,
        "r": r_vals, "r_char": r_char, "s2_ba": s2_ba,
    }

def calculate_kdm_scores(row, biom_cols, params, ca_val):
    num, den = 0, 0
    k, q, s = params["k"], params["q"], params["s"]
    s2_ba = params["s2_ba"]

    for j, biom in enumerate(biom_cols):
        val = row[biom]
        if np.isnan(val) or abs(k[j]) < 1e-9:
            continue

        weight = (k[j] / s[j]) ** 2
        char_age = (val - q[j]) / k[j]

        num += char_age * weight
        den += weight

    if den < 1e-9:
        return np.nan, np.nan

    ba_e = num / den
    num_ec = num + (ca_val / s2_ba)
    den_ec = den + (1.0 / s2_ba)
    ba_ec = num_ec / den_ec

    return ba_e, ba_ec

def run_kdm(df_train, df_test, age_col, biom_cols, s2_ba_floor=0.1, out_path=None):
    logger.info("Running KDM (using original core implementation)...")
    kdm_params = train_kdm_params(df_train, age_col, biom_cols, s2_ba_floor)
    
    ba_ec_list = []
    for idx, row in df_test.iterrows():
        _, val_ec = calculate_kdm_scores(row, biom_cols, kdm_params, row[age_col])
        ba_ec_list.append(val_ec)

    df_out = df_test[[age_col]].copy()
    df_out["KDM_BA"] = ba_ec_list
    
    metrics = evaluate_and_plot(df_out[age_col].values, df_out["KDM_BA"].values, "KDM", out_path, "#1f77b4")
    return df_out, metrics

def run_pca_dubina(df_train, df_test, age_col, biom_cols, out_path=None):
    logger.info("Running PCA-Dubina (using original core implementation)...")
    
    #train Parameters
    x_train = df_train[biom_cols].to_numpy(dtype=float)
    z_values, mu_x, sd_x = zscore(x_train)

    pca = PCA(n_components=1, svd_solver="full")
    pca.fit(z_values)
    pc1 = pca.components_[0].copy()

    ca_train = df_train[age_col].to_numpy(dtype=float)
    r = np.corrcoef(z_values @ pc1, ca_train)[0, 1]
    if r < 0:
        pc1 *= -1

    wn = pc1 / sd_x
    w0 = float(-np.sum(wn * mu_x))
    BAS_train = x_train.dot(wn) + w0

    BA_train = t_scale(BAS_train, ca_train)
    BAc_train, b, ca_mean, ca_sd = dubina_correct(BA_train, ca_train)
    
    bas_mu = np.nanmean(BAS_train)
    bas_sd = np.nanstd(BAS_train, ddof=0) or 1.0

    #apply to Test Data
    x_test = df_test[biom_cols].to_numpy(dtype=float)
    ca_test = df_test[age_col].to_numpy(dtype=float)
    
    BAS_test = x_test.dot(wn) + w0
    BA_test = (BAS_test - bas_mu) * (ca_sd / bas_sd) + ca_mean
    BAc_test = BA_test + (ca_test - ca_mean) * (1.0 - b)

    df_out = df_test[[age_col]].copy()
    df_out["PCA_BA"] = BAc_test
    
    metrics = evaluate_and_plot(df_out[age_col].values, df_out["PCA_BA"].values, "PCA-Dubina", out_path, "#ff7f0e")
    return df_out, metrics
