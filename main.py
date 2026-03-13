import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, ElasticNetCV, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

#here we load the config.yml file that controls the entire model
def load_config(config_path="config.yml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

#here choice of imputation will be set as defined in config
def apply_imputation(df, cols, method):
    if method == "median":
        df[cols] = SimpleImputer(strategy="median").fit_transform(df[cols])
    elif method == "mean":
        df[cols] = SimpleImputer(strategy="mean").fit_transform(df[cols])
    elif method == "zero":
        df[cols] = SimpleImputer(strategy="constant", fill_value=0).fit_transform(df[cols])
    elif method == "knn":
        df[cols] = KNNImputer(n_neighbors=5).fit_transform(df[cols])
    else:
        raise ValueError(f"unknown imputation method: {method}")
    return df

#here plots will be generated, nothing special
def plot_ba_vs_ca(y_true, y_pred, method_name, filepath, color, axis_limit=130):
    r, _ = stats.pearsonr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    stats_text = f"Pearson r: {r:.3f}\nR²: {r2:.3f}\nRMSE: {rmse:.3f}\nMAE: {mae:.3f}"

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, label=method_name, color=color)
    plt.plot([0, axis_limit], [0, axis_limit], 'r--', linewidth=2, label="Identity (y=x)")
    
    plt.xlim(0, axis_limit)
    plt.ylim(0, axis_limit)
    
    plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    plt.title(f"{method_name} vs Chronological Age")
    plt.xlabel("Chronological Age (CA)")
    plt.ylabel("Biological Age (BA)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

#our KDM method -- taken from previous BACalc
def run_kdm(df_train, df_test, age_col, biom_cols, config, out_dir):
    print("Running KDM...")
    k_vals, q_vals, s_vals, r_vals = [], [], [], []
    X_age_train = df_train[[age_col]].values

    for biom in biom_cols:
        y_train = df_train[[biom]].values
        if np.var(y_train) < 1e-10:
            k, q, s, r = 0.0, 0.0, 1.0, 0.0
        else:
            mdl = LinearRegression().fit(X_age_train, y_train)
            k, q = mdl.coef_[0][0], mdl.intercept_[0]
            s = max(np.sqrt(mean_squared_error(y_train, mdl.predict(X_age_train))), 1e-6)
            r = stats.pearsonr(df_train[age_col], df_train[biom])[0]
        k_vals.append(k); q_vals.append(q); s_vals.append(s); r_vals.append(r if not np.isnan(r) else 0.0)

    num = sum((r**2) / np.sqrt(1 - r**2) for r in r_vals if 1e-6 <= abs(r) < 1)
    den = sum(r / np.sqrt(1 - r**2) for r in r_vals if 1e-6 <= abs(r) < 1)
    r_char = np.clip(num / den if abs(den) > 1e-9 else 0.0, -0.9999, 0.9999)

    train_diffs = []
    for _, row in df_train.iterrows():
        n_kdm, d_kdm = 0, 0
        for j, biom in enumerate(biom_cols):
            if abs(k_vals[j]) < 1e-9: continue
            w = (k_vals[j] / s_vals[j]) ** 2
            n_kdm += ((row[biom] - q_vals[j]) / k_vals[j]) * w
            d_kdm += w
        train_diffs.append((n_kdm / d_kdm) - row[age_col] if d_kdm > 1e-9 else np.nan)
        
    term1 = np.nanvar(train_diffs, ddof=0)
    ca_range = df_train[age_col].max() - df_train[age_col].min()
    term2 = ((1.0 - r_char**2) / (r_char**2)) * (((ca_range**2) / 12.0) / len(biom_cols)) if abs(r_char) > 1e-6 else 0
    s2_ba = max(config.get('s2_ba_floor', 0.1), term1 - term2)

    df_out = df_test[[age_col]].copy()
    ba_ec_vals = []
    for _, row in df_test.iterrows():
        n_kdm, d_kdm = 0, 0
        for j, biom in enumerate(biom_cols):
            if abs(k_vals[j]) < 1e-9: continue
            w = (k_vals[j] / s_vals[j]) ** 2
            n_kdm += ((row[biom] - q_vals[j]) / k_vals[j]) * w
            d_kdm += w
        
        ba_e = n_kdm / d_kdm if d_kdm > 1e-9 else np.nan
        ba_ec = (ba_e * d_kdm + row[age_col] / s2_ba) / (d_kdm + 1.0 / s2_ba)
        ba_ec_vals.append(ba_ec)

    df_out["KDM_BA"] = ba_ec_vals
    df_out.to_csv(out_dir / "kdm_predictions.csv", index=False)
    plot_ba_vs_ca(df_out[age_col], df_out["KDM_BA"], "KDM (Test Set)", out_dir / "plot_kdm.png", "#1f77b4")
    return df_out

#our PCA-Dubina method -- taken from our previous BACalc
def run_pca_dubina(df_train, df_test, age_col, biom_cols, out_dir):
    print("Running PCA-Dubina...")
    scaler = StandardScaler()
    z_train = scaler.fit_transform(df_train[biom_cols])
    z_test = scaler.transform(df_test[biom_cols])
    
    ca_train = df_train[age_col].to_numpy(dtype=float)
    ca_test = df_test[age_col].to_numpy(dtype=float)

    pca = PCA(n_components=1)
    bas_train = pca.fit_transform(z_train).flatten()
    bas_test = pca.transform(z_test).flatten()

    if np.corrcoef(bas_train, ca_train)[0, 1] < 0:
        bas_train *= -1
        bas_test *= -1

    bas_train_mu, bas_train_sd = np.nanmean(bas_train), (np.nanstd(bas_train, ddof=0) or 1.0)
    ca_train_mu, ca_train_sd = np.nanmean(ca_train), np.nanstd(ca_train, ddof=0)
    
    ba_train = (bas_train - bas_train_mu) * (ca_train_sd / bas_train_sd) + ca_train_mu
    ba_test = (bas_test - bas_train_mu) * (ca_train_sd / bas_train_sd) + ca_train_mu
    
    b = np.nanmean((ba_train - np.nanmean(ba_train)) * (ca_train - ca_train_mu)) / (np.nanvar(ca_train, ddof=0) + 1e-12)
    bac_test = ba_test + (ca_test - ca_train_mu) * (1.0 - b)

    df_out = df_test[[age_col]].copy()
    df_out["PCA_BA"] = bac_test
    df_out.to_csv(out_dir / "pca_dubina_predictions.csv", index=False)
    plot_ba_vs_ca(df_out[age_col], df_out["PCA_BA"], "PCA-Dubina (Test Set)", out_dir / "plot_pca_dubina.png", "#ff7f0e")
    return df_out

#here newly added ml methods
def run_ml_models(df_train, df_test, age_col, biom_cols, config, split_config, out_dir):
    print("Running Machine Learning Models...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(df_train[biom_cols])
    X_test_s = scaler.transform(df_test[biom_cols])
    
    y_train = df_train[age_col]
    results_df = pd.DataFrame({"Chronological_Age": df_test[age_col]}, index=df_test.index)

    grid_search_enabled = config.get('grid_search', False)
    cv_folds = config.get('cv_folds', 5)
    random_seed = split_config.get('random_state', 42)
    models = {}

    if config['elastic_net']['run']:
        if grid_search_enabled:
            base_model = ElasticNet(random_state=random_seed)
            grid = config['elastic_net'].get('param_grid', {})
            models["Elastic Net"] = (GridSearchCV(base_model, grid, cv=cv_folds, scoring='neg_mean_squared_error', n_jobs=-1), "#2ca02c")
        else:
            models["Elastic Net"] = (ElasticNetCV(cv=cv_folds, l1_ratio=config['elastic_net'].get('l1_ratio', 0.5), random_state=random_seed), "#2ca02c")

    if config['random_forest']['run']:
        if grid_search_enabled:
            base_model = RandomForestRegressor(random_state=random_seed)
            grid = config['random_forest'].get('param_grid', {})
            models["Random Forest"] = (GridSearchCV(base_model, grid, cv=cv_folds, scoring='neg_mean_squared_error', n_jobs=-1), "#d62728")
        else:
            models["Random Forest"] = (RandomForestRegressor(n_estimators=config['random_forest']['n_estimators'], max_depth=config['random_forest']['max_depth'], random_state=random_seed), "#d62728")

    if config['xgboost']['run']:
        if grid_search_enabled:
            base_model = XGBRegressor(random_state=random_seed)
            grid = config['xgboost'].get('param_grid', {})
            models["XGBoost"] = (GridSearchCV(base_model, grid, cv=cv_folds, scoring='neg_mean_squared_error', n_jobs=-1), "#9467bd")
        else:
            models["XGBoost"] = (XGBRegressor(n_estimators=config['xgboost']['n_estimators'], learning_rate=config['xgboost']['learning_rate'], max_depth=config['xgboost']['max_depth'], random_state=random_seed), "#9467bd")

    for name, (model, color) in models.items():
        print(f"  Training {name}...")
        model.fit(X_train_s, y_train)
        
        if grid_search_enabled and hasattr(model, 'best_params_'):
            print(f"    [Tuned] Best params for {name}: {model.best_params_}")
            
        preds = model.predict(X_test_s)
        results_df[f"{name}_BA"] = preds
        plot_ba_vs_ca(results_df["Chronological_Age"], preds, f"{name} (Test Set)", out_dir / f"plot_ml_{name.replace(' ', '_').lower()}.png", color)

    results_df.to_csv(out_dir / "ml_predictions_test_set.csv", index=True)
    return results_df

#the main func
def main():
    config = load_config()

    out_dir = Path(config['output']['output_dir'])

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(config['data']['input_file'])

    age_col, biom_cols = config['data']['age_col'], config['data']['biomarkers']
    
    for c in [age_col] + biom_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    df = df.dropna(subset=[age_col]).reset_index(drop=True)

    df = apply_imputation(df, biom_cols, config['data']['imputation_method'])

    #here we split the entire dataset as user defined % for ALL models
    df_train, df_test = train_test_split(
        df, 
        test_size=config['split']['test_size'], 
        random_state=config['split']['random_state']
    )

    df_kdm, df_pca, df_ml = None, None, None

    #if KDM is on run it
    if config['classical_models']['kdm']['run']:
        df_kdm = run_kdm(df_train, df_test, age_col, biom_cols, config['classical_models']['kdm'], out_dir)

    #run pca dubina if on
    if config['classical_models']['pca_dubina']['run']:
        df_pca = run_pca_dubina(df_train, df_test, age_col, biom_cols, out_dir)

    #run the ml models if user turned on
    if config['ml_models']['run_ml']:
        df_ml = run_ml_models(df_train, df_test, age_col, biom_cols, config['ml_models'], config['split'], out_dir)

    #run the ensemble to get a combined results
    if config.get('ensemble', {}).get('run'):
        print(f"Running Ensemble ({config['ensemble']['method']})...")
        
        ensemble_df = pd.DataFrame({"Chronological_Age": df_test[age_col]}, index=df_test.index)
        pred_cols = []
        
        if df_kdm is not None:
            ensemble_df["KDM_BA"] = df_kdm["KDM_BA"]
            pred_cols.append("KDM_BA")
        if df_pca is not None:
            ensemble_df["PCA_BA"] = df_pca["PCA_BA"]
            pred_cols.append("PCA_BA")
        if df_ml is not None:
            for c in [col for col in df_ml.columns if col != "Chronological_Age"]:
                ensemble_df[c] = df_ml[c]
                pred_cols.append(c)

        if pred_cols:
            if config['ensemble']['method'] == "median":
                ensemble_df["Ensemble_BA"] = ensemble_df[pred_cols].median(axis=1)
            elif config['ensemble']['method'] == "mean":
                ensemble_df["Ensemble_BA"] = ensemble_df[pred_cols].mean(axis=1)
            else:
                raise ValueError("Ensemble method must be 'mean' or 'median'.")

            ensemble_df.to_csv(out_dir / "ensemble_predictions.csv", index=False)
            plot_ba_vs_ca(
                ensemble_df["Chronological_Age"], 
                ensemble_df["Ensemble_BA"], 
                f"Ensemble ({config['ensemble']['method'].capitalize()})", 
                out_dir / "plot_ensemble.png", 
                "#e377c2"
            )

    print(f"\nPipeline complete! All results saved to the '{out_dir}' directory.")


if __name__ == "__main__":
    main()
