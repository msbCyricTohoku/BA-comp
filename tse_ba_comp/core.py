import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging
from .utils import apply_imputation_fit_transform, evaluate_and_plot
from .classical import run_kdm, run_pca_dubina
from .ml_models import run_ml_models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run(
    data,
    biomarkers,
    age='age',
    impute='median',
    test_size=0.2,
    seed=42,
    kdm=True,
    kdm_s2_floor=0.1,
    pca=True,
    ml=True,
    tune=False,
    cv_folds=5,
    ensemble='median',
    out=None
):
    """
    Master function for the tse_ba_comp library.
    """
    out_path = Path(out) if out else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading and preparing data...")
    df = pd.read_csv(data) if isinstance(data, str) else data.copy()
    
    for c in [age] + biomarkers:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    df = df.dropna(subset=[age]).reset_index(drop=True)

    #split bypass for none, important for classical methods
    if test_size is None:
        logger.info("test_size is None. Bypassing split. Evaluating on the ENTIRE dataset.")
        df_train = df.copy()
        df_test = df.copy()
    else:
        df_train, df_test = train_test_split(df, test_size=test_size, random_state=seed)

    df_train, df_test, imputer = apply_imputation_fit_transform(df_train, df_test, biomarkers, impute)

    all_metrics = []
    final_preds = pd.DataFrame({"Chronological_Age": df_test[age]}, index=df_test.index)
    pred_cols = []

    if kdm:
        p_path = out_path / "plot_kdm.png" if out_path else None
        kdm_preds, m = run_kdm(df_train, df_test, age, biomarkers, s2_ba_floor=kdm_s2_floor, out_path=p_path)
        final_preds = final_preds.join(kdm_preds["KDM_BA"])
        pred_cols.append("KDM_BA")
        all_metrics.append(m)

    if pca:
        p_path = out_path / "plot_pca_dubina.png" if out_path else None
        pca_preds, m = run_pca_dubina(df_train, df_test, age, biomarkers, out_path=p_path)
        final_preds = final_preds.join(pca_preds["PCA_BA"])
        pred_cols.append("PCA_BA")
        all_metrics.append(m)

    if ml:
        ml_preds, m_list = run_ml_models(
            df_train, df_test, age, biomarkers, 
            ml_arg=ml, tune=tune, cv_folds=cv_folds, seed=seed, out_dir=out_path
        )
        if not ml_preds.empty:
            ml_cols = [c for c in ml_preds.columns if c != "Chronological_Age"]
            final_preds = final_preds.join(ml_preds[ml_cols])
            pred_cols.extend(ml_cols)
            all_metrics.extend(m_list)

    if pred_cols and ensemble:
        logger.info(f"Running Ensemble ({ensemble})...")
        if ensemble == 'median':
            final_preds["Ensemble_BA"] = final_preds[pred_cols].median(axis=1)
        elif ensemble == 'mean':
            final_preds["Ensemble_BA"] = final_preds[pred_cols].mean(axis=1)
            
        p_path = out_path / "plot_ensemble.png" if out_path else None
        m = evaluate_and_plot(final_preds["Chronological_Age"], final_preds["Ensemble_BA"], f"Ensemble ({ensemble.capitalize()})", p_path, "#e377c2")
        all_metrics.append(m)

    metrics_df = pd.DataFrame(all_metrics)
    
    if out_path:
        final_preds.to_csv(out_path / "all_predictions_test_set.csv", index=False)
        metrics_df.to_csv(out_path / "metrics_summary.csv", index=False)
        logger.info(f"Saved results to {out_path}")

    return {"predictions": final_preds, "metrics": metrics_df}
