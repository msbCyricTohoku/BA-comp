import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import logging
from .utils import evaluate_and_plot

logger = logging.getLogger(__name__)

def run_ml_models(df_train, df_test, age_col, biom_cols, ml_arg=True, tune=False, cv_folds=5, seed=42, out_dir=None):
    logger.info("Setting up Machine Learning Models...")
    
    # 1. Normalize the ml_arg into a dictionary of parameters
    ml_params = {}
    if ml_arg is True:
        ml_params = {"en": {}, "rf": {}, "xgb": {}}
    elif isinstance(ml_arg, list):
        ml_params = {model_name: {} for model_name in ml_arg}
    elif isinstance(ml_arg, dict):
        ml_params = ml_arg
    elif ml_arg is False or ml_arg is None:
        return pd.DataFrame(), []

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(df_train[biom_cols])
    X_test_s = scaler.transform(df_test[biom_cols])
    y_train = df_train[age_col]
    
    results_df = pd.DataFrame({"Chronological_Age": df_test[age_col]}, index=df_test.index)
    ml_metrics = []
    models = {}
    
    # 2. Setup Elastic Net
    if "en" in ml_params:
        if tune:
            param_grid = {"alpha": [0.01, 0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.5, 0.7, 0.9, 1.0]}
            models["Elastic Net"] = (GridSearchCV(ElasticNet(random_state=seed), param_grid, cv=cv_folds, scoring='neg_mean_squared_error', n_jobs=-1), "#2ca02c")
        else:
            l1 = ml_params["en"].get("l1_ratio", 0.5)
            models["Elastic Net"] = (ElasticNetCV(cv=cv_folds, l1_ratio=l1, random_state=seed, n_jobs=-1), "#2ca02c")

    # 3. Setup Random Forest
    if "rf" in ml_params:
        if tune:
            param_grid = {"n_estimators": [100, 200, 300], "max_depth": [None, 5, 10, 20]}
            models["Random Forest"] = (GridSearchCV(RandomForestRegressor(random_state=seed), param_grid, cv=cv_folds, scoring='neg_mean_squared_error', n_jobs=-1), "#d62728")
        else:
            n_est = ml_params["rf"].get("n_estimators", 100)
            max_d = ml_params["rf"].get("max_depth", None)
            models["Random Forest"] = (RandomForestRegressor(n_estimators=n_est, max_depth=max_d, random_state=seed, n_jobs=-1), "#d62728")

    # 4. Setup XGBoost
    if "xgb" in ml_params:
        if tune:
            param_grid = {"n_estimators": [100, 200, 300], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 6, 9]}
            models["XGBoost"] = (GridSearchCV(XGBRegressor(random_state=seed), param_grid, cv=cv_folds, scoring='neg_mean_squared_error', n_jobs=-1), "#9467bd")
        else:
            n_est = ml_params["xgb"].get("n_estimators", 100)
            lr = ml_params["xgb"].get("learning_rate", 0.1)
            max_d = ml_params["xgb"].get("max_depth", 6)
            models["XGBoost"] = (XGBRegressor(n_estimators=n_est, learning_rate=lr, max_depth=max_d, random_state=seed, n_jobs=-1), "#9467bd")

    # Fit and Evaluate
    for name, (model, color) in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train_s, y_train)
        
        if tune and hasattr(model, 'best_params_'):
            logger.info(f"[Tuned] Best params for {name}: {model.best_params_}")
            
        preds = model.predict(X_test_s)
        results_df[f"{name}_BA"] = preds
        
        plot_path = out_dir / f"plot_ml_{name.replace(' ', '_').lower()}.png" if out_dir else None
        metrics = evaluate_and_plot(results_df["Chronological_Age"], preds, f"{name} (Test Set)", plot_path, color)
        ml_metrics.append(metrics)

    return results_df, ml_metrics
