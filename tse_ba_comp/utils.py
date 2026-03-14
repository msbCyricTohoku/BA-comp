import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)

def apply_imputation_fit_transform(df_train, df_test, cols, method):
    logger.info(f"Applying {method} imputation...")
    if method == "median":
        imputer = SimpleImputer(strategy="median")
    elif method == "mean":
        imputer = SimpleImputer(strategy="mean")
    elif method == "zero":
        imputer = SimpleImputer(strategy="constant", fill_value=0)
    elif method == "knn":
        imputer = KNNImputer(n_neighbors=5)
    else:
        raise ValueError(f"Unknown imputation method: {method}")
    
    df_train[cols] = imputer.fit_transform(df_train[cols])
    df_test[cols] = imputer.transform(df_test[cols])
    return df_train, df_test, imputer

def evaluate_and_plot(y_true, y_pred, method_name, filepath=None, color="#1f77b4"):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t, y_p = y_true[mask], y_pred[mask]
    
    if len(y_t) < 2:
        return {"Model": method_name, "Pearson_r": np.nan, "R2": np.nan, "RMSE": np.nan, "MAE": np.nan}

    r, _ = stats.pearsonr(y_t, y_p)
    r2 = r2_score(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mae = mean_absolute_error(y_t, y_p)

    if filepath:
        axis_limit = max(y_t.max(), y_p.max()) * 1.05
        plt.figure(figsize=(6, 6))
        plt.scatter(y_t, y_p, alpha=0.5, label=method_name, color=color)
        plt.plot([0, axis_limit], [0, axis_limit], 'r--', linewidth=2, label="Identity (y=x)")
        
        plt.xlim(0, axis_limit)
        plt.ylim(0, axis_limit)
        
        stats_text = f"Pearson r: {r:.3f}\nR²: {r2:.3f}\nRMSE: {rmse:.3f}\nMAE: {mae:.3f}"
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
    
    return {"Model": method_name, "Pearson_r": r, "R2": r2, "RMSE": rmse, "MAE": mae}
