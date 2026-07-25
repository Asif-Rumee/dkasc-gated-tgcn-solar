import numpy as np
from sklearn.metrics import r2_score

def calculate_comprehensive_metrics(y_true, y_pred, night_threshold=5.0):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    daytime_mask = (y_true > night_threshold) & (~np.isnan(y_true))
    mape = np.mean(np.abs((y_true[daytime_mask] - y_pred[daytime_mask]) / y_true[daytime_mask])) * 100.0 if np.sum(daytime_mask) > 0 else float('nan')
    r2 = r2_score(y_true.flatten(), y_pred.flatten())
    
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}