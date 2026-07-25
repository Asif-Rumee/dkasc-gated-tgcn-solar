import pandas as pd
import numpy as np
import torch

def compute_correlation_adjacency(csv_path: str, threshold: float = 0.3) -> torch.Tensor:
    df = pd.read_csv(csv_path).ffill().bfill()
    ap_cols = [f"Node{i}_Active_Power" for i in range(1, 7)]
    
    corr_matrix = np.nan_to_num(df[ap_cols].corr().values, nan=0.0)
    adj = np.where(np.abs(corr_matrix) >= threshold, corr_matrix, 0.0)
    adj = np.clip(adj + np.eye(6), 0, 1)
    
    row_sum = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(row_sum, -0.5, where=row_sum > 0)
    d_inv_sqrt[row_sum <= 0] = 0.0
    D_inv_sqrt = np.diag(d_inv_sqrt)
    
    return torch.tensor(D_inv_sqrt @ adj @ D_inv_sqrt, dtype=torch.float32)