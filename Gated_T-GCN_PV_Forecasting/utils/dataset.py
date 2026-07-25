import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class SolarSpatialTemporalDataset(Dataset):
    """Pre-tensorized sliding window dataset."""
    def __init__(self, csv_path: str, sequence_length: int = 12, mode: str = 'train', split_ratio: float = 0.8):
        self.sequence_length = sequence_length
        df = pd.read_csv(csv_path).ffill().bfill()
        
        feature_cols = [col for col in df.columns if col != 'Timestamp']
        raw_data = np.nan_to_num(df[feature_cols].values, nan=0.0)
        
        self.num_nodes = 6
        self.num_features = 4  
        structured_data = raw_data.reshape(-1, self.num_nodes, self.num_features)
        
        split_idx = int(len(structured_data) * split_ratio)
        train_split_data = structured_data[:split_idx]
        
        self.mean = np.mean(train_split_data, axis=(0, 1), keepdims=True)
        self.std = np.std(train_split_data, axis=(0, 1), keepdims=True) + 1e-8
        
        normalized_all = (structured_data - self.mean) / self.std
        
        if mode == 'train':
            data_partition = normalized_all[:split_idx]
            raw_partition = structured_data[:split_idx]
        else:
            data_partition = normalized_all[split_idx:]
            raw_partition = structured_data[split_idx:]

        self.num_samples = len(data_partition) - self.sequence_length
        
        data_partition_torch = torch.from_numpy(data_partition).float()
        raw_partition_torch = torch.from_numpy(raw_partition).float()
        
        self.X_tensors = data_partition_torch.unfold(0, self.sequence_length, 1).permute(0, 3, 1, 2)[:self.num_samples].contiguous()
        self.Y_tensors = data_partition_torch[self.sequence_length : self.sequence_length + self.num_samples, :, 0:1].contiguous()
        self.Y_raw_tensors = raw_partition_torch[self.sequence_length : self.sequence_length + self.num_samples, :, 0:1].contiguous()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X_tensors[idx], self.Y_tensors[idx], self.Y_raw_tensors[idx]

    def denormalize_predictions(self, pred_tensor):
        return (pred_tensor * self.std[0, 0, 0]) + self.mean[0, 0, 0]