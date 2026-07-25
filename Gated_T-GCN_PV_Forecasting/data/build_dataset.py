import os
import gc
import pandas as pd
import numpy as np

# Target DKASC BP Solar raw data files
DEFAULT_RAW_FILES = [
    "16A_BP_Solar_North.csv",
    "16B_BP_Solar_Flat.csv",
    "16C_BP_Solar_East.csv",
    "16D_BP_Solar_West.csv",
    "3_BP_Solar_Roof.csv",
    "11_BP_Solar_Ground.csv"
]


def detect_csv_structure(file_path):
    """Scans raw DKASC files to auto-detect header indices, delimiters, and feature columns."""
    delimiters = [',', ';', '\t']
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for row_idx in range(50):
            line = f.readline()
            if not line:
                break
            line = line.rstrip('\r\n')
                
            for sep in delimiters:
                raw_columns = line.split(sep)
                cleaned_columns = [col.strip().strip('"').strip("'").lower() for col in raw_columns]
                
                has_ts = any('timestamp' in c or 'date' in c or 'time' in c for c in cleaned_columns)
                has_power = any('power' in c or 'kw' in c for c in cleaned_columns)
                
                if has_ts and has_power:
                    ts_col = next((raw for raw, clean in zip(raw_columns, cleaned_columns) if 'timestamp' in clean or 'date' in clean or 'time' in clean), None)
                    ae_col = next((raw for raw, clean in zip(raw_columns, cleaned_columns) if 'energy' in clean), None)
                    cp_col = next((raw for raw, clean in zip(raw_columns, cleaned_columns) if 'current' in clean or 'phase' in clean), None)
                    p_col = next((raw for raw, clean in zip(raw_columns, cleaned_columns) if any(k in clean for k in ['active power', 'power (kw)', 'power', 'kw'])), None)
                    ws_col = next((raw for raw, clean in zip(raw_columns, cleaned_columns) if 'wind_speed' in clean or 'wind speed' in clean or 'ws' in clean), None)
                    t_col = next((raw for raw, clean in zip(raw_columns, cleaned_columns) if 'temperature' in clean or 'temp' in clean or 'celsius' in clean), None)
                    rh_col = next((raw for raw, clean in zip(raw_columns, cleaned_columns) if 'humidity' in clean or 'relative' in clean or 'rh' in clean), None)
                    g_col = next((raw for raw, clean in zip(raw_columns, cleaned_columns) if any(k in clean for k in ['global_horizontal', 'horizontal radiation', 'ghi', 'radiation'])), None)
                    dh_col = next((raw for raw, clean in zip(raw_columns, cleaned_columns) if 'diffuse_horizontal' in clean or 'dhi' in clean), None)
                    wd_col = next((raw for raw, clean in zip(raw_columns, cleaned_columns) if 'direction' in clean or 'wind_dir' in clean or 'wd' in clean), None)
                    r_col = next((raw for raw, clean in zip(raw_columns, cleaned_columns) if 'rainfall' in clean or 'rain' in clean), None)
                    gt_col = next((raw for raw, clean in zip(raw_columns, cleaned_columns) if 'tilted' in clean and ('global' in clean or 'gti' in clean)), None)
                    dt_col = next((raw for raw, clean in zip(raw_columns, cleaned_columns) if 'tilted' in clean and 'diffuse' in clean), None)
                    
                    return row_idx, sep, ts_col, ae_col, cp_col, p_col, ws_col, t_col, rh_col, g_col, dh_col, wd_col, r_col, gt_col, dt_col
                    
    raise ValueError(f"Could not locate valid header layout in {os.path.basename(file_path)}.")


def create_final_gnn_dataset(file_paths=None, output_csv_path="final_solargrid_2year_dataset.csv", start_date="2024-01-01 00:00:00", end_date="2025-12-31 23:55:00", freq="5min"):
    """Extracts, cleans, and aligns the 6 BP Solar array CSV files into a consolidated GNN timeline."""
    if file_paths is None:
        file_paths = DEFAULT_RAW_FILES

    print(f"Setting up uniform timeline baseline ({start_date} to {end_date})...")
    
    # Flexible datetime parsing to handle multi-format DKASC outputs
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    final_df = pd.DataFrame(index=pd.date_range(start=start_dt, end=end_dt, freq=freq))
    final_df.index.name = 'Timestamp'
    columns_added_count = 0

    for node_idx, file_path in enumerate(file_paths):
        node_num = node_idx + 1
        print(f"Processing Node {node_num} [{os.path.basename(file_path)})...")
        
        if not os.path.exists(file_path):
            print(f"--> File not found: {file_path}. Skipping.")
            continue
            
        (row_skip, delimiter, ts_col, ae_col, cp_col, p_col, ws_col, t_col, 
         rh_col, g_col, dh_col, wd_col, r_col, gt_col, dt_col) = detect_csv_structure(file_path)
        
        raw_cols_list = [ts_col, ae_col, cp_col, p_col, ws_col, t_col, rh_col, g_col, dh_col, wd_col, r_col, gt_col, dt_col]
        all_needed_cols = [c for c in raw_cols_list if c is not None]
        numeric_feature_cols = [c for c in raw_cols_list[1:] if c is not None]
        
        chunks = []
        for chunk in pd.read_csv(file_path, sep=delimiter, skiprows=row_skip, usecols=all_needed_cols, chunksize=100000, low_memory=False):
            chunk[ts_col] = pd.to_datetime(chunk[ts_col], errors='coerce')
            mask = (chunk[ts_col] >= start_dt) & (chunk[ts_col] <= end_dt)
            filtered_chunk = chunk.loc[mask]
            if not filtered_chunk.empty:
                chunks.append(filtered_chunk)
        
        if not chunks:
            print(f"--> No valid rows found in timeframe for {file_path}.")
            continue
            
        node_df = pd.concat(chunks, axis=0).drop_duplicates(subset=[ts_col]).set_index(ts_col)
        del chunks
        
        for col in numeric_feature_cols:
            node_df[col] = pd.to_numeric(node_df[col], errors='coerce')
            
        node_df = node_df.reindex(final_df.index).ffill().bfill().astype(np.float32)
        
        # Standardize Node feature names for graph construction
        clean_column_mapping = {}
        if ae_col: clean_column_mapping[ae_col] = f'Node{node_num}_Active_Energy_Delivered_Received'
        if cp_col: clean_column_mapping[cp_col] = f'Node{node_num}_Current_Phase_Average'
        if p_col:  clean_column_mapping[p_col]  = f'Node{node_num}_Active_Power'
        if ws_col: clean_column_mapping[ws_col] = f'Node{node_num}_Wind_Speed'
        if t_col:  clean_column_mapping[t_col]  = f'Node{node_num}_Weather_Temperature_Celsius'
        if rh_col: clean_column_mapping[rh_col] = f'Node{node_num}_Weather_Relative_Humidity'
        if g_col:  clean_column_mapping[g_col]  = f'Node{node_num}_Global_Horizontal_Radiation'
        if dh_col: clean_column_mapping[dh_col] = f'Node{node_num}_Diffuse_Horizontal_Radiation'
        if wd_col: clean_column_mapping[wd_col] = f'Node{node_num}_Wind_Direction'
        if r_col:  clean_column_mapping[r_col]  = f'Node{node_num}_Weather_Daily_Rainfall'
        if gt_col: clean_column_mapping[gt_col] = f'Node{node_num}_Radiation_Global_Tilted'
        if dt_col: clean_column_mapping[dt_col] = f'Node{node_num}_Radiation_Diffuse_Tilted'
        
        node_df = node_df.rename(columns=clean_column_mapping)
        final_df = final_df.join(node_df)
        columns_added_count += len(clean_column_mapping)
        del node_df
        gc.collect()

    if columns_added_count > 0:
        final_df.to_csv(output_csv_path)
        print(f"Consolidated dataset successfully generated: '{output_csv_path}'")
    else:
        print("Warning: No columns merged. Verify file paths and datetime overlap.")


def generate_mock_csv(filename: str = "final_solargrid_2year_dataset.csv", timesteps: int = 8000):
    """Fallback generator matching the exact capacities and orientations of the 6 BP Solar nodes."""
    np.random.seed(42)
    timestamps = pd.date_range(start="2026-01-01 00:00", periods=timesteps, freq="5min")
    data = {"Timestamp": timestamps}
    base_diurnal = np.maximum(0, np.sin(np.linspace(0, 32 * np.pi, timesteps)))
    
    # Capacities (kW): [16A: 2.0, 16B: 2.0, 16C: 2.0, 16D: 2.0, 3: 5.0, 11: 5.0]
    capacities = [2.0, 2.0, 2.0, 2.0, 5.0, 5.0]
    
    for i, cap in enumerate(capacities, start=1):
        ap_signal = base_diurnal * cap
        if i == 3:   # 16C East orientation: Shift peak earlier
            ap_signal = np.roll(ap_signal, -6)
        elif i == 4: # 16D West orientation: Shift peak later
            ap_signal = np.roll(ap_signal, 6)
            
        data[f"Node{i}_Active_Power"] = ap_signal + np.random.normal(0, 0.02 * cap, timesteps)
        data[f"Node{i}_Radiation_Global_Tilted"] = base_diurnal * 750.0 + np.random.normal(0, 12.0, timesteps)
        data[f"Node{i}_Diffuse_Horizontal_Radiation"] = base_diurnal * 200.0 + np.random.normal(0, 6.0, timesteps)
        data[f"Node{i}_Weather_Temperature_Celsius"] = base_diurnal * 15.0 + 12.0 + np.random.normal(0, 0.5, timesteps)
        
    pd.DataFrame(data).to_csv(filename, index=False)
    print(f"Fallback dataset created: '{filename}'")