import os
import psutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.build_dataset import create_final_gnn_dataset, generate_mock_csv, DEFAULT_RAW_FILES
from models.tgcn_models import SequentialWrapper
from utils.dataset import SolarSpatialTemporalDataset
from utils.graph_utils import compute_correlation_adjacency
from utils.metrics import calculate_comprehensive_metrics
from utils.plotting import generate_benchmark_plots

# Optimize thread concurrency
physical_cores = psutil.cpu_count(logical=False)
torch.set_num_threads(physical_cores)

def main():
    csv_file = "final_solargrid_2year_dataset.csv"
    sequence_length = 12
    batch_size = 1024  
    hidden_dim = 64
    epochs = 25  
    learning_rate = 0.001 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if the consolidated CSV exists; if not, build it from the 6 raw files
    if not os.path.exists(csv_file):
        raw_files_exist = any(os.path.exists(f) for f in DEFAULT_RAW_FILES)
        
        if raw_files_exist:
            print("Raw BP Solar CSV files detected! Building consolidated dataset...")
            create_final_gnn_dataset(file_paths=DEFAULT_RAW_FILES, output_csv_path=csv_file)
        else:
            print("Raw files not found. Falling back to synthetic mock data...")
            generate_mock_csv(csv_file, timesteps=8000)
        
    print("\n--- Initializing Data Loaders ---")
    dataset_train = SolarSpatialTemporalDataset(csv_file, sequence_length=sequence_length, mode='train')
    dataset_test = SolarSpatialTemporalDataset(csv_file, sequence_length=sequence_length, mode='val')
    
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    adj_matrix = compute_correlation_adjacency(csv_file).to(device)
    
    models = {
        'Proposed Gated T-GCN': SequentialWrapper(4, hidden_dim, 'gated', sequence_length).to(device),
        'Vanilla T-GCN': SequentialWrapper(4, hidden_dim, 'vanilla', sequence_length).to(device),
        'A3T-GCN': SequentialWrapper(4, hidden_dim, 'a3tgcn', sequence_length).to(device),
        'Standard GCN': SequentialWrapper(4, hidden_dim, 'gcn', sequence_length).to(device),
        'Standard GRU': SequentialWrapper(4, hidden_dim, 'gru', sequence_length).to(device),
        'Historical Average': SequentialWrapper(4, hidden_dim, 'ha', sequence_length).to(device)
    }
    
    loss_histories = {m: [] for m in models}
    predictions_registry = {}
    all_gates = None
    
    for name, model in models.items():
        print(f"\n[Training] Model Execution: {name}")
        if len(list(model.parameters())) > 0:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
            criterion = nn.MSELoss()
            
            for epoch in range(1, epochs + 1):
                model.train()
                epoch_loss = 0.0
                for X_b, Y_b, _ in loader_train:
                    X_b, Y_b = X_b.to(device), Y_b.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    preds = model(X_b, adj_matrix)
                    loss = criterion(preds, Y_b)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_loss += loss.item() * X_b.size(0)
                
                epoch_loss /= len(loader_train.dataset)
                loss_histories[name].append(epoch_loss)
                if epoch % 5 == 0 or epoch == epochs:
                    print(f"  Epoch [{epoch:02d}/{epochs:02d}] | Train MSE: {epoch_loss:.5f}")
        else:
            loss_histories[name] = [0.0] * epochs

        model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for X_b, _, Y_raw_b in loader_test:
                X_b = X_b.to(device)
                if name == 'Proposed Gated T-GCN':
                    preds, gates = model(X_b, adj_matrix, return_gates=True)
                    if all_gates is None:
                        all_gates = []
                    all_gates.append(gates.cpu().numpy())
                else:
                    preds = model(X_b, adj_matrix)
                
                preds_denorm = dataset_test.denormalize_predictions(preds.cpu()).numpy()
                all_preds.append(preds_denorm)
                all_trues.append(Y_raw_b.numpy())
                
        predictions_registry[name] = np.concatenate(all_preds, axis=0)
        if 'True_Targets' not in predictions_registry:
            predictions_registry['True_Targets'] = np.concatenate(all_trues, axis=0)
            
    if all_gates:
        all_gates = np.concatenate(all_gates, axis=0)

    # Print Validation Table
    y_true_raw = predictions_registry['True_Targets']
    print("\n" + "="*85 + "\nCOMPREHENSIVE STATISTICAL VALIDATION TABLE\n" + "="*85)
    for name in models.keys():
        metrics = calculate_comprehensive_metrics(y_true_raw, predictions_registry[name])
        print(f"{name:<22} | MAE: {metrics['MAE']:6.3f} kW | RMSE: {metrics['RMSE']:6.3f} kW | MAPE: {metrics['MAPE']:5.2f}% | R2: {metrics['R2']:.4f}")
    print("="*85)

    generate_benchmark_plots(y_true_raw, predictions_registry, all_gates, loss_histories, epochs)

if __name__ == '__main__':
    main()