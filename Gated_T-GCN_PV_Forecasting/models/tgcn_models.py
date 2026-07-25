import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedIdentityTGCNCell(nn.Module):
    """Proposed Model: Feature-Wise Spatial Identity Preservation"""
    def __init__(self, input_dim: int, hidden_dim: int):
        super(GatedIdentityTGCNCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.W_s = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_g = nn.Linear(input_dim, hidden_dim, bias=True)
        self.shortcut = nn.Linear(input_dim, hidden_dim, bias=False) if input_dim != hidden_dim else nn.Identity()
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, X: torch.Tensor, adj: torch.Tensor, H_hidden: torch.Tensor, return_gate: bool = False):
        batch_size, num_nodes, _ = X.shape
        
        X_spatial = self.W_s(X)
        H_spatial = F.relu(torch.matmul(adj, X_spatial))
        
        g = torch.sigmoid(self.W_g(X))
        H_final = g * self.shortcut(X) + (1 - g) * H_spatial
        
        H_final_flat = H_final.reshape(-1, self.hidden_dim)
        H_hidden_flat = H_hidden.reshape(-1, self.hidden_dim)
        H_next_flat = self.gru_cell(H_final_flat, H_hidden_flat)
        H_next = H_next_flat.reshape(batch_size, num_nodes, self.hidden_dim)
        
        if return_gate:
            return H_next, g
        return H_next


class VanillaTGCNCell(nn.Module):
    """Baseline: Standard T-GCN (Pure Neighborhood Blending)"""
    def __init__(self, input_dim: int, hidden_dim: int):
        super(VanillaTGCNCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.W_s = nn.Linear(input_dim, hidden_dim, bias=False)
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, X: torch.Tensor, adj: torch.Tensor, H_hidden: torch.Tensor):
        batch_size, num_nodes, _ = X.shape
        X_spatial = self.W_s(X)
        H_spatial = F.relu(torch.matmul(adj, X_spatial))
        
        H_spatial_flat = H_spatial.reshape(-1, self.hidden_dim)
        H_hidden_flat = H_hidden.reshape(-1, self.hidden_dim)
        H_next_flat = self.gru_cell(H_spatial_flat, H_hidden_flat)
        return H_next_flat.reshape(batch_size, num_nodes, self.hidden_dim)


class SequentialWrapper(nn.Module):
    """Unified wrapper supporting multiple baseline paradigms"""
    def __init__(self, input_dim: int, hidden_dim: int, model_type: str = 'gated', time_steps: int = 12):
        super(SequentialWrapper, self).__init__()
        self.hidden_dim = hidden_dim
        self.model_type = model_type
        self.time_steps = time_steps
        
        if model_type in ['gated', 'vanilla', 'a3tgcn']:
            self.cell = GatedIdentityTGCNCell(input_dim, hidden_dim) if model_type == 'gated' else VanillaTGCNCell(input_dim, hidden_dim)
            if model_type == 'a3tgcn':
                self.attn_layer = nn.Linear(hidden_dim, 1)
            self.regression = nn.Linear(hidden_dim, 1)
        elif model_type == 'gru':
            self.cell = nn.GRUCell(input_dim, hidden_dim)
            self.regression = nn.Linear(hidden_dim, 1)
        elif model_type == 'gcn':
            self.gcn_weight = nn.Linear(input_dim, hidden_dim, bias=False)
            self.regression = nn.Linear(time_steps * hidden_dim, 1)

    def forward(self, X_seq: torch.Tensor, adj: torch.Tensor, return_gates: bool = False):
        batch_size, time_steps, num_nodes, input_dim = X_seq.shape
        device = X_seq.device
        
        if self.model_type == 'ha':
            return X_seq[:, :, :, 0:1].mean(dim=1)
            
        if self.model_type == 'gcn':
            X_spatial = self.gcn_weight(X_seq)
            H_spatial = F.relu(torch.matmul(adj, X_spatial))
            H_flat_temporal = H_spatial.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, time_steps * self.hidden_dim)
            return self.regression(H_flat_temporal)
            
        if self.model_type == 'a3tgcn':
            H_hidden = torch.zeros(batch_size, num_nodes, self.hidden_dim, device=device)
            h_states = []
            for t in range(time_steps):
                H_hidden = self.cell(X_seq[:, t, :, :], adj, H_hidden)
                h_states.append(H_hidden.unsqueeze(1))
            H_seq = torch.cat(h_states, dim=1)           
            attn_weights = F.softmax(self.attn_layer(H_seq), dim=1)       
            return self.regression(torch.sum(attn_weights * H_seq, dim=1))
            
        H_hidden = torch.zeros(batch_size, num_nodes, self.hidden_dim, device=device)
        gates_history = []

        for t in range(time_steps):
            X_t = X_seq[:, t, :, :]
            if self.model_type == 'gated':
                if return_gates:
                    H_hidden, g = self.cell(X_t, adj, H_hidden, return_gate=True)
                    gates_history.append(g.unsqueeze(1))
                else:
                    H_hidden = self.cell(X_t, adj, H_hidden)
            elif self.model_type == 'vanilla':
                H_hidden = self.cell(X_t, adj, H_hidden)
            elif self.model_type == 'gru':
                X_t_flat = X_t.reshape(-1, input_dim)
                H_flat = H_hidden.reshape(-1, self.hidden_dim)
                H_hidden = self.cell(X_t_flat, H_flat).reshape(batch_size, num_nodes, self.hidden_dim)
                
        out = self.regression(H_hidden)
        if return_gates and self.model_type == 'gated':
            return out, torch.cat(gates_history, dim=1)
        return out