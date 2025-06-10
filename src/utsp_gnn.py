import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import List, Tuple
import os
import math

class GraphAttentionLayer(nn.Module):
    """그래프 어텐션 레이어"""
    
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(0.2)
        
    def forward(self, h, adj):
        Wh = self.W(h)  # [N, out_features]
        N = Wh.size(0)
        
        # Attention mechanism
        Wh1 = Wh.repeat(1, N).view(N * N, -1)
        Wh2 = Wh.repeat(N, 1)
        attention_input = torch.cat([Wh1, Wh2], dim=1)
        
        e = self.leakyrelu(self.a(attention_input).squeeze(1))
        e = e.view(N, N)
        
        # Mask with adjacency matrix
        e = e.masked_fill(adj == 0, -1e9)
        attention = F.softmax(e, dim=1)
        attention = self.dropout(attention)
        
        h_prime = torch.matmul(attention, Wh)
        return h_prime

class UTSPGNNModel(nn.Module):
    """UTSP를 위한 그래프 신경망 모델"""
    
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=64, n_layers=3):
        super().__init__()
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Tour construction components
        self.context_embedding = nn.Linear(output_dim, output_dim)
        self.decoder = nn.Linear(output_dim * 2, 1)
        
    def forward(self, coords, mask=None):
        """
        Args:
            coords: [batch_size, n_nodes, 2] - 도시 좌표
            mask: [batch_size, n_nodes] - 방문 마스크
        """
        batch_size, n_nodes, _ = coords.shape
        
        # Node features encoding
        h = self.node_encoder(coords)  # [batch_size, n_nodes, hidden_dim]
        
        # Create adjacency matrix (완전 그래프)
        adj = torch.ones(batch_size, n_nodes, n_nodes, device=coords.device)
        
        # Graph attention layers
        for gat_layer in self.gat_layers:
            h_new = []
            for i in range(batch_size):
                h_i = gat_layer(h[i], adj[i])
                h_new.append(h_i)
            h = torch.stack(h_new)
            h = F.relu(h)
        
        # Output projection
        node_embeddings = self.output_proj(h)  # [batch_size, n_nodes, output_dim]
        
        return node_embeddings
    
    def decode_tour(self, node_embeddings, start_node=0):
        """투어 디코딩"""
        batch_size, n_nodes, embed_dim = node_embeddings.shape
        tours = []
        
        for b in range(batch_size):
            embeddings = node_embeddings[b]  # [n_nodes, embed_dim]
            
            tour = [start_node]
            visited = set([start_node])
            current_embedding = embeddings[start_node]
            
            while len(tour) < n_nodes:
                # Context embedding
                context = self.context_embedding(current_embedding)
                
                # Score calculation
                scores = []
                candidates = []
                for i in range(n_nodes):
                    if i not in visited:
                        candidate_embedding = embeddings[i]
                        combined = torch.cat([context, candidate_embedding])
                        score = self.decoder(combined)
                        scores.append(score)
                        candidates.append(i)
                
                if not candidates:
                    break
                
                # Select next node
                scores = torch.stack(scores).squeeze()
                next_idx = torch.argmax(scores).item()
                next_node = candidates[next_idx]
                
                tour.append(next_node)
                visited.add(next_node)
                current_embedding = embeddings[next_node]
            
            tour.append(start_node)  # Return to start
            tours.append(tour)
        
        return tours

class UTSPGNNSolver:
    """UTSP GNN 기반 TSP 솔버"""
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UTSPGNNModel()
        self.model.to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("Warning: No pre-trained model found. Training required.")
    
    def train(self, train_coords_list, epochs=100, lr=1e-3, save_path=None):
        """자기지도학습으로 모델 훈련"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for coords in train_coords_list:
                coords_tensor = torch.FloatTensor(coords).unsqueeze(0).to(self.device)
                
                # Forward pass
                embeddings = self.model(coords_tensor)
                tours = self.model.decode_tour(embeddings)
                
                # Compute tour length
                tour = tours[0]
                tour_length = self._compute_tour_length(coords, tour)
                
                # Self-supervised loss (minimize tour length)
                loss = torch.tensor(tour_length, requires_grad=True).to(self.device)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Average Loss: {total_loss/len(train_coords_list):.4f}")
        
        if save_path:
            self.save_model(save_path)
    
    def solve(self, coords):
        """TSP 문제 해결"""
        self.model.eval()
        
        with torch.no_grad():
            coords_tensor = torch.FloatTensor(coords).unsqueeze(0).to(self.device)
            embeddings = self.model(coords_tensor)
            tours = self.model.decode_tour(embeddings)
            
            tour = tours[0]
            tour_length = self._compute_tour_length(coords, tour)
            
            return tour, tour_length