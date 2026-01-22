import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvLayer(nn.Module):
    """
    A custom Graph Convolution layer that updates node features based on edge features 
    and neighbor node features.
    
    Update rule:
    h_i' = MLP_node(h_i || aggregation(h_j, e_ij))
    """
    def __init__(self, node_dim=128, edge_dim=64):
        super(GraphConvLayer, self).__init__()
        
        # Message function: combines neighbor node and edge feature
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(node_dim, node_dim)
        )
        
        # Update function: combines old node state and aggregated message
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + node_dim, node_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(node_dim, node_dim)
        )

    def forward(self, node_feats, edge_index, edge_feats):
        """
        node_feats: (N, node_dim)
        edge_index: (2, E)
        edge_feats: (E, edge_dim)
        """
        src = edge_index[0]
        dst = edge_index[1]
        
        # 1. Compute messages
        # msg_ij = MLP(h_j || e_ij)
        neighbor_node_feats = node_feats[dst] # Features of the 'destination' (neighbor)
        msg_input = torch.cat([neighbor_node_feats, edge_feats], dim=1)
        messages = self.msg_mlp(msg_input) # (E, node_dim)
        
        # 2. Aggregate messages (scatter add/mean)
        # We need to sum messages for each source node 'i'
        # Since standard scatter_add is not always straightforward without extra libs in older pytorch,
        # we can use index_add_ or a dense matrix if N is small.
        # But for N~100, a simple loop or index_add is fine. 
        # Here we use index_add_:
        
        aggregated = torch.zeros_like(node_feats)
        aggregated.index_add_(0, src, messages) # Sum messages into the source node
        
        # 3. Update node states
        update_input = torch.cat([node_feats, aggregated], dim=1)
        new_node_feats = self.update_mlp(update_input)
        
        return new_node_feats

class EdgeUpdateLayer(nn.Module):
    """
    Updates edge features based on the features of the connected nodes.
    e_ij' = MLP(h_i || h_j || e_ij)
    """
    def __init__(self, node_dim=128, edge_dim=64):
        super(EdgeUpdateLayer, self).__init__()
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, edge_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(edge_dim, edge_dim)
        )
        
    def forward(self, node_feats, edge_index, edge_feats):
        src = edge_index[0]
        dst = edge_index[1]
        
        h_i = node_feats[src]
        h_j = node_feats[dst]
        
        inp = torch.cat([h_i, h_j, edge_feats], dim=1)
        new_edge_feats = self.edge_mlp(inp)
        
        return new_edge_feats

class TipGNN(nn.Module):
    def __init__(self, visual_dim=256, spatial_dim=3, hidden_dim=256, edge_dim=128, num_layers=3):
        super(TipGNN, self).__init__()
        
        # Encoders
        self.node_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(spatial_dim, edge_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(edge_dim, edge_dim),
            nn.LayerNorm(edge_dim)
        )
        
        # Message Passing Layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GraphConvLayer(hidden_dim, edge_dim))
            self.layers.append(EdgeUpdateLayer(hidden_dim, edge_dim))
            
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, node_visuals, edge_index, edge_spatials):
        """
        node_visuals: (N, 256)
        edge_index: (2, E) - src, dst
        edge_spatials: (E, 3) - dx, dy, dist
        
        Returns: 
            edge_probs: (E, 1) probability of each edge being a 'match'
        """
        # 1. Encode
        h = self.node_encoder(node_visuals) # (N, 128)
        e = self.edge_encoder(edge_spatials) # (E, 64)
        
        # 2. Iterate
        for i in range(0, len(self.layers), 2):
            gconv = self.layers[i]
            eupd = self.layers[i+1]
            
            h = gconv(h, edge_index, e) + h # Residual connection
            e = eupd(h, edge_index, e) + e  # Residual connection
            
        # 3. Classify edges
        # Classifier takes: h_src || h_dst || e_final
        src = edge_index[0]
        dst = edge_index[1]
        
        h_src = h[src]
        h_dst = h[dst]
        
        cls_input = torch.cat([h_src, h_dst, e], dim=1)
        probs = self.classifier(cls_input)
        
        return probs
