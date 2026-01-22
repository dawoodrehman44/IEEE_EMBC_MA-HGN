class SetTransformer(nn.Module):
    """Set Transformer for cell encoding"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        self.norm2 = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        x = self.input_proj(x)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        out = self.ffn(x)
        out = self.norm2(out)
        return out


class MemoryBank(nn.Module):
    """Memory bank for prototype storage"""
    
    def __init__(self, num_prototypes, embedding_dim, momentum=0.999):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.momentum = momentum
        self.register_buffer('prototypes', torch.randn(num_prototypes, embedding_dim))
        self.prototypes = F.normalize(self.prototypes, dim=1)
        
    @torch.no_grad()
    def update(self, embeddings):
        embeddings = F.normalize(embeddings, dim=1)
        sim = torch.mm(embeddings, self.prototypes.t())
        _, indices = sim.max(dim=1)
        
        for i in range(self.num_prototypes):
            mask = (indices == i)
            if mask.sum() > 0:
                new_proto = embeddings[mask].mean(dim=0)
                new_proto = F.normalize(new_proto, dim=0)
                self.prototypes[i] = self.momentum * self.prototypes[i] + \
                                    (1 - self.momentum) * new_proto
                self.prototypes[i] = F.normalize(self.prototypes[i], dim=0)


class GraphConvLayer(nn.Module):
    """Graph convolution for cell neighborhood modeling"""
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, x, edge_index):
        row, col = edge_index
        deg = torch.bincount(row, minlength=x.size(0)).float().clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)
        
        out = torch.zeros_like(x)
        out.index_add_(0, row, x[col])
        out = out * deg_inv_sqrt.unsqueeze(1)
        
        out = self.linear(out)
        out = self.norm(out)
        return F.relu(out)


class HierarchicalAggregation(nn.Module):
    """Hierarchical aggregation following biological structure"""
    
    def __init__(self, cell_dim, num_levels=4):
        super().__init__()
        self.num_levels = num_levels
        
        self.level_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cell_dim, cell_dim),
                nn.ReLU(),
                nn.LayerNorm(cell_dim)
            ) for _ in range(num_levels)
        ])
        
    def forward(self, cell_features):
        batch_size, num_cells, cell_dim = cell_features.shape
        hierarchy_features = []
        
        # Level 1: Individual cells
        level1 = cell_features.mean(dim=1)
        hierarchy_features.append(self.level_encoders[0](level1))
        
        # Level 2: Cell clusters
        cells_per_cluster = num_cells // 4
        level2_list = []
        for i in range(4):
            start_idx = i * cells_per_cluster
            end_idx = start_idx + cells_per_cluster if i < 3 else num_cells
            cluster_feat = cell_features[:, start_idx:end_idx, :].mean(dim=1)
            level2_list.append(cluster_feat)
        level2 = torch.stack(level2_list, dim=1).mean(dim=1)
        hierarchy_features.append(self.level_encoders[1](level2))
        
        # Level 3: Lineages
        mid_point = num_cells // 2
        lineage1 = cell_features[:, :mid_point, :].mean(dim=1)
        lineage2 = cell_features[:, mid_point:, :].mean(dim=1)
        level3 = (lineage1 + lineage2) / 2
        hierarchy_features.append(self.level_encoders[2](level3))
        
        # Level 4: Sample level
        level4 = cell_features.mean(dim=1)
        hierarchy_features.append(self.level_encoders[3](level4))
        
        return torch.stack(hierarchy_features, dim=1)


class AdaptiveMultiScaleFusion(nn.Module):
    """Adaptive fusion with attention over scales"""
    
    def __init__(self, cell_dim, num_scales):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(cell_dim, cell_dim // 2),
            nn.ReLU(),
            nn.Linear(cell_dim // 2, 1)
        )
        
    def forward(self, multi_scale_features):
        attn_scores = self.attention(multi_scale_features).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)
        fused = (multi_scale_features * attn_weights).sum(dim=1)
        return fused, attn_weights.squeeze(-1)


class HBGMACN(nn.Module):
    """Hierarchical Biological Graph Memory-Augmented Contrastive Network"""
    
    def __init__(self, input_dim, cell_dim=128, population_dim=64,
                 num_prototypes=200, num_heads=4, k_neighbors=10,
                 hierarchy_levels=4):
        super().__init__()
        
        self.k_neighbors = k_neighbors
        
        # Component 1: Cell-level encoding
        self.cell_encoder = SetTransformer(input_dim, cell_dim, cell_dim, num_heads)
        
        # Component 2: Memory bank
        self.memory_bank = MemoryBank(num_prototypes, cell_dim)
        
        # Component 3: Graph network
        self.graph_conv1 = GraphConvLayer(cell_dim, cell_dim)
        self.graph_conv2 = GraphConvLayer(cell_dim, cell_dim)
        
        # Component 4: Hierarchical aggregation
        self.hierarchy = HierarchicalAggregation(cell_dim, hierarchy_levels)
        self.fusion = AdaptiveMultiScaleFusion(cell_dim, hierarchy_levels + 1)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(cell_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
        self.classifier[-1].bias.data.fill_(-0.215)
        
    def forward(self, x, label=None):
        batch_size, num_cells, _ = x.shape
        
        # Encode cells
        cell_embeddings = self.cell_encoder(x)
        
        # Graph processing
        cell_flat = cell_embeddings.reshape(-1, cell_embeddings.size(-1))
        
        edge_indices = []
        for i in range(batch_size):
            start_idx = i * num_cells
            end_idx = start_idx + num_cells
            sample_cells = cell_embeddings[i]
            
            dist = torch.cdist(sample_cells, sample_cells)
            _, knn_idx = torch.topk(dist, k=self.k_neighbors + 1, largest=False, dim=1)
            knn_idx = knn_idx[:, 1:]
            
            src = torch.arange(num_cells, device=x.device).unsqueeze(1).repeat(1, self.k_neighbors)
            dst = knn_idx
            edges = torch.stack([src.flatten(), dst.flatten()], dim=0)
            edges = edges + start_idx
            edge_indices.append(edges)
        
        edge_index = torch.cat(edge_indices, dim=1)
        
        graph_features = self.graph_conv1(cell_flat, edge_index)
        graph_features = self.graph_conv2(graph_features, edge_index)
        graph_features = graph_features.reshape(batch_size, num_cells, -1)
        
        # Hierarchical aggregation
        hierarchy_features = self.hierarchy(graph_features)
        
        # Add cell-level features
        cell_pooled = graph_features.mean(dim=1).unsqueeze(1)
        all_features = torch.cat([cell_pooled, hierarchy_features], dim=1)
        
        # Adaptive fusion
        fused_features, attn_weights = self.fusion(all_features)
        
        # Classification
        logits = self.classifier(fused_features).squeeze(1)
        
        return {
            'logits': logits,
            'cell_embeddings': cell_embeddings,
            'graph_features': graph_features,
            'hierarchy_features': hierarchy_features,
            'attn_weights': attn_weights,
            'pooled_features': fused_features
        }