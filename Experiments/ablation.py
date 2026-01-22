
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
        """
        Args:
            x: [batch*num_cells, in_dim]
            edge_index: [2, num_edges]
        """
        row, col = edge_index
        
        # Aggregate neighbors
        deg = torch.bincount(row, minlength=x.size(0)).float().clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)
        
        # Message passing
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
        """
        Args:
            cell_features: [batch, num_cells, cell_dim]
        Returns:
            hierarchy_features: [batch, num_levels, cell_dim]
        """
        batch_size, num_cells, cell_dim = cell_features.shape
        hierarchy_features = []
        
        # Level 1: Individual cells (sample randomly)
        level1 = cell_features.mean(dim=1)
        hierarchy_features.append(self.level_encoders[0](level1))
        
        # Level 2: Cell clusters (split into groups)
        cells_per_cluster = num_cells // 4
        level2_list = []
        for i in range(4):
            start_idx = i * cells_per_cluster
            end_idx = start_idx + cells_per_cluster if i < 3 else num_cells
            cluster_feat = cell_features[:, start_idx:end_idx, :].mean(dim=1)
            level2_list.append(cluster_feat)
        level2 = torch.stack(level2_list, dim=1).mean(dim=1)
        hierarchy_features.append(self.level_encoders[1](level2))
        
        # Level 3: Lineages (half and half)
        mid_point = num_cells // 2
        lineage1 = cell_features[:, :mid_point, :].mean(dim=1)
        lineage2 = cell_features[:, mid_point:, :].mean(dim=1)
        level3 = (lineage1 + lineage2) / 2
        hierarchy_features.append(self.level_encoders[2](level3))
        
        # Level 4: Sample level
        level4 = cell_features.mean(dim=1)
        hierarchy_features.append(self.level_encoders[3](level4))
        
        return torch.stack(hierarchy_features, dim=1)  # [batch, num_levels, cell_dim]


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
        """
        Args:
            multi_scale_features: [batch, num_scales, cell_dim]
        Returns:
            fused: [batch, cell_dim]
        """
        # Compute attention weights
        attn_scores = self.attention(multi_scale_features).squeeze(-1)  # [batch, num_scales]
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # [batch, num_scales, 1]
        
        # Weighted sum
        fused = (multi_scale_features * attn_weights).sum(dim=1)  # [batch, cell_dim]
        return fused, attn_weights.squeeze(-1)


# ==================== MAIN MODEL: HBG-MACN ====================

class HBGMACN(nn.Module):
    """Hierarchical Biological Graph Memory-Augmented Contrastive Network"""
    
    def __init__(self, input_dim, cell_dim=128, population_dim=64,
                 num_prototypes=100, num_heads=4, k_neighbors=10,
                 hierarchy_levels=4, use_memory=True, use_graph=True,
                 use_hierarchy=True):
        super().__init__()
        
        self.use_memory = use_memory
        self.use_graph = use_graph
        self.use_hierarchy = use_hierarchy
        self.k_neighbors = k_neighbors
        
        # Component 1: Cell-level encoding
        self.cell_encoder = SetTransformer(input_dim, cell_dim, cell_dim, num_heads)
        
        # Component 2: Memory bank
        if use_memory:
            self.memory_bank = MemoryBank(num_prototypes, cell_dim)
        
        # Component 3: Graph network
        if use_graph:
            self.graph_conv1 = GraphConvLayer(cell_dim, cell_dim)
            self.graph_conv2 = GraphConvLayer(cell_dim, cell_dim)
        
        # Component 4: Hierarchical aggregation
        if use_hierarchy:
            self.hierarchy = HierarchicalAggregation(cell_dim, hierarchy_levels)
            self.fusion = AdaptiveMultiScaleFusion(cell_dim, hierarchy_levels + 1)
            fusion_dim = cell_dim
        else:
            fusion_dim = cell_dim
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
        self.classifier[-1].bias.data.fill_(-0.215)
        
    def forward(self, x, label=None):
        """
        Args:
            x: [batch, num_cells, input_dim]
        """
        batch_size, num_cells, _ = x.shape
        
        # Encode cells
        cell_embeddings = self.cell_encoder(x)  # [batch, num_cells, cell_dim]
        
        # Graph processing
        if self.use_graph:
            # Reshape for graph processing
            cell_flat = cell_embeddings.reshape(-1, cell_embeddings.size(-1))
            
            # Build k-NN graph per sample
            edge_indices = []
            for i in range(batch_size):
                start_idx = i * num_cells
                end_idx = start_idx + num_cells
                sample_cells = cell_embeddings[i]
                
                # Compute pairwise distances
                dist = torch.cdist(sample_cells, sample_cells)
                _, knn_idx = torch.topk(dist, k=self.k_neighbors + 1, largest=False, dim=1)
                knn_idx = knn_idx[:, 1:]  # Remove self
                
                # Create edge index
                src = torch.arange(num_cells, device=x.device).unsqueeze(1).repeat(1, self.k_neighbors)
                dst = knn_idx
                edges = torch.stack([src.flatten(), dst.flatten()], dim=0)
                edges = edges + start_idx  # Offset for batch
                edge_indices.append(edges)
            
            edge_index = torch.cat(edge_indices, dim=1)
            
            # Graph convolution
            graph_features = self.graph_conv1(cell_flat, edge_index)
            graph_features = self.graph_conv2(graph_features, edge_index)
            graph_features = graph_features.reshape(batch_size, num_cells, -1)
        else:
            graph_features = cell_embeddings
        
        # Hierarchical aggregation
        if self.use_hierarchy:
            hierarchy_features = self.hierarchy(graph_features)  # [batch, levels, cell_dim]
            
            # Add cell-level features
            cell_pooled = graph_features.mean(dim=1).unsqueeze(1)  # [batch, 1, cell_dim]
            all_features = torch.cat([cell_pooled, hierarchy_features], dim=1)
            
            # Adaptive fusion
            fused_features, attn_weights = self.fusion(all_features)
        else:
            fused_features = graph_features.mean(dim=1)
            attn_weights = None
        
        # Classification
        logits = self.classifier(fused_features).squeeze(1)
        
        outputs = {
            'logits': logits,
            'cell_embeddings': cell_embeddings,
            'graph_features': graph_features if self.use_graph else None,
            'hierarchy_features': hierarchy_features if self.use_hierarchy else None,
            'attn_weights': attn_weights,
            'pooled_features': fused_features
        }
        
        return outputs


EXPERIMENTS = {
    'full_model': {
        'name': 'Full HBG-MACN',
        'description': 'Complete model with all components',
        'use_memory': True,
        'use_graph': True,
        'use_hierarchy': True,
        'contrastive_weight': 0.1,
        'graph_smooth_weight': 0.05,
        'hierarchy_weight': 0.05
    },
    
    'no_memory': {
        'name': 'No Memory Banks',
        'description': 'Remove memory-augmented contrastive learning',
        'use_memory': False,
        'use_graph': True,
        'use_hierarchy': True,
        'contrastive_weight': 0.0,
        'graph_smooth_weight': 0.05,
        'hierarchy_weight': 0.05
    },
    
    'no_graph': {
        'name': 'No Graph Network',
        'description': 'Remove graph convolution',
        'use_memory': True,
        'use_graph': False,
        'use_hierarchy': True,
        'contrastive_weight': 0.1,
        'graph_smooth_weight': 0.0,
        'hierarchy_weight': 0.05
    },
    
    'no_hierarchy': {
        'name': 'No Hierarchy',
        'description': 'Remove hierarchical aggregation',
        'use_memory': True,
        'use_graph': True,
        'use_hierarchy': False,
        'contrastive_weight': 0.1,
        'graph_smooth_weight': 0.05,
        'hierarchy_weight': 0.0
    },
    
    'memory_only': {
        'name': 'Memory Only',
        'description': 'Only memory banks, no graph or hierarchy',
        'use_memory': True,
        'use_graph': False,
        'use_hierarchy': False,
        'contrastive_weight': 0.1,
        'graph_smooth_weight': 0.0,
        'hierarchy_weight': 0.0
    },
    
    'graph_only': {
        'name': 'Graph Only',
        'description': 'Only graph network, no memory or hierarchy',
        'use_memory': False,
        'use_graph': True,
        'use_hierarchy': False,
        'contrastive_weight': 0.0,
        'graph_smooth_weight': 0.05,
        'hierarchy_weight': 0.0
    },
    
    'hierarchy_only': {
        'name': 'Hierarchy Only',
        'description': 'Only hierarchical aggregation',
        'use_memory': False,
        'use_graph': False,
        'use_hierarchy': True,
        'contrastive_weight': 0.0,
        'graph_smooth_weight': 0.0,
        'hierarchy_weight': 0.05
    },
    
    'baseline_set_transformer': {
        'name': 'Baseline Set Transformer',
        'description': 'Set Transformer with simple pooling',
        'use_memory': False,
        'use_graph': False,
        'use_hierarchy': False,
        'contrastive_weight': 0.0,
        'graph_smooth_weight': 0.0,
        'hierarchy_weight': 0.0
    }
}