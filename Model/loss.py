class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, embeddings, memory_bank):
        embeddings = F.normalize(embeddings, dim=1)
        prototypes = F.normalize(memory_bank.prototypes, dim=1)
        sim = torch.mm(embeddings, prototypes.t()) / self.temperature
        loss = -torch.log_softmax(sim, dim=1).mean()
        return loss


class GraphSmoothLoss(nn.Module):
    """Encourages similar cells to have similar predictions"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, features, k=10):
        batch_size, num_cells, dim = features.shape
        total_loss = 0
        
        for i in range(batch_size):
            sample_features = features[i]
            dist = torch.cdist(sample_features, sample_features)
            _, knn_idx = torch.topk(dist, k=k+1, largest=False, dim=1)
            knn_idx = knn_idx[:, 1:]
            
            for j in range(num_cells):
                neighbors = sample_features[knn_idx[j]]
                diff = sample_features[j].unsqueeze(0) - neighbors
                loss = (diff ** 2).sum()
                total_loss += loss
        
        return total_loss / (batch_size * num_cells)


class HierarchyConsistencyLoss(nn.Module):
    """Ensures predictions respect biological hierarchy"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, hierarchy_features):
        num_levels = hierarchy_features.size(1)
        loss = 0
        for i in range(num_levels - 1):
            diff = hierarchy_features[:, i, :] - hierarchy_features[:, i+1, :]
            loss += (diff ** 2).mean()
        
        return loss / (num_levels - 1)
