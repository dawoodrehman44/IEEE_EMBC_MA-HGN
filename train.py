def train_epoch(model, loader, optimizer, device, config):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_cls_loss = 0
    total_cont_loss = 0
    total_graph_loss = 0
    total_hier_loss = 0
    all_preds = []
    all_labels = []
    
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([21.0/26.0]).to(device))
    contrastive_loss_fn = ContrastiveLoss()
    graph_smooth_loss_fn = GraphSmoothLoss()
    hierarchy_loss_fn = HierarchyConsistencyLoss()
    
    for batch in tqdm(loader, desc="Training"):
        cells = batch['cells'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(cells, labels)
        
        # Classification loss
        cls_loss = bce_loss_fn(outputs['logits'], labels.float())
        
        # CHANGE THIS SECTION - Boost classification much more
        total_loss_batch = cls_loss * 50.0  # Increased from nothing to 50x
        
        # Contrastive loss
        pooled_features = outputs['pooled_features']
        cont_loss = contrastive_loss_fn(pooled_features, model.memory_bank)
        total_loss_batch += config.CONTRASTIVE_WEIGHT * cont_loss
        
        # Graph smooth loss - REDUCE THIS
        graph_loss = graph_smooth_loss_fn(outputs['graph_features'])
        total_loss_batch += config.GRAPH_SMOOTH_WEIGHT * 0.1 * graph_loss  # Make it 10x smaller
        
        # Hierarchy consistency loss - REDUCE THIS  
        hier_loss = hierarchy_loss_fn(outputs['hierarchy_features'])
        total_loss_batch += config.HIERARCHY_WEIGHT * 0.1 * hier_loss  # Make it 10x smaller
        
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update memory
        with torch.no_grad():
            model.memory_bank.update(pooled_features)
        
        # Track metrics
        total_loss += total_loss_batch.item()
        total_cls_loss += cls_loss.item()
        total_cont_loss += cont_loss.item()
        total_graph_loss += graph_loss.item()
        total_hier_loss += hier_loss.item()
        
        preds = (torch.sigmoid(outputs['logits']) > 0.5).long()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    return {
        'loss': total_loss / len(loader),
        'cls_loss': total_cls_loss / len(loader),
        'cont_loss': total_cont_loss / len(loader),
        'graph_loss': total_graph_loss / len(loader),
        'hier_loss': total_hier_loss / len(loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
