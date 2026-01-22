def validate(model, loader, device):
    """Validate the model"""
    model.eval()
    
    total_loss = 0  # ADD THIS
    all_preds = []
    all_probs = []
    all_labels = []
    all_sample_ids = []
    
    bce_loss_fn = nn.BCEWithLogitsLoss()  # ADD THIS
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            cells = batch['cells'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(cells, labels)
            
            # ADD THIS: Calculate validation loss
            loss = bce_loss_fn(outputs['logits'], labels.float())
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs['logits'])
            preds = (probs > 0.5).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_sample_ids.extend(batch['sample_ids'])
    
    # Compute comprehensive metrics
    metrics = compute_comprehensive_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = total_loss / len(loader)  # ADD THIS
    metrics['predictions'] = all_preds
    metrics['probabilities'] = all_probs
    metrics['labels'] = all_labels
    metrics['sample_ids'] = all_sample_ids
    
    # Print prediction distribution
    unique_preds, counts = np.unique(all_preds, return_counts=True)
    print(f"\n[DEBUG] Prediction distribution: {dict(zip(unique_preds, counts))}")
    print(f"[DEBUG] Label distribution: {dict(zip(*np.unique(all_labels, return_counts=True)))}")
    
    return metrics