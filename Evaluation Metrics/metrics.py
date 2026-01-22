def compute_comprehensive_metrics(y_true, y_pred, y_probs):
    """Compute all evaluation metrics"""
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    if len(precision_per_class) >= 2:
        metrics['precision_class0'] = precision_per_class[0]
        metrics['precision_class1'] = precision_per_class[1]
        metrics['recall_class0'] = recall_per_class[0]
        metrics['recall_class1'] = recall_per_class[1]
        metrics['f1_class0'] = f1_per_class[0]
        metrics['f1_class1'] = f1_per_class[1]
    
    # AUC
    try:
        metrics['auc'] = roc_auc_score(y_true, y_probs)
    except:
        metrics['auc'] = 0.0
    
    # Confusion matrix metrics
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # Additional metrics
    metrics['cohens_kappa'] = cohen_kappa_score(y_true, y_pred)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    return metrics
