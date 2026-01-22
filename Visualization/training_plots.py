def plot_training_curves_live(history, save_path, current_epoch):
    """Plot and update training curves every epoch"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'HBG-MACN Training - Epoch {current_epoch}', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Total Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Classification Loss
    axes[0, 1].plot(epochs, history['train_cls_loss'], 'b-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Classification Loss')
    axes[0, 1].set_title('Classification Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Contrastive Loss
    axes[0, 2].plot(epochs, history['train_cont_loss'], 'g-', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Contrastive Loss')
    axes[0, 2].set_title('Contrastive Loss')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1, 0].plot(epochs, history['train_accuracy'], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, history['val_accuracy'], 'r-', label='Val', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 1].plot(epochs, history['train_f1'], 'b-', label='Train', linewidth=2)
    axes[1, 1].plot(epochs, history['val_f1'], 'r-', label='Val', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # AUC
    axes[1, 2].plot(epochs, history['val_auc'], 'r-', linewidth=2)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('AUC')
    axes[1, 2].set_title('ROC AUC')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(labels, predictions, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'Malignant'],
                yticklabels=['Healthy', 'Malignant'])
    plt.xlabel('Predicted', fontweight='bold')
    plt.ylabel('True', fontweight='bold')
    plt.title('Confusion Matrix - HBG-MACN', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to: {save_path}")


def plot_memory_prototypes(model, save_path):
    """Visualize memory prototypes"""
    from sklearn.manifold import TSNE
    
    prototypes = model.memory_bank.prototypes.cpu().numpy()
    
    tsne = TSNE(n_components=2, random_state=42)
    protos_2d = tsne.fit_transform(prototypes)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(protos_2d[:, 0], protos_2d[:, 1], c='blue', alpha=0.6, s=100)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('Memory Bank Prototypes (t-SNE)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Prototypes saved to: {save_path}")


def print_detailed_metrics(metrics, phase="Validation"):
    """Print detailed metrics table"""
    print(f"\n{'='*80}")
    print(f"{phase} - DETAILED METRICS")
    print(f"{'='*80}")
    print(f"{'Metric':<30} {'Value':<15}")
    print(f"{'-'*80}")
    print(f"{'Accuracy':<30} {metrics['accuracy']:.4f}")
    print(f"{'Balanced Accuracy':<30} {metrics['balanced_accuracy']:.4f}")
    print(f"{'Precision':<30} {metrics['precision']:.4f}")
    print(f"{'Recall (Sensitivity)':<30} {metrics['recall']:.4f}")
    print(f"{'Specificity':<30} {metrics.get('specificity', 0):.4f}")
    print(f"{'F1 Score':<30} {metrics['f1']:.4f}")
    print(f"{'AUC':<30} {metrics['auc']:.4f}")
    print(f"{'MCC':<30} {metrics['mcc']:.4f}")
    print(f"{'Cohen Kappa':<30} {metrics['cohens_kappa']:.4f}")
    print(f"{'-'*80}")
    if 'true_positives' in metrics:
        print(f"{'True Positives':<30} {metrics['true_positives']}")
        print(f"{'True Negatives':<30} {metrics['true_negatives']}")
        print(f"{'False Positives':<30} {metrics['false_positives']}")
        print(f"{'False Negatives':<30} {metrics['false_negatives']}")
    print(f"{'='*80}\n")