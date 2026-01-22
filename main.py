def main():
    print(f"Using device: {Config.DEVICE}\n")
    
    # Set seeds
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    
    # Create directories
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)
    
    config_dict = Config.get_config()
    sample_data, train_ids, val_ids, marker_cols = load_data(config_dict)
    
    # Create datasets
    train_dataset = CytoPathDataset(
        sample_data, train_ids, marker_cols,
        cells_per_sample=Config.MAX_CELLS, mode='train'
    )
    val_dataset = CytoPathDataset(
        sample_data, val_ids, marker_cols,
        cells_per_sample=Config.MAX_CELLS, mode='validation'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    
    # ===== UPDATED: Use dynamic input_dim =====
    print("\nInitializing HBG-MACN model...")
    
    if Config.INPUT_DIM is None:
        raise ValueError("INPUT_DIM not set! Data loading failed.")
    
    input_dim = Config.INPUT_DIM
    print(f"Model input dimension: {input_dim} markers")
    # ==========================================
    
    model = HBGMACN(
        input_dim=input_dim,  # ← Dynamic value (60 for SDY997)
        cell_dim=Config.CELL_DIM,
        population_dim=Config.POPULATION_DIM,
        num_prototypes=Config.NUM_PROTOTYPES,
        num_heads=Config.NUM_HEADS,
        k_neighbors=Config.K_NEIGHBORS,
        hierarchy_levels=Config.HIERARCHY_LEVELS
    ).to(Config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LR,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7, verbose=True
    )
    
    # Training loop
    print("Starting training...\n")
    
    history = defaultdict(list)
    best_val_f1 = 0.0
    patience_counter = 0
    patience = 50
    
    for epoch in range(Config.EPOCHS):
        print(f"{'='*80}")
        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        print(f"{'='*80}")
        
        train_metrics = train_epoch(model, train_loader, optimizer, Config.DEVICE, Config)
        val_metrics = validate(model, val_loader, Config.DEVICE)
        
        scheduler.step(train_metrics['loss'])
        
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f} | "
              f"Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f} | "  # Now val_metrics has 'loss'
              f"Acc: {val_metrics['accuracy']:.4f} | "
              f"F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f}")
        
        # Save history
        for k, v in train_metrics.items():
            history[f'train_{k}'].append(v)
        for k, v in val_metrics.items():
            if k not in ['predictions', 'probabilities', 'labels', 'sample_ids']:
                history[f'val_{k}'].append(v)
        
        # Update plot every epoch
        plot_training_curves_live(
            history,
            os.path.join(Config.PLOTS_DIR, 'training_live.png'),
            epoch + 1
        )
        
        # Save best
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_metrics['f1'],
                'val_auc': val_metrics['auc'],
                'history': dict(history),
                'marker_cols': marker_cols,      # ← ADD THIS
                'input_dim': Config.INPUT_DIM,   # ← ADD THIS
                'config': {                       # ← ADD THIS
                    'dataset_name': Config.DATASET_NAME,
                    'num_markers': len(marker_cols),
                    'marker_names': marker_cols
                }
            }, os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth'))
            
            print(f"\n✓ Best model saved (F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    checkpoint = torch.load(os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final validation
    print("\n" + "="*80)
    print("FINAL VALIDATION RESULTS")
    print("="*80)
    val_metrics = validate(model, val_loader, Config.DEVICE)
    
    # Print detailed metrics
    print_detailed_metrics(val_metrics, "Final Validation")
    
    # Save results
    results_df = pd.DataFrame({
        'sample_id': val_metrics['sample_ids'],
        'true_label': val_metrics['labels'],
        'predicted_label': val_metrics['predictions'],
        'probability': val_metrics['probabilities']
    })
    results_df.to_csv(os.path.join(Config.RESULTS_DIR, 'predictions.csv'), index=False)
    
    # Save summary metrics
    summary_metrics = {k: v for k, v in val_metrics.items() 
                      if k not in ['predictions', 'probabilities', 'labels', 'sample_ids']}
    summary_df = pd.DataFrame([summary_metrics])
    summary_df.to_csv(os.path.join(Config.RESULTS_DIR, 'summary_metrics.csv'), index=False)
    
    # Plots
    print("\nGenerating final plots...")
    plot_confusion_matrix(val_metrics['labels'], val_metrics['predictions'],
                         os.path.join(Config.PLOTS_DIR, 'confusion_matrix.png'))
    plot_memory_prototypes(model, os.path.join(Config.PLOTS_DIR, 'prototypes_tsne.png'))
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Results saved to: {Config.RESULTS_DIR}")
    print(f"Plots saved to: {Config.PLOTS_DIR}")
    print(f"Model saved to: {Config.CHECKPOINT_DIR}")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("HBG-MACN TRAINING - SDY2011 Dataset")
    print("="*80)
    print("Dataset: SDY2011 (COVID-19 Classification)")
    print("Task: COVID Negative vs COVID Positive")
    print("="*80 + "\n")
    
    # ===== PHASE 1: ANALYZE MARKERS =====
    print("🔍 PHASE 1: ANALYZING MARKERS\n")
    
    Config.DATASET_NAME = 'SDY2011'
    Config.CHECKPOINT_DIR = 'checkpoints_SDY2011/'
    Config.RESULTS_DIR = 'results_SDY2011/'
    Config.PLOTS_DIR = 'plots_SDY2011/'
    
    config_dict = Config.get_config()
    config_dict['dataset_name'] = 'SDY2011'
    
    # Run detailed analysis
    recommended_markers, marker_freq = analyze_dataset_markers_detailed(config_dict)
    
    print(f"\n✓ Selected {len(recommended_markers)} markers for training")
    
    # Show marker summary
    cd_markers = [m for m in recommended_markers if any(x in m.upper() for x in ['CD', 'HLA', 'IG'])]
    print(f"\n📊 Marker Breakdown:")
    print(f"  - CD/HLA/IG markers: {len(cd_markers)}")
    print(f"  - Total markers: {len(recommended_markers)}")
    
    if len(cd_markers) > 0:
        print(f"\n🧬 Sample CD markers:")
        for i, marker in enumerate(cd_markers[:12]):
            print(f"  {i+1:2d}. {marker}")
        if len(cd_markers) > 12:
            print(f"  ... and {len(cd_markers) - 12} more")
    
    print("\n" + "="*80)
    
    # ===== PHASE 2: CONFIRM AND TRAIN =====
    print("\n🔷 PHASE 2: MODEL TRAINING\n")
    
    print(f"Ready to train HBG-MACN with {len(recommended_markers)} markers")
    print(f"Expected model input dimension: {len(recommended_markers)}")
    print(f"Training parameters:")
    print(f"  - Batch size: {Config.BATCH_SIZE}")
    print(f"  - Epochs: {Config.EPOCHS}")
    print(f"  - Learning rate: {Config.LR}")
    print(f"  - Device: {Config.DEVICE}")
    
    user_input = input("\nProceed with training? (yes/no): ").strip().lower()
    
    if user_input != 'yes':
        print("\n❌ Training cancelled. Review marker analysis and restart when ready.")
        exit()
    
    print("\n" + "="*80)
    print("🚀 STARTING TRAINING")
    print("="*80 + "\n")
    
    try:
        main()
        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"📁 Model saved to: {Config.CHECKPOINT_DIR}")
        print(f"📊 Results saved to: {Config.RESULTS_DIR}")
        print(f"📈 Plots saved to: {Config.PLOTS_DIR}")
        print("\n💡 Next steps:")
        print("  1. Review training curves in plots_SDY2011/")
        print("  2. Check validation metrics in results_SDY2011/")
        print("  3. Run experiments code to generate interpretability analysis")
        print("="*80 + "\n")
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ TRAINING FAILED")
        print("="*80)
        print(f"Error: {str(e)}\n")
        import traceback
        traceback.print_exc()
        print("\n" + "="*80 + "\n")
    
    # Print final summary
    print("\n" + "="*80)
    print("MULTI-DATASET TRAINING COMPLETE!")
    print("="*80)
    print("\nMarker Analysis:")
    for dataset_name, analysis in all_marker_analyses.items():
        if analysis:
            print(f"  {dataset_name}: {analysis['num_markers']} markers")
    
    print("\nTraining Results:")
    for dataset_name, status in all_results.items():
        print(f"  {dataset_name}: {status}")
    print("="*80 + "\n")