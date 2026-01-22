# ==================== EXPERIMENT 1: t-SNE EMBEDDING EVOLUTION ====================

def experiment_tsne_evolution(model, loader, device, save_dir):
    """
    Show how representations evolve through the model
    4 panels: Raw → Cell Embeddings → Graph Features → Final Features
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: t-SNE EMBEDDING EVOLUTION")
    print("="*80)
    
    model.eval()
    
    # Collect features from all samples
    raw_features_list = []
    cell_embeddings_list = []
    graph_features_list = []
    final_features_list = []
    labels_list = []
    sample_ids_list = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features"):
            cells = batch['cells'].to(device)
            labels = batch['labels']
            
            outputs = model(cells, return_all=True)
            
            # Flatten cell-level features (take subset for speed)
            batch_size = cells.shape[0]
            for i in range(batch_size):
                # Sample 500 cells per sample for visualization
                n_cells_viz = min(500, cells.shape[1])
                indices = np.random.choice(cells.shape[1], n_cells_viz, replace=False)
                
                raw_features_list.append(outputs['raw_features'][i, indices].cpu().numpy())
                cell_embeddings_list.append(outputs['cell_embeddings'][i, indices].cpu().numpy())
                graph_features_list.append(outputs['graph_features'][i, indices].cpu().numpy())
                
                # Repeat sample-level features for each cell
                final_feat = outputs['fused_features'][i].cpu().numpy()
                final_features_list.append(np.tile(final_feat, (n_cells_viz, 1)))
                
                labels_list.extend([labels[i].item()] * n_cells_viz)
                sample_ids_list.extend([batch['sample_ids'][i]] * n_cells_viz)
    
    # Concatenate all
    raw_features = np.vstack(raw_features_list)
    cell_embeddings = np.vstack(cell_embeddings_list)
    graph_features = np.vstack(graph_features_list)
    final_features = np.vstack(final_features_list)
    labels_array = np.array(labels_list)
    
    print(f"Total cells for t-SNE: {len(labels_array)}")
    
    # Subsample for t-SNE speed (max 5000 cells)
    if len(labels_array) > 5000:
        indices = np.random.choice(len(labels_array), 5000, replace=False)
        raw_features = raw_features[indices]
        cell_embeddings = cell_embeddings[indices]
        graph_features = graph_features[indices]
        final_features = final_features[indices]
        labels_array = labels_array[indices]
    
    # Apply t-SNE to each representation
    print("Running t-SNE on raw features...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    raw_2d = tsne.fit_transform(raw_features)
    
    print("Running t-SNE on cell embeddings...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    cell_2d = tsne.fit_transform(cell_embeddings)
    
    print("Running t-SNE on graph features...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    graph_2d = tsne.fit_transform(graph_features)
    
    print("Running t-SNE on final features...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    final_2d = tsne.fit_transform(final_features)
    
    # Save all t-SNE coordinates to CSV
    tsne_df = pd.DataFrame({
        'raw_x': raw_2d[:, 0],
        'raw_y': raw_2d[:, 1],
        'cell_x': cell_2d[:, 0],
        'cell_y': cell_2d[:, 1],
        'graph_x': graph_2d[:, 0],
        'graph_y': graph_2d[:, 1],
        'final_x': final_2d[:, 0],
        'final_y': final_2d[:, 1],
        'label': labels_array
    })
    tsne_df.to_csv(save_dir / 'tsne_evolution.csv', index=False)
    print(f"✓ Saved: tsne_evolution.csv")
    
    # Create 2x2 plot
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    
    colors = ['#3498db', '#e74c3c']  # Blue for COVID-, red for COVID+
    titles = ['(A) Raw Features', '(B) Cell Embeddings', 
              '(C) Graph Features', '(D) Final Features']
    data_list = [raw_2d, cell_2d, graph_2d, final_2d]
    
    for ax, data, title in zip(axes.flat, data_list, titles):
        for label in [0, 1]:
            mask = labels_array == label
            ax.scatter(data[mask, 0], data[mask, 1], 
                      c=colors[label], 
                      label=['COVID-', 'COVID+'][label],
                      alpha=0.6, s=5, edgecolors='none')
        
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('t-SNE 1', fontsize=12)
        ax.set_ylabel('t-SNE 2', fontsize=12)
        ax.legend(fontsize=12, markerscale=5, loc='upper right')
        ax.tick_params(labelsize=12)
        ax.grid(True, alpha=0.2, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'tsne_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: tsne_evolution.png")
    print(f"✓ Results saved to: {save_dir}\n")

# ==================== EXPERIMENT 2: PROTOTYPE VISUALIZATION ====================

def experiment_prototype_heatmap(model, loader, device, save_dir):
    """
    Visualize top 20 prototypes and their marker expressions
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: PROTOTYPE VISUALIZATION")
    print("="*80)
    
    model.eval()
    
    # Get prototypes
    prototypes = model.memory_bank.prototypes.cpu().numpy()  # [200, 128]
    
    print(f"Prototypes shape: {prototypes.shape}")
    
    # Collect prototype activations per sample
    prototype_activations = []
    labels_list = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing prototype usage"):
            cells = batch['cells'].to(device)
            labels = batch['labels']
            
            outputs = model(cells, return_all=True)
            cell_embeddings = outputs['cell_embeddings']  # [batch, num_cells, 128]
            
            for i in range(len(batch['sample_ids'])):
                sample_cells = cell_embeddings[i]  # [num_cells, 128]
                
                # Normalize
                sample_cells_norm = F.normalize(sample_cells, dim=1)
                prototypes_norm = F.normalize(torch.FloatTensor(prototypes).to(device), dim=1)
                
                # Compute similarity
                similarities = torch.mm(sample_cells_norm, prototypes_norm.t())  # [num_cells, 200]
                
                # Soft assignment
                soft_assignments = F.softmax(similarities / 0.07, dim=1)
                
                # Average across cells
                usage_vector = soft_assignments.mean(dim=0).cpu().numpy()
                
                prototype_activations.append(usage_vector)
                labels_list.append(labels[i].item())
    
    prototype_activations = np.array(prototype_activations)  # [num_samples, 200]
    labels_array = np.array(labels_list)
    
    # Find top 20 discriminative prototypes
    covid_neg_mean = prototype_activations[labels_array == 0].mean(axis=0)
    covid_pos_mean = prototype_activations[labels_array == 1].mean(axis=0)
    
    usage_diff = np.abs(covid_neg_mean - covid_pos_mean)
    top_20_idx = np.argsort(usage_diff)[-20:][::-1]
    
    print(f"Top 20 discriminative prototypes: {top_20_idx}")
    
    # Get top 20 prototype embeddings
    top_prototypes = prototypes[top_20_idx]  # [20, 128]
    
    print("Computing prototype-marker correlations...")
    
    # Collect all cells and their marker values
    all_cells_markers = []
    all_cells_proto_sim = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Collecting cells", leave=False):
            cells = batch['cells'].to(device)
            
            outputs = model(cells, return_all=True)
            cell_embeddings = outputs['cell_embeddings']
            
            # Flatten across batch
            cells_flat = cells.reshape(-1, cells.shape[-1])  # [batch*num_cells, num_markers]
            embeddings_flat = cell_embeddings.reshape(-1, cell_embeddings.shape[-1])
            
            # Sample subset for speed
            n_sample = min(1000, cells_flat.shape[0])
            indices = np.random.choice(cells_flat.shape[0], n_sample, replace=False)
            
            cells_subset = cells_flat[indices].cpu().numpy()
            embeddings_subset = embeddings_flat[indices]
            
            # Compute similarity to top 20 prototypes
            embeddings_norm = F.normalize(embeddings_subset, dim=1)
            top_protos_norm = F.normalize(torch.FloatTensor(top_prototypes).to(device), dim=1)
            
            similarities = torch.mm(embeddings_norm, top_protos_norm.t()).cpu().numpy()  # [n_sample, 20]
            
            all_cells_markers.append(cells_subset)
            all_cells_proto_sim.append(similarities)
    
    all_cells_markers = np.vstack(all_cells_markers)  # [total_cells, num_markers]
    all_cells_proto_sim = np.vstack(all_cells_proto_sim)  # [total_cells, 20]
    
    print(f"Collected {all_cells_markers.shape[0]} cells")
    
    # Compute correlation between each prototype similarity and each marker
    from scipy.stats import pearsonr
    
    num_markers = len(Config.MARKER_COLS)
    prototype_marker_profile = np.zeros((20, num_markers))
    
    for p_idx in range(20):
        for m_idx in range(num_markers):
            corr, _ = pearsonr(all_cells_proto_sim[:, p_idx], all_cells_markers[:, m_idx])
            prototype_marker_profile[p_idx, m_idx] = corr
    
    # Save prototype profiles to CSV
    # Select top 15 most important markers for visualization
    marker_variance = np.abs(prototype_marker_profile).sum(axis=0)
    top_15_markers = np.argsort(marker_variance)[-15:][::-1]
    
    prototype_df = pd.DataFrame(
        prototype_marker_profile[:, top_15_markers],
        columns=[Config.MARKER_NAMES[i] for i in top_15_markers],
        index=[f'Proto_{idx}' for idx in top_20_idx]
    )
    prototype_df.to_csv(save_dir / 'prototype_marker_profiles.csv')
    print(f"✓ Saved: prototype_marker_profiles.csv")
    
    # Also save activation statistics
    activation_stats = pd.DataFrame({
        'prototype_id': top_20_idx,
        'covid_neg_mean': covid_neg_mean[top_20_idx],
        'covid_pos_mean': covid_pos_mean[top_20_idx],
        'difference': usage_diff[top_20_idx]
    })
    activation_stats.to_csv(save_dir / 'prototype_activations.csv', index=False)
    print(f"✓ Saved: prototype_activations.csv")
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(7, 5))
    
    sns.heatmap(prototype_marker_profile[:, top_15_markers], 
                cmap='RdBu_r', center=0, vmin=-0.5, vmax=0.5,
                xticklabels=[Config.MARKER_NAMES[i] for i in top_15_markers],
                yticklabels=[f'P{idx}' for idx in top_20_idx],
                cbar_kws={'label': 'Correlation'},
                ax=ax)
    
    ax.set_xlabel('Markers', fontsize=11)
    ax.set_ylabel('Prototypes', fontsize=11)
    
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'prototype_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: prototype_heatmap.png")
    print(f"✓ Results saved to: {save_dir}\n")

# ==================== EXPERIMENT 3: COMPONENT SYNERGY ====================

def experiment_component_synergy(model, loader, device, save_dir):
    """
    Track 2-3 samples through the pipeline showing prediction evolution
    """
    print("\n" + "="*80)
    print("EXPERIMENT 3: COMPONENT SYNERGY ANALYSIS")
    print("="*80)
    
    model.eval()
    
    # Find interesting samples: 1 correct COVID-, 1 correct COVID+, 1 error case
    sample_cases = {'correct_covid_neg': None, 'correct_covid_pos': None, 'error': None}
    
    with torch.no_grad():
        for batch in loader:
            cells = batch['cells'].to(device)
            labels = batch['labels']
            sample_id = batch['sample_ids'][0]
            
            # Get predictions at each stage
            outputs = model(cells, return_all=True)
            
            # Manual prediction at each stage
            cell_emb_pooled = outputs['cell_embeddings'].mean(dim=1)
            stage1_logit = model.classifier(cell_emb_pooled).squeeze(1)
            stage1_prob = torch.sigmoid(stage1_logit).item()
            
            # Stage 2: + Graph
            graph_pooled = outputs['graph_features'].mean(dim=1)
            stage2_logit = model.classifier(graph_pooled).squeeze(1)
            stage2_prob = torch.sigmoid(stage2_logit).item()
            
            # Stage 3: + Hierarchy
            hier_pooled = outputs['hierarchy_features'][:, -1, :]
            stage3_logit = model.classifier(hier_pooled).squeeze(1)
            stage3_prob = torch.sigmoid(stage3_logit).item()
            
            # Stage 4: Full model
            final_prob = torch.sigmoid(outputs['logits']).item()
            
            true_label = labels[0].item()
            final_pred = 1 if final_prob > 0.5 else 0
            
            # Store interesting cases
            if true_label == 0 and final_pred == 0 and sample_cases['correct_covid_neg'] is None:
                sample_cases['correct_covid_neg'] = {
                    'sample_id': sample_id,
                    'label': true_label,
                    'stage1': stage1_prob,
                    'stage2': stage2_prob,
                    'stage3': stage3_prob,
                    'stage4': final_prob
                }
            
            if true_label == 1 and final_pred == 1 and sample_cases['correct_covid_pos'] is None:
                sample_cases['correct_covid_pos'] = {
                    'sample_id': sample_id,
                    'label': true_label,
                    'stage1': stage1_prob,
                    'stage2': stage2_prob,
                    'stage3': stage3_prob,
                    'stage4': final_prob
                }
            
            if final_pred != true_label and sample_cases['error'] is None:
                sample_cases['error'] = {
                    'sample_id': sample_id,
                    'label': true_label,
                    'stage1': stage1_prob,
                    'stage2': stage2_prob,
                    'stage3': stage3_prob,
                    'stage4': final_prob
                }
            
            if all(v is not None for v in sample_cases.values()):
                break
    
    # Save to CSV
    synergy_data = []
    for case_type, case_data in sample_cases.items():
        if case_data is not None:
            synergy_data.append({
                'case_type': case_type,
                'sample_id': case_data['sample_id'],
                'true_label': case_data['label'],
                'stage1_set_transformer': case_data['stage1'],
                'stage2_graph': case_data['stage2'],
                'stage3_hierarchy': case_data['stage3'],
                'stage4_full_model': case_data['stage4']
            })
    
    synergy_df = pd.DataFrame(synergy_data)
    synergy_df.to_csv(save_dir / 'component_synergy.csv', index=False)
    print(f"✓ Saved: component_synergy.csv")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(3.5, 3))
    
    stages = ['Set\nTransformer', '+ Graph', '+ Hierarchy', 'Full\nModel']
    x = np.arange(len(stages))
    
    colors_map = {
        'correct_covid_neg': '#3498db',
        'correct_covid_pos': '#e74c3c',
        'error': '#f39c12'
    }
    labels_map = {
        'correct_covid_neg': 'Correct (COVID-)',
        'correct_covid_pos': 'Correct (COVID+)',
        'error': 'Error'
    }
    
    for idx, (case_type, case_data) in enumerate(sample_cases.items()):
        if case_data is not None:
            probs = [case_data['stage1'], case_data['stage2'], 
                    case_data['stage3'], case_data['stage4']]
            
            ax.plot(x, probs, marker='o', linewidth=2, markersize=6,
                   color=colors_map[case_type], label=labels_map[case_type])
    
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Threshold')
    
    ax.set_xlabel('Model Component', fontsize=8, fontweight='bold')
    ax.set_ylabel('COVID+ Probability', fontsize=8, fontweight='bold')
    ax.set_title('Prediction Evolution Through Pipeline', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=7)
    ax.set_ylim([0, 1])
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3, linewidth=0.5, axis='y')
    ax.tick_params(labelsize=7)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'component_synergy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: component_synergy.png")
    print(f"✓ Results saved to: {save_dir}\n")

# ==================== EXPERIMENT 4: SALIENCY MAPS ====================

def experiment_saliency_maps(model, loader, device, save_dir):
    """
    Compute integrated gradients to identify important markers
    """
    print("\n" + "="*80)
    print("EXPERIMENT 4: MARKER SALIENCY ANALYSIS")
    print("="*80)
    
    # Collect gradients for all samples
    all_gradients = []
    all_labels = []
    all_sample_ids = []
    
    for batch in tqdm(loader, desc="Computing gradients"):
        cells = batch['cells'].to(device)
        labels = batch['labels']
        
        # IMPORTANT: Model must be in train mode for gradients
        model.train()
        cells.requires_grad = True
        
        outputs = model(cells, return_all=True)
        logits = outputs['logits']
        
        # Compute loss on absolute logits
        model.zero_grad()
        loss = logits.abs().mean()
        loss.backward()
        
        # Get gradients
        gradients = cells.grad.abs().mean(dim=1).cpu().numpy()  # [batch, num_markers]
        
        all_gradients.append(gradients)
        all_labels.extend(labels.numpy())
        all_sample_ids.extend(batch['sample_ids'])
        
        cells.requires_grad = False
        model.eval()
    
    all_gradients = np.vstack(all_gradients)
    all_labels = np.array(all_labels)
    
    # Compute mean gradient per class
    covid_neg_gradients = all_gradients[all_labels == 0].mean(axis=0)
    covid_pos_gradients = all_gradients[all_labels == 1].mean(axis=0)
    overall_gradients = all_gradients.mean(axis=0)
    
    # Get top 10 markers
    top_10_idx = np.argsort(overall_gradients)[-10:][::-1]
    
    # Save to CSV
    marker_names_full = Config.MARKER_NAMES
    
    saliency_df = pd.DataFrame({
        'marker': marker_names_full,
        'overall_importance': overall_gradients,
        'covid_neg_importance': covid_neg_gradients,
        'covid_pos_importance': covid_pos_gradients,
        'difference': np.abs(covid_neg_gradients - covid_pos_gradients)
    })
    saliency_df = saliency_df.sort_values('overall_importance', ascending=False)
    saliency_df.to_csv(save_dir / 'marker_saliency.csv', index=False)
    print(f"✓ Saved: marker_saliency.csv")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(3.5, 4))
    
    top_markers = [marker_names_full[i] for i in top_10_idx]
    
    x = np.arange(len(top_markers))
    width = 0.35
    
    ax.barh(x - width/2, covid_neg_gradients[top_10_idx], width, 
            label='COVID-', color='#3498db', edgecolor='black', linewidth=0.5)
    ax.barh(x + width/2, covid_pos_gradients[top_10_idx], width,
            label='COVID+', color='#e74c3c', edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(x)
    ax.set_yticklabels(top_markers, fontsize=7)
    ax.set_xlabel('Gradient Magnitude', fontsize=8, fontweight='bold')
    ax.set_title('Top 10 Important Markers', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, linewidth=0.5, axis='x')
    ax.tick_params(labelsize=7)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'marker_saliency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: marker_saliency.png")
    print(f"\nTop 10 Important Markers:")
    for i, idx in enumerate(top_10_idx[:10]):
        marker_name = marker_names_full[idx]
        print(f"  {i+1}. {marker_name}: {overall_gradients[idx]:.4f}")
    
    print(f"\n✓ Results saved to: {save_dir}\n")

# ==================== EXPERIMENT 5: CASE STUDY ====================

def experiment_case_study(model, loader, device, save_dir):
    """
    Deep dive into 2 samples: 1 correct, 1 error
    Show all components for each
    """
    print("\n" + "="*80)
    print("EXPERIMENT 5: CASE STUDY ANALYSIS")
    print("="*80)
    
    model.eval()
    
    # Find cases
    correct_case = None
    error_case = None
    
    with torch.no_grad():
        for batch in loader:
            cells = batch['cells'].to(device)
            labels = batch['labels']
            
            outputs = model(cells, return_all=True)
            prob = torch.sigmoid(outputs['logits']).item()
            pred = 1 if prob > 0.5 else 0
            true_label = labels[0].item()
            
            if pred == true_label and correct_case is None:
                correct_case = {
                    'sample_id': batch['sample_ids'][0],
                    'cells': cells,
                    'label': true_label,
                    'prob': prob,
                    'outputs': outputs
                }
            
            if pred != true_label and error_case is None:
                error_case = {
                    'sample_id': batch['sample_ids'][0],
                    'cells': cells,
                    'label': true_label,
                    'prob': prob,
                    'outputs': outputs
                }
            
            if correct_case is not None and error_case is not None:
                break
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(7, 6))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    case_data = []
    
    for row, (case, case_name) in enumerate([(correct_case, 'Correct'), (error_case, 'Error')]):
        if case is None:
            continue
        
        # Panel 1: t-SNE of cells
        ax1 = fig.add_subplot(gs[row, 0])
        
        cell_emb = case['outputs']['cell_embeddings'][0].cpu().numpy()
        tsne = TSNE(n_components=2, random_state=42)
        cell_2d = tsne.fit_transform(cell_emb[:500])
        
        ax1.scatter(cell_2d[:, 0], cell_2d[:, 1], s=1, alpha=0.5, c='#3498db')
        ax1.set_title(f'{case_name}: Cell Embeddings', fontsize=8, fontweight='bold')
        ax1.set_xlabel('t-SNE 1', fontsize=7)
        ax1.set_ylabel('t-SNE 2', fontsize=7)
        ax1.tick_params(labelsize=6)
        
        # Panel 2: Attention weights
        ax2 = fig.add_subplot(gs[row, 1])
        
        attn = case['outputs']['attn_weights'][0].cpu().numpy()
        levels = ['Cell', 'L1', 'L2', 'L3', 'L4']
        
        ax2.bar(levels, attn, color='#2ecc71', edgecolor='black', linewidth=0.5)
        ax2.set_title(f'{case_name}: Attention Weights', fontsize=8, fontweight='bold')
        ax2.set_ylabel('Weight', fontsize=7)
        ax2.tick_params(labelsize=6)
        ax2.set_ylim([0, max(attn) * 1.2])
        
        # Panel 3: Prediction
        ax3 = fig.add_subplot(gs[row, 2])
        
        true_label = ['COVID-', 'COVID+'][case['label']]
        pred_label = 'COVID+' if case['prob'] > 0.5 else 'COVID-'
        
        color = '#2ecc71' if pred_label == true_label else '#e74c3c'
        
        ax3.barh([0], [case['prob']], color=color, edgecolor='black', linewidth=0.5)
        ax3.axvline(x=0.5, color='gray', linestyle='--', linewidth=1)
        ax3.set_xlim([0, 1])
        ax3.set_yticks([])
        ax3.set_xlabel('COVID+ Probability', fontsize=7)
        ax3.set_title(f'{case_name}: Prediction\nTrue: {true_label}', 
                     fontsize=8, fontweight='bold')
        ax3.tick_params(labelsize=6)
        
        # Save case data
        case_data.append({
            'case_type': case_name,
            'sample_id': case['sample_id'],
            'true_label': case['label'],
            'predicted_prob': case['prob'],
            'predicted_label': 1 if case['prob'] > 0.5 else 0,
            'cell_attn': attn[0],
            'level1_attn': attn[1],
            'level2_attn': attn[2],
            'level3_attn': attn[3],
            'level4_attn': attn[4]
        })
    
    plt.suptitle('Case Study: Model Interpretability', fontsize=10, fontweight='bold', y=0.98)
    plt.savefig(save_dir / 'case_study.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save to CSV
    case_df = pd.DataFrame(case_data)
    case_df.to_csv(save_dir / 'case_study.csv', index=False)
    
    print(f"✓ Saved: case_study.png")
    print(f"✓ Saved: case_study.csv")
    print(f"✓ Results saved to: {save_dir}\n")

# ==================== EXPERIMENT 6: GRAPH STRUCTURE COMPARISON ====================

def experiment_graph_comparison(model, loader, device, save_dir):
    """
    Compare graph structure: COVID- vs COVID+ (side-by-side)
    """
    print("\n" + "="*80)
    print("EXPERIMENT 6: GRAPH STRUCTURE COMPARISON")
    print("="*80)
    
    model.eval()
    
    # Find one COVID- and one COVID+ sample
    covid_neg_sample = None
    covid_pos_sample = None
    
    with torch.no_grad():
        for batch in loader:
            cells = batch['cells'].to(device)
            labels = batch['labels']
            
            outputs = model(cells, return_all=True)
            
            if labels[0].item() == 0 and covid_neg_sample is None:
                covid_neg_sample = {
                    'sample_id': batch['sample_ids'][0],
                    'embeddings': outputs['cell_embeddings'][0],
                    'edge_index': outputs['edge_index'][:, :Config.K_NEIGHBORS * Config.MAX_CELLS]
                }
            
            if labels[0].item() == 1 and covid_pos_sample is None:
                covid_pos_sample = {
                    'sample_id': batch['sample_ids'][0],
                    'embeddings': outputs['cell_embeddings'][0],
                    'edge_index': outputs['edge_index'][:, :Config.K_NEIGHBORS * Config.MAX_CELLS]
                }
            
            if covid_neg_sample is not None and covid_pos_sample is not None:
                break
    
    # Create side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))
    
    graph_metrics = []
    
    for ax, sample, title, label in [(ax1, covid_neg_sample, 'COVID-', 0),
                                      (ax2, covid_pos_sample, 'COVID+', 1)]:
        if sample is None:
            continue
        
        # Get embeddings and edges
        embeddings = sample['embeddings'].cpu().numpy()
        edge_index = sample['edge_index'].cpu().numpy()
        
        # Subsample for visualization
        n_viz = 500
        indices = np.random.choice(embeddings.shape[0], n_viz, replace=False)
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        emb_2d = tsne.fit_transform(embeddings[indices])
        
        # Create graph
        G = nx.Graph()
        for i in range(n_viz):
            G.add_node(i, pos=emb_2d[i])
        
        # Add edges (filter to visualized nodes)
        edge_mask = (edge_index[0] < n_viz) & (edge_index[1] < n_viz)
        edges_filtered = edge_index[:, edge_mask]
        
        for i in range(min(edges_filtered.shape[1], 2000)):
            src, dst = edges_filtered[0, i], edges_filtered[1, i]
            if src < n_viz and dst < n_viz:
                G.add_edge(src, dst)
        
        # Compute metrics
        clustering_coef = nx.average_clustering(G)
        avg_degree = np.mean([d for n, d in G.degree()])
        
        graph_metrics.append({
            'sample_type': title,
            'sample_id': sample['sample_id'],
            'clustering_coefficient': clustering_coef,
            'average_degree': avg_degree,
            'num_nodes': n_viz,
            'num_edges': G.number_of_edges()
        })
        
        # Plot
        pos = nx.get_node_attributes(G, 'pos')
        
        nx.draw_networkx_edges(G, pos, alpha=0.7, width=0.9, ax=ax)
        
        color = "#078017" if label == 0 else "#da121c"
        nx.draw_networkx_nodes(G, pos, node_size=13, node_color=color, 
                              alpha=0.9, ax=ax, edgecolors='none')
        
        ax.set_title(f'{title}\nClust: {clustering_coef:.3f}, Deg: {avg_degree:.1f}',
                    fontsize=9)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'graph_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(graph_metrics)
    metrics_df.to_csv(save_dir / 'graph_metrics.csv', index=False)
    
    print(f"✓ Saved: graph_comparison.png")
    print(f"✓ Saved: graph_metrics.csv")
    print(f"\nGraph Metrics:")
    print(metrics_df.to_string(index=False))
    print(f"\n✓ Results saved to: {save_dir}\n")

# ==================== EXPERIMENT 7: MARKER DISCOVERY ====================

def experiment_marker_discovery(model, loader, device, save_dir):
    """
    Comprehensive marker discovery with clinical validation
    """
    print("\n" + "="*80)
    print("EXPERIMENT 7: MARKER DISCOVERY & CLINICAL VALIDATION")
    print("="*80)
    
    model.eval()
    
    # Integrated Gradients implementation
    def integrated_gradients(model, inputs, baseline=None, steps=50):
        if baseline is None:
            baseline = torch.zeros_like(inputs)
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps).to(device)
        
        gradients = []
        for alpha in alphas:
            interpolated = baseline + alpha * (inputs - baseline)
            interpolated.requires_grad = True
            
            outputs = model(interpolated, return_all=True)
            logits = outputs['logits']
            
            model.zero_grad()
            logits.backward(torch.ones_like(logits))
            
            gradients.append(interpolated.grad.cpu().numpy())
            
            interpolated.requires_grad = False
        
        gradients = np.array(gradients).mean(axis=0)
        integrated_grad = (inputs.cpu().numpy() - baseline.cpu().numpy()) * gradients
        
        return integrated_grad
    
    # Collect integrated gradients
    all_importances = []
    all_labels = []
    
    print("Computing integrated gradients (this may take a while)...")
    
    for batch in tqdm(loader, desc="Computing importances", total=min(30, len(loader))):
        cells = batch['cells'].to(device)
        labels = batch['labels']
        
        # Compute integrated gradients
        baseline = torch.zeros_like(cells)
        importance = integrated_gradients(model, cells, baseline, steps=30)
        
        # Average over cells
        importance_mean = np.abs(importance).mean(axis=1)  # [batch, num_markers]
        
        all_importances.append(importance_mean)
        all_labels.extend(labels.numpy())
        
        if len(all_importances) >= 30:
            break
    
    all_importances = np.vstack(all_importances)
    all_labels = np.array(all_labels)
    
    # Compute statistics
    overall_importance = all_importances.mean(axis=0)
    covid_neg_importance = all_importances[all_labels == 0].mean(axis=0)
    covid_pos_importance = all_importances[all_labels == 1].mean(axis=0)
    
    # Known clinical markers for COVID immune response
    clinical_markers = {
        'CD3': 'T-cell marker',
        'CD4': 'Helper T-cell',
        'CD8': 'Cytotoxic T-cell',
        'CD19': 'B-cell marker',
        'CD20': 'B-cell marker',
        'CD14': 'Monocyte marker',
        'CD16': 'NK cell marker',
        'CD56': 'NK cell marker',
        'CD38': 'Activation marker',
        'HLA-DR': 'Antigen presentation',
        'CD11b': 'Myeloid marker',
        'CD11c': 'Dendritic cell',
        'CD45': 'Pan-leukocyte',
        'CD45RA': 'Naive marker',
        'CD45RO': 'Memory marker',
        'IgG': 'Antibody response',
        'IgA': 'Mucosal immunity',
        'IgM': 'Early antibody'
    }
    
    # Create comprehensive results
    marker_results = []
    marker_names_full = Config.MARKER_NAMES
    
    for i in range(len(marker_names_full)):
        marker_name = marker_names_full[i]
        
        # Check if clinically relevant
        clinical_relevance = 'Unknown'
        for key, value in clinical_markers.items():
            if key.upper() in marker_name.upper():
                clinical_relevance = value
                break
        
        marker_results.append({
            'marker': marker_name,
            'overall_importance': overall_importance[i],
            'covid_neg_importance': covid_neg_importance[i],
            'covid_pos_importance': covid_pos_importance[i],
            'difference': abs(covid_neg_importance[i] - covid_pos_importance[i]),
            'clinical_relevance': clinical_relevance
        })
    
    # Sort by overall importance
    marker_df = pd.DataFrame(marker_results)
    marker_df = marker_df.sort_values('overall_importance', ascending=False)
    marker_df.to_csv(save_dir / 'marker_discovery.csv', index=False)
    print(f"✓ Saved: marker_discovery.csv")
    
    # Create validation table
    print("\n" + "="*60)
    print("TOP 10 MARKERS - CLINICAL VALIDATION")
    print("="*60)
    print(f"{'Rank':<6} {'Marker':<25} {'Importance':<12} {'Clinical Relevance'}")
    print("-"*60)
    
    for idx, (_, row) in enumerate(marker_df.head(10).iterrows(), 1):
        validation = '✓' if row['clinical_relevance'] != 'Unknown' else '?'
        print(f"{idx:<6} {row['marker']:<25} {row['overall_importance']:<12.4f} "
              f"{validation} {row['clinical_relevance']}")
    
    print("="*60)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(3.5, 4))
    
    top_10 = marker_df.head(10)
    
    colors = ['#2ecc71' if r['clinical_relevance'] != 'Unknown' else '#95a5a6' 
             for _, r in top_10.iterrows()]
    
    y_pos = np.arange(len(top_10))
    
    ax.barh(y_pos, top_10['overall_importance'].values, 
            color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_10['marker'].values, fontsize=7)
    ax.set_xlabel('Importance Score', fontsize=8, fontweight='bold')
    ax.set_title('Top 10 Discriminative Markers\n(Green = Known Clinical Marker)', 
                 fontsize=9, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, linewidth=0.5, axis='x')
    ax.tick_params(labelsize=7)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'marker_discovery.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved: marker_discovery.png")
    print(f"✓ Results saved to: {save_dir}\n")

# ==================== EXPERIMENT 8: HIERARCHICAL INTERPRETATION ====================

def experiment_hierarchical_interpretation(model, loader, device, save_dir):
    """
    Qualitative analysis: What does each hierarchy level capture?
    """
    print("\n" + "="*80)
    print("EXPERIMENT 8: HIERARCHICAL LEVEL INTERPRETATION")
    print("="*80)
    
    model.eval()
    
    # Collect CELL-level features
    level_features = {
        'cell': [],
        'level1': [],
        'level2': [],
        'level3': [],
        'level4': []
    }
    labels_list = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting hierarchy features"):
            cells = batch['cells'].to(device)
            labels = batch['labels']
            
            outputs = model(cells, return_all=True)
            
            batch_size = cells.shape[0]
            n_cells_per_sample = min(500, cells.shape[1])
            
            for i in range(batch_size):
                indices = np.random.choice(cells.shape[1], n_cells_per_sample, replace=False)
                
                cell_emb = outputs['cell_embeddings'][i, indices].cpu().numpy()
                level_features['cell'].append(cell_emb)
                
                hier_feats = outputs['hierarchy_features'][i]
                
                for level_idx in range(4):
                    level_name = f'level{level_idx+1}'
                    level_feat = hier_feats[level_idx].unsqueeze(0).repeat(n_cells_per_sample, 1)
                    level_features[level_name].append(level_feat.cpu().numpy())
                
                labels_list.extend([labels[i].item()] * n_cells_per_sample)
    
    # Concatenate
    for key in level_features:
        level_features[key] = np.vstack(level_features[key])
    
    labels_array = np.array(labels_list)
    
    print(f"Total cells for visualization: {len(labels_array)}")
    
    # Compute class separation
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import cross_val_score
    
    separation_scores = {}
    
    for level_name, features in level_features.items():
        lda = LinearDiscriminantAnalysis()
        cv_scores = cross_val_score(lda, features, labels_array, cv=5, scoring='accuracy')
        acc = cv_scores.mean()
        
        separation_scores[level_name] = acc
        print(f"{level_name}: {acc:.3f} separation accuracy (CV)")
    
    # Save to CSV
    separation_df = pd.DataFrame({
        'hierarchy_level': list(separation_scores.keys()),
        'separation_score': list(separation_scores.values())
    })
    separation_df.to_csv(save_dir / 'hierarchical_separation.csv', index=False)
    print(f"✓ Saved: hierarchical_separation.csv")
    
    # Visualization 1: t-SNE of each level
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    
    colors = ["#3498db", "#e74c3c"]
    
    for idx, (level_name, features) in enumerate(level_features.items()):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        n_samples = min(5000, len(features))
        indices = np.random.choice(len(features), n_samples, replace=False)
        features_subset = features[indices]
        labels_subset = labels_array[indices]
        
        # t-SNE
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features_subset)
        
        for label in [0, 1]:
            mask = labels_subset == label
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                      c=colors[label], 
                      alpha=0.8, s=5, edgecolors='none')
        
        ax.set_title(f'{level_name.title()}', fontsize=11)
        ax.set_xlabel('t-SNE 1', fontsize=12)
        ax.set_ylabel('t-SNE 2', fontsize=12)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2, linewidth=0.5)
    
    # Panel 6: Combined Legend + Separation Scores
    ax_legend = axes[1, 2]
    ax_legend.axis('off')
    
    ax_legend.text(0.5, 0.95, 'Class Separation Scores', 
                   ha='center', va='top', fontsize=10,
                   transform=ax_legend.transAxes)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='COVID-'),
        Patch(facecolor='#e74c3c', label='COVID+')
    ]
    ax_legend.legend(handles=legend_elements, loc='lower center', 
                     fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.15))
    
    levels_ordered = ['cell', 'level1', 'level2', 'level3', 'level4']
    level_labels = ['Cell-Level', 'Level 1', 'Level 2', 'Level 3', 'Level 4']
    
    y_start = 0.65
    y_step = 0.12
    
    ax_legend.text(0.5, y_start + 0.15, 'Separation Accuracy:', 
                   ha='center', va='top', fontsize=10,
                   transform=ax_legend.transAxes)
    
    for i, (level, label) in enumerate(zip(levels_ordered, level_labels)):
        score = separation_scores[level]
        y_pos = y_start - i * y_step
        
        if score >= 0.90:
            color = "#050505"
        elif score >= 0.75:
            color = "#000000"
        else:
            color = "#000000"
        
        ax_legend.text(0.1, y_pos, f'{label}:', 
                       ha='left', va='center', fontsize=12,
                       transform=ax_legend.transAxes)
        
        ax_legend.text(0.85, y_pos, f'{score:.3f}', 
                       ha='right', va='center', fontsize=12,
                       color=color, transform=ax_legend.transAxes)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'hierarchical_tsne.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: hierarchical_tsne.png")
    
    # Visualization 2: Bar chart
    fig, ax = plt.subplots(figsize=(3.5, 3))
    
    levels_ordered = ['cell', 'level1', 'level2', 'level3', 'level4']
    level_labels = ['Cell-Level', 'Level 1\n(Cells)', 'Level 2\n(Clusters)', 
                    'Level 3\n(Lineages)', 'Level 4\n(Sample)']
    scores = [separation_scores[level] for level in levels_ordered]
    
    colors_bar = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    
    x = np.arange(len(level_labels))
    ax.bar(x, scores, color=colors_bar, edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Class Separation Score', fontsize=8, fontweight='bold')
    ax.set_xlabel('Hierarchy Level', fontsize=8, fontweight='bold')
    ax.set_title('Discriminative Power Across Hierarchy', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(level_labels, fontsize=7)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, linewidth=0.5, axis='y')
    ax.tick_params(labelsize=7)
    
    for i, v in enumerate(scores):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=6)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'hierarchical_separation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: hierarchical_separation.png")
    
    # Visualization 3: Attention correlation
    print("\nComputing attention-separation correlation...")
    
    attention_weights = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Getting attention", leave=False):
            cells = batch['cells'].to(device)
            outputs = model(cells, return_all=True)
            attention_weights.append(outputs['attn_weights'].cpu().numpy())
    
    attention_weights = np.vstack(attention_weights)
    
    import scipy.stats as stats
    
    correlation_data = []
    
    for i, level_name in enumerate(['cell', 'level1', 'level2', 'level3', 'level4']):
        corr, p_val = stats.pearsonr(attention_weights[:, i], 
                                      [separation_scores[level_name]] * len(attention_weights))
        
        correlation_data.append({
            'level': level_name,
            'mean_attention': attention_weights[:, i].mean(),
            'separation_score': separation_scores[level_name],
            'correlation': corr
        })
    
    corr_df = pd.DataFrame(correlation_data)
    corr_df.to_csv(save_dir / 'hierarchical_attention_separation.csv', index=False)
    print(f"✓ Saved: hierarchical_attention_separation.csv")
    
    print(f"\n✓ Results saved to: {save_dir}\n")
    
    # Print interpretation
    print("="*80)
    print("HIERARCHICAL INTERPRETATION:")
    print("="*80)
    best_level = max(separation_scores.items(), key=lambda x: x[1])
    print(f"Most discriminative level: {best_level[0].upper()} (score: {best_level[1]:.3f})")
    print(f"\nInterpretation:")
    print(f"  - Cell-level: Captures individual cell abnormalities")
    print(f"  - Level 1-2: Captures local cell clustering patterns")
    print(f"  - Level 3-4: Captures population-level distributions")
    print("="*80)