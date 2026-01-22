class CytoPathDataset(Dataset):
    """Flow cytometry dataset with proper cell-level sampling"""
    
    def __init__(self, sample_data, sample_ids, marker_cols, 
                 cells_per_sample=5000, mode='train'):
        self.sample_data = sample_data
        self.sample_ids = sample_ids
        self.marker_cols = marker_cols
        self.cells_per_sample = cells_per_sample
        self.mode = mode
        
        self.labels = []
        self.valid_ids = []
        for sid in sample_ids:
            if sid in sample_data:
                self.labels.append(sample_data[sid]['label'])
                self.valid_ids.append(sid)
        
        self.labels = np.array(self.labels)
        print(f"{mode} dataset: {len(self.valid_ids)} samples")
    
    def __len__(self):
        return len(self.valid_ids)
    
    def __getitem__(self, idx):
        sample_id = self.valid_ids[idx]
        cell_df = self.sample_data[sample_id]['cells']
        label = self.sample_data[sample_id]['label']
        
        # ===== CRITICAL: Only use markers that exist in this file =====
        # Some files might have extra columns, but we only use common ones
        available_markers = [col for col in self.marker_cols if col in cell_df.columns]
        
        # Safety check
        if len(available_markers) != len(self.marker_cols):
            missing = set(self.marker_cols) - set(available_markers)
            raise ValueError(f"Sample {sample_id} missing {len(missing)} markers: {list(missing)[:5]}")
        
        cell_data = cell_df[available_markers].values
        # ============================================================
        
        # Sample cells
        n_total = len(cell_data)
        if n_total > self.cells_per_sample:
            indices = np.random.choice(n_total, self.cells_per_sample, replace=False)
            sampled_cells = cell_data[indices]
        else:
            indices = np.random.choice(n_total, self.cells_per_sample, replace=True)
            sampled_cells = cell_data[indices]
        
        # Transform
        sampled_cells = np.arcsinh(sampled_cells / 5.0)
        
        return {
            'cells': torch.FloatTensor(sampled_cells),
            'label': torch.LongTensor([label])[0],
            'sample_id': sample_id
        }


def collate_fn(batch):
    """No padding needed - all samples have same cell count"""
    cells = torch.stack([b['cells'] for b in batch])
    labels = torch.stack([b['label'] for b in batch])
    sample_ids = [b['sample_id'] for b in batch]
    
    return {
        'cells': cells,
        'labels': labels,
        'sample_ids': sample_ids
    }


def load_data(config_dict):
    """Load data based on dataset type"""
    
    print("="*80)
    print(f"LOADING DATA: {Config.DATASET_NAME}")
    print("="*80)
    
    if config_dict['file_format'] == 'parquet':
        return load_parquet_data(config_dict)
    elif config_dict['file_format'] == 'fcs':
        return load_fcs_data(config_dict)
    else:
        raise ValueError(f"Unknown file format: {config_dict['file_format']}")


def load_parquet_data(config_dict):
    """Load parquet data (your BM dataset)"""
    
    labels_df = pd.read_excel(config_dict['label_file'])
    print(f"Total samples in label file: {len(labels_df)}")
    
    def read_parquet_file(file_path):
        table = pq.read_table(file_path)
        return table.to_pandas()
    
    sample_data = {}
    dataset_dir = config_dict['dataset_dir']
    
    for idx, row in labels_df.iterrows():
        fid = str(int(row['flow_id']))
        label = row['true_class']
        usage = row['usage']
        
        parquet_path = dataset_dir / f"{fid}.parquet"
        gzip_path = dataset_dir / f"{fid}.parquet.gzip"
        
        if parquet_path.exists():
            file_path = parquet_path
        elif gzip_path.exists():
            file_path = gzip_path
        else:
            continue
        
        cell_df = read_parquet_file(file_path)
        sample_data[fid] = {
            "cells": cell_df,
            "label": label,
            "usage": usage
        }
    
    train_ids = [fid for fid, data in sample_data.items() if data['usage'] == 'train']
    val_ids = [fid for fid, data in sample_data.items() if data['usage'] == 'validation']
    
    marker_cols = config_dict['marker_cols']
    
    print(f"Loaded {len(sample_data)} samples")
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")
    print(f"Markers: {len(marker_cols)}")
    print("="*80 + "\n")
    
    return sample_data, train_ids, val_ids, marker_cols


def load_fcs_data(config_dict):
    """Load FCS data (SDY datasets) with smart marker selection"""
    
    import json
    import fcsparser
    
    print(f"Loading manifest from: {config_dict['manifest_file']}")
    
    # ===== STEP 0: ANALYZE MARKERS FIRST =====
    recommended_markers, marker_frequency = analyze_dataset_markers_detailed(config_dict)
    
    print(f"\n{'='*80}")
    print(f"PROCEEDING WITH {len(recommended_markers)} RECOMMENDED MARKERS")
    print(f"{'='*80}\n")
    
    # Load manifest
    with open(config_dict['manifest_file'], 'r') as f:
        manifest = json.load(f)
    
    if isinstance(manifest, list):
        entries = manifest
    elif isinstance(manifest, dict):
        entries = manifest.get('files', manifest.get('data', []))
    else:
        raise ValueError(f"Unknown manifest format: {type(manifest)}")
    
    sample_data = {}
    dataset_dir = config_dict['dataset_dir']
    label_mapping = config_dict['label_mapping']
    
    skipped_not_fcs = 0
    skipped_no_file = 0
    skipped_no_match = 0
    skipped_parse_error = 0
    skipped_missing_markers = 0
    
    print(f"[LOADING FCS FILES WITH RECOMMENDED MARKERS]")
    
    for entry in tqdm(entries, desc="Loading samples"):
        filename = entry.get('fileName')
        if not filename:
            continue
            
        if not filename.endswith('.fcs'):
            skipped_not_fcs += 1
            continue
        
        file_path = dataset_dir / filename
        if not file_path.exists():
            skipped_no_file += 1
            continue
        
        treatment_arm_raw = entry.get('armName', '')
        treatment_arm = treatment_arm_raw.lower().replace(' ', '').replace('-', '')
        
        if treatment_arm not in label_mapping:
            skipped_no_match += 1
            continue
        
        label = label_mapping[treatment_arm]
        subject_id = entry.get('subjectAccession', filename)
        
        # Parse FCS file
        try:
            meta, cell_df = fcsparser.parse(str(file_path), reformat_meta=True)
            
            # Check if file has all recommended markers
            available_markers = set(cell_df.columns)
            missing_markers = set(recommended_markers) - available_markers
            
            if len(missing_markers) > 0:
                # Skip files missing too many markers (>10%)
                if len(missing_markers) > len(recommended_markers) * 0.1:
                    skipped_missing_markers += 1
                    continue
                
                # For files missing few markers, fill with zeros
                for marker in missing_markers:
                    cell_df[marker] = 0.0
            
            # Select only recommended markers in consistent order
            cell_df = cell_df[recommended_markers]
            
            sample_data[filename] = {
                "cells": cell_df,
                "label": label,
                "usage": "unknown",
                "subject_id": subject_id
            }
            
        except Exception as e:
            skipped_parse_error += 1
            continue
    
    # Summary
    print(f"\n[LOADING SUMMARY]")
    print(f"  Successfully loaded: {len(sample_data)} samples")
    print(f"  Skipped - Not FCS: {skipped_not_fcs}")
    print(f"  Skipped - File not found: {skipped_no_file}")
    print(f"  Skipped - Label not matching: {skipped_no_match}")
    print(f"  Skipped - Parse errors: {skipped_parse_error}")
    print(f"  Skipped - Missing markers: {skipped_missing_markers}")
    
    # Set INPUT_DIM dynamically
    Config.INPUT_DIM = len(recommended_markers)
    print(f"  ✓ Using {len(recommended_markers)} markers")
    
    # Random train/val split
    from sklearn.model_selection import train_test_split
    all_ids = list(sample_data.keys())
    all_labels = [sample_data[sid]['label'] for sid in all_ids]
    
    train_ids, val_ids = train_test_split(
        all_ids, test_size=0.2, random_state=Config.SEED, stratify=all_labels
    )
    
    print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}")
    
    # Print class distribution
    train_labels = [sample_data[sid]['label'] for sid in train_ids]
    val_labels = [sample_data[sid]['label'] for sid in val_ids]
    print(f"  Train - Class 0: {train_labels.count(0)}, Class 1: {train_labels.count(1)}")
    print(f"  Val   - Class 0: {val_labels.count(0)}, Class 1: {val_labels.count(1)}")
    
    print("="*80 + "\n")
    
    return sample_data, train_ids, val_ids, recommended_markers