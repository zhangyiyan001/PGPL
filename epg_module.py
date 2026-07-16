# -*- coding:utf-8 -*-
"""
Core Enhanced Prototype Guidance (EPG) and Collaborative Progressive
Self-Training (CPST) components, including normalization, sample selection,
training-pool management, COU, and FIU.
"""
import numpy as np
import json
import os
from typing import Dict, Tuple, List
import torch

# ==================== Normalization ====================

def compute_normalization_stats(hsi_cube, lidar_map, valid_mask, save_path='results/normalization_stats.json'):
    """
    Compute and save z-score normalization statistics.
    
    Args:
        hsi_cube: HSI cube with shape (H, W, D).
        lidar_map: LiDAR map with shape (H, W) or (H, W, C).
        valid_mask: Boolean mask where True indicates a valid pixel.
        save_path: Output path for the statistics.
    
    Returns:
        stats_dict: Dictionary containing means and standard deviations.
    """
    H, W, D = hsi_cube.shape
    
    # Compute HSI statistics for each band.
    hsi_means = []
    hsi_stds = []
    
    print("Computing HSI normalization statistics...")
    for band in range(D):
        band_data = hsi_cube[:, :, band]
        valid_data = band_data[valid_mask]
        
        mean_val = np.mean(valid_data)
        std_val = np.std(valid_data)
        
        hsi_means.append(float(mean_val))
        hsi_stds.append(float(std_val))
    
    # Compute LiDAR statistics after removing invalid values.
    print("Computing LiDAR normalization statistics...")
    lidar_valid = lidar_map[valid_mask]
    
    # Remove NaN and infinite values.
    lidar_valid = lidar_valid[~np.isnan(lidar_valid)]
    lidar_valid = lidar_valid[~np.isinf(lidar_valid)]
    
    # Clip to the 1st and 99th percentiles.
    p1, p99 = np.percentile(lidar_valid, [1, 99])
    lidar_clipped = np.clip(lidar_valid, p1, p99)
    
    lidar_mean = float(np.mean(lidar_clipped))
    lidar_std = float(np.std(lidar_clipped))
    
    # Build the statistics dictionary.
    stats_dict = {
        'hsi_means': hsi_means,
        'hsi_stds': hsi_stds,
        'lidar_mean': lidar_mean,
        'lidar_std': lidar_std,
        'lidar_p1': float(p1),
        'lidar_p99': float(p99),
        'num_bands': D,
        'valid_pixels': int(np.sum(valid_mask))
    }
    
    # Save the statistics.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(stats_dict, f, indent=2)
    
    print(f"Normalization statistics saved to: {save_path}")
    print(f"  - HSI bands: {D}")
    print(f"  - Valid pixels: {np.sum(valid_mask)}")
    print(f"  - LiDAR range: [{p1:.2f}, {p99:.2f}]")
    
    return stats_dict

def normalize_with_stats(hsi_cube, lidar_map, stats_path='results/normalization_stats.json'):
    """
    Normalize HSI and LiDAR data using saved z-score statistics.
    
    Args:
        hsi_cube: Raw HSI data with shape (H, W, D).
        lidar_map: Raw LiDAR data with shape (H, W) or (H, W, C).
        stats_path: Path to the normalization statistics.
    
    Returns:
        hsi_normalized: Normalized HSI data.
        lidar_normalized: Normalized LiDAR data.
    """
    # Load statistics.
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    H, W, D = hsi_cube.shape
    hsi_normalized = np.zeros_like(hsi_cube, dtype=np.float32)
    
    # Normalize HSI bands.
    for band in range(D):
        mean = stats['hsi_means'][band]
        std = stats['hsi_stds'][band]
        if std == 0:
            std = 1e-8
        hsi_normalized[:, :, band] = (hsi_cube[:, :, band] - mean) / std
    
    # Normalize LiDAR channels.
    lidar_clipped = np.clip(lidar_map, stats['lidar_p1'], stats['lidar_p99'])
    lidar_clipped = np.nan_to_num(lidar_clipped, nan=stats['lidar_mean'])
    lidar_normalized = (lidar_clipped - stats['lidar_mean']) / stats['lidar_std']
    
    return hsi_normalized, lidar_normalized

# ==================== EPG ====================

def build_epg_unit_vectors(hsi_cube, valid_mask):
    """
    Build L2-normalized spectral vectors for EPG.
    
    Args:
        hsi_cube: HSI data with shape (H, W, D).
        valid_mask: Mask of valid pixels with shape (H, W).
    
    Returns:
        X_unit: L2-normalized spectral vectors with shape (N, D).
        coords: Corresponding row-column coordinates with shape (N, 2).
    """
    H, W, D = hsi_cube.shape
    
    # Extract valid pixels.
    coords = np.argwhere(valid_mask)  # (N, 2)
    X = hsi_cube[valid_mask]  # (N, D)
    
    # Apply L2 normalization.
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero.
    X_unit = X / norms
    
    return X_unit.astype(np.float32), coords

def compute_sam_matrix(X_unit, S_c):
    """
    Compute the spectral angle mapper (SAM) distance matrix.
    
    Args:
        X_unit: L2-normalized query spectra with shape (N, D).
        S_c: L2-normalized seed spectra with shape (M, D).
    
    Returns:
        SAM_c: SAM matrix with shape (N, M), measured in radians.
    """
    # Compute cosine similarity.
    cos_sim = np.dot(X_unit, S_c.T)  # (N, M)
    
    # Clip values to avoid numerical errors in arccos.
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    
    # Convert cosine similarity to angles in radians.
    SAM_c = np.arccos(cos_sim)
    
    return SAM_c

def epg_sample_selection(
    hsi_cube,
    lidar_map,
    gt_map,
    seeds_dict,
    gold_train_mask,
    test_mask,
    T_sam=0.15,
    T_h=0.75,
    topk_per_class=2000,
    lambda_epg=0.5,
    exclude_test_from_candidates=True,
    selection_mask_path='results/epg_selection_mask.npy',
    selection_labels_path='results/epg_labels.npy'
):
    """
    Select EPG samples using spectral-angle and height constraints.
    
    Args:
        hsi_cube: Normalized HSI data with shape (H, W, D).
        lidar_map: Z-score-normalized LiDAR data.
        gt_map: Ground-truth label map.
        seeds_dict: Seed dictionary in class-to-coordinate format.
        gold_train_mask: Mask of gold labels.
        test_mask: Test-set mask.
        T_sam: SAM threshold in radians.
        T_h: Height-difference threshold in z-score units.
        topk_per_class: Number or ratio of Top-K samples per class.
        lambda_epg: Weight between SAM and height difference.
    
    Returns:
        training_pool: Selected training-pool dictionary.
        preparation_pool: Preparation-pool mask.
    """
    H, W, D = hsi_cube.shape
    if lidar_map.ndim == 2:
        lidar_cube = lidar_map[..., np.newaxis]
    elif lidar_map.ndim == 3:
        lidar_cube = lidar_map
    else:
        raise ValueError("Unsupported LiDAR dimensionality")
    lidar_cube = np.asarray(lidar_cube, dtype=np.float32)
    lidar_channels = lidar_cube.shape[-1]
    valid_mask = (gt_map != 0)
    
    print("\n========== EPG Sample Selection ==========")
    print(
        f"Parameters: T_sam={T_sam:.3f} rad ({T_sam*180/np.pi:.1f} deg), "
        f"T_h={T_h:.2f} sigma, lambda_epg={lambda_epg:.3f}"
    )
    if 0 < topk_per_class < 1:
        print(f"Top-K per class: {topk_per_class*100:.1f}% of candidates")
    else:
        print(f"Top-K per class: {topk_per_class}")
    
    # Build normalized vectors for EPG.
    print("\nBuilding normalized EPG vectors...")
    X_unit, all_coords = build_epg_unit_vectors(hsi_cube, valid_mask)
    h_all = lidar_cube[valid_mask].reshape(-1, lidar_channels)  # (N, C)
    if len(h_all) == 0:
        raise ValueError("No valid pixels are available for EPG sample selection")

    h_range_vec = h_all.max(axis=0) - h_all.min(axis=0)
    h_range = float(np.linalg.norm(h_range_vec))
    if h_range < 1e-6:
        h_range = 1e-6
    print(f"Valid pixels: {len(X_unit)}, LiDAR range norm: {h_range:.4f}")
    
    # Initialize selection masks.
    epg_selection_mask = np.zeros((H, W), dtype=bool)
    epg_labels = np.zeros((H, W), dtype=int)
    epg_scores = np.zeros((H, W), dtype=float)
    
    # Process each class.
    num_classes = len(seeds_dict)
    selected_per_class = {}
    
    for class_key, seed_coords in seeds_dict.items():
        if len(seed_coords) == 0:
            print(f"\n{class_key}: no seed samples; skipping")
            continue
        
        class_id = int(class_key.split('_')[1])
        seed_coords = np.array(seed_coords)  # (M, 2) [row, col]
        
        # Extract seed spectra and heights.
        S_c = []
        h_seed_c = []
        for row, col in seed_coords:
            spectrum = hsi_cube[row, col]  # (D,)
            norm = np.linalg.norm(spectrum)
            if norm > 0:
                S_c.append(spectrum / norm)
            else:
                S_c.append(spectrum)
            lidar_vec = lidar_cube[row, col].reshape(-1)
            h_seed_c.append(lidar_vec)

        S_c = np.array(S_c)  # (M, D)
        h_seed_c = np.array(h_seed_c, dtype=np.float32)  # (M, C)

        # Compute SAM distances.
        SAM_c = compute_sam_matrix(X_unit, S_c)  # (N, M)

        # Find the nearest seed for each pixel.
        j_star = np.argmin(SAM_c, axis=1)  # (N,)
        sam_best = SAM_c[np.arange(len(X_unit)), j_star]  # (N,)
        delta_vectors = h_all - h_seed_c[j_star]
        delta_h_best = np.linalg.norm(delta_vectors, axis=1)  # (N,)
        
        # Apply both constraints.
        mask_sam = sam_best < T_sam
        mask_h = delta_h_best < T_h
        mask_pass = mask_sam & mask_h

        sam_pass = int(np.sum(mask_sam))
        h_pass = int(np.sum(mask_h))
        both_pass = int(np.sum(mask_pass))
        print(f"  Constraint counts -> SAM: {sam_pass}, height: {h_pass}, both: {both_pass}")
        
        # Compute the combined score; lower is better.
        score = (sam_best / (np.pi / 2)) + lambda_epg * (delta_h_best / h_range)
        
        # Exclude gold and test regions.
        candidate_global_mask = valid_mask.copy()
        candidate_global_mask[gold_train_mask] = False
        if exclude_test_from_candidates:
            candidate_global_mask[test_mask] = False
        
        # Map the local mask to global coordinates.
        candidate_local_indices = []
        for i in range(len(all_coords)):
            coord = all_coords[i]
            row, col = int(coord[0]), int(coord[1])
            # Convert explicitly to bool to avoid array ambiguity.
            is_candidate = bool(candidate_global_mask[row, col])
            # Handle NumPy scalars and one-dimensional slices.
            _v = mask_pass[i]
            if isinstance(_v, np.ndarray):
                passed_epg = bool(_v.ravel()[0])
            else:
                passed_epg = bool(_v)
            if is_candidate and passed_epg:
                candidate_local_indices.append(i)
        candidate_local_indices = np.array(candidate_local_indices, dtype=np.int64)
        
        if len(candidate_local_indices) == 0:
            print(f"\n{class_key} (class {class_id}): no samples passed EPG gating")
            selected_per_class[class_id] = 0
            continue
        
        # Select Top-K using either a fixed count or a ratio.
        candidate_scores = score[candidate_local_indices]
        if 0 < topk_per_class < 1:
            k = int(np.ceil(len(candidate_local_indices) * topk_per_class))
            k = max(1, min(k, len(candidate_local_indices)))
        else:
            k = min(int(topk_per_class), len(candidate_local_indices))
        topk_indices_in_candidate = np.argsort(candidate_scores)[:k]
        selected_local_indices = candidate_local_indices[topk_indices_in_candidate]
        
        # Mark selected samples.
        for idx in selected_local_indices:
            row, col = int(all_coords[idx][0]), int(all_coords[idx][1])
            epg_selection_mask[row, col] = True
            epg_labels[row, col] = class_id
            epg_scores[row, col] = score[idx]
        
        selected_per_class[class_id] = k
        print(f"\n{class_key} (class {class_id}): {len(candidate_local_indices)} candidates, {k} selected")
        print(f"  SAM range: [{sam_best[selected_local_indices].min():.4f}, {sam_best[selected_local_indices].max():.4f}]")
        print(f"  Height-difference range: [{delta_h_best[selected_local_indices].min():.4f}, {delta_h_best[selected_local_indices].max():.4f}]")
    
    # Build the training pool from gold and EPG samples.
    training_pool = {
        'coords': [],
        'labels': [],
        'confidences': [],
        'status': []
    }
    
    # Add gold samples.
    gold_coords = np.argwhere(gold_train_mask)
    for coord in gold_coords:
        row, col = int(coord[0]), int(coord[1])
        training_pool['coords'].append([row, col])
        training_pool['labels'].append(int(gt_map[row, col]))
        training_pool['confidences'].append(1.0)
        training_pool['status'].append('gold')
    
    # Add EPG samples.
    epg_coords = np.argwhere(epg_selection_mask)
    for coord in epg_coords:
        row, col = int(coord[0]), int(coord[1])
        training_pool['coords'].append([row, col])
        training_pool['labels'].append(int(epg_labels[row, col]))
        training_pool['confidences'].append(1.0)
        training_pool['status'].append('epg')
    
    # Convert lists to NumPy arrays.
    training_pool['coords'] = np.array(training_pool['coords'])
    training_pool['labels'] = np.array(training_pool['labels'])
    training_pool['confidences'] = np.array(training_pool['confidences'])
    training_pool['status'] = np.array(training_pool['status'])
    
    # Build the preparation pool.
    preparation_mask = valid_mask.copy()
    preparation_mask[gold_train_mask] = False
    if exclude_test_from_candidates:
        preparation_mask[test_mask] = False
    preparation_mask[epg_selection_mask] = False
    
    print(f"\n========== EPG Selection Complete ==========")
    print(f"Gold samples: {len(gold_coords)}")
    print(f"Selected EPG samples: {len(epg_coords)}")
    print(f"Total training-pool samples: {len(training_pool['coords'])}")
    print(f"Preparation-pool samples: {np.sum(preparation_mask)}")
    
    # Save EPG selections.
    if selection_mask_path:
        os.makedirs(os.path.dirname(selection_mask_path), exist_ok=True)
        np.save(selection_mask_path, epg_selection_mask)
    if selection_labels_path:
        os.makedirs(os.path.dirname(selection_labels_path), exist_ok=True)
        np.save(selection_labels_path, epg_labels)
    
    return training_pool, preparation_mask

# ==================== CPST ====================

class CPSTPool:
    """Manage training and preparation pools during CPST."""
    
    def __init__(self, training_pool, preparation_mask, hsi_cube, lidar_map, gt_map):
        """
        Args:
            training_pool: Initial training-pool dictionary.
            preparation_mask: Preparation-pool mask.
            hsi_cube: HSI data with shape (H, W, D).
            lidar_map: LiDAR data with shape (H, W) or (H, W, C).
            gt_map: Ground-truth labels used for diagnostics.
        """
        self.training_pool = training_pool
        self.preparation_mask = preparation_mask.copy()
        self.hsi_cube = hsi_cube
        self.lidar_map = lidar_map
        self.gt_map = gt_map
        
        # Cache preparation-pool coordinates.
        self.preparation_coords = np.argwhere(preparation_mask)
        
        print("Stage 2: Collaborative Progressive Self-Training")
        print("CPSTPool initialized:")
        print(f"  Training pool: {len(self.training_pool['coords'])}")
        print(f"  Preparation pool: {len(self.preparation_coords)}")
    
    def get_training_size(self):
        return len(self.training_pool['coords'])
    
    def get_preparation_size(self):
        return len(self.preparation_coords)
    
    def sample_from_preparation(self, sample_size=10000):
        """Randomly sample coordinates from the preparation pool."""
        if len(self.preparation_coords) == 0:
            return np.array([]), np.array([])
        
        sample_size = min(sample_size, len(self.preparation_coords))
        indices = np.random.choice(len(self.preparation_coords), sample_size, replace=False)
        sampled_coords = self.preparation_coords[indices]
        
        return sampled_coords, indices
    
    def add_to_training(self, coords, pseudo_labels, confidences):
        """Move samples from the preparation pool to the training pool."""
        # Add samples to the training pool.
        new_coords = coords.tolist()
        new_labels = pseudo_labels.tolist()
        new_confidences = confidences.tolist()
        new_status = ['cou'] * len(coords)
        
        self.training_pool['coords'] = np.concatenate([self.training_pool['coords'], coords])
        self.training_pool['labels'] = np.concatenate([self.training_pool['labels'], pseudo_labels])
        self.training_pool['confidences'] = np.concatenate([self.training_pool['confidences'], confidences])
        self.training_pool['status'] = np.concatenate([self.training_pool['status'], new_status])
        
      
        try:
            self.last_cou_added = {
                'coords': np.array(coords, dtype=np.int64),
                'labels': np.array(pseudo_labels, dtype=np.int64)
            }
        except Exception:
            pass
        
        # Remove samples from the preparation pool.
        for row, col in coords:
            self.preparation_mask[row, col] = False
        
        self.preparation_coords = np.argwhere(self.preparation_mask)
    
    def update_training_pool(self, indices, new_labels, new_confidences):
        """Update pseudo-labels and confidence scores during FIU."""
        for offset, idx in enumerate(indices):
            if self.training_pool['status'][idx] != 'gold':
                self.training_pool['labels'][idx] = new_labels[offset]
                self.training_pool['confidences'][idx] = new_confidences[offset]

# Backward-compatible alias.
ActiveLearningPool = CPSTPool
def get_confidence_threshold(epoch, total_epochs=140):
    """Return the dynamic confidence threshold for an epoch."""
    progress = epoch / total_epochs
    
    if progress < 0.3:
        return 0.99
    elif progress < 0.6:
        return 0.97
    elif progress < 0.85:
        return 0.95
    else:
        return 0.92

def cou_step(model, pool, device, conf_threshold=0.95, quota_per_class=200, patch_size=11, sample_size=10000):
    """
    Confidence-based Online Update
    Move high-confidence samples from the preparation pool to the training pool.
    
    Args:
        model: Current model.
        pool: CPSTPool instance.
        device: Inference device.
        conf_threshold: Confidence threshold.
        quota_per_class: Maximum samples per class.
        patch_size: Spatial patch size.
    
    Returns:
        added_count: Number of added samples.
    """
    model.eval()
    
    # Sample from the preparation pool.
    sampled_coords, sample_indices = pool.sample_from_preparation(sample_size=sample_size)
    
    if len(sampled_coords) == 0:
        return 0
    
    # Extract patches and predict labels.
    predictions = []
    confidences = []
    
    with torch.no_grad():
        # Run batched inference.
        batch_size = 256
        for i in range(0, len(sampled_coords), batch_size):
            batch_coords = sampled_coords[i:i+batch_size]
            
            # Extract patches.
            hsi_patches = []
            lidar_patches = []
            
            pad = patch_size // 2
            hsi_padded = np.pad(pool.hsi_cube, ((pad, pad), (pad, pad), (0, 0)), 'constant')
            
            # Preserve the LiDAR channel dimension.
            lidar_map = pool.lidar_map
            if lidar_map.ndim == 2:
                lidar_map = np.expand_dims(lidar_map, axis=-1)
            lidar_padded = np.pad(lidar_map, ((pad, pad), (pad, pad), (0, 0)), 'constant')
            
            for coord in batch_coords:
                row, col = int(coord[0]), int(coord[1])
                hsi_patch = hsi_padded[row:row+patch_size, col:col+patch_size, :]
                lidar_patch = lidar_padded[row:row+patch_size, col:col+patch_size, :]
                
                # Transpose to (C, H, W).
                hsi_patch = np.transpose(hsi_patch, (2, 0, 1))
                lidar_patch = np.transpose(lidar_patch, (2, 0, 1))
                
                hsi_patches.append(hsi_patch)
                lidar_patches.append(lidar_patch)
            
            hsi_batch = torch.FloatTensor(np.array(hsi_patches)).to(device)
            lidar_batch = torch.FloatTensor(np.array(lidar_patches)).to(device)
            
            # Predict labels and confidence scores.
            outputs = model(hsi_batch, lidar_batch)
            probs = torch.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            confidences.extend(max_probs.cpu().numpy())
    
    predictions = np.array(predictions)
    confidences = np.array(confidences)
    
    # Filter high-confidence samples.
    high_conf_mask = confidences > conf_threshold
    high_conf_coords = sampled_coords[high_conf_mask]
    high_conf_preds = predictions[high_conf_mask]
    high_conf_confs = confidences[high_conf_mask]
    
    if len(high_conf_coords) == 0:
        return 0
    
    # Apply per-class quotas.
    added_coords = []
    added_labels = []
    added_confs = []
    
    for class_id in range(1, 16):  # Houston has 15 classes.
        class_mask = high_conf_preds == (class_id - 1)
        class_coords = high_conf_coords[class_mask]
        class_confs = high_conf_confs[class_mask]
        
        if len(class_coords) == 0:
            continue
        
        # Select Top-K by confidence.
        if 0 < float(quota_per_class) < 1:
            k = int(np.ceil(float(quota_per_class) * len(class_coords)))
            k = max(1, min(k, len(class_coords)))
        else:
            k = int(min(int(quota_per_class), len(class_coords)))
        topk_indices = np.argsort(class_confs)[::-1][:k]
        
        added_coords.append(class_coords[topk_indices])
        added_labels.append(np.full(k, class_id))
        added_confs.append(class_confs[topk_indices])
    
    if len(added_coords) == 0:
        return 0
    
    added_coords = np.concatenate(added_coords)
    added_labels = np.concatenate(added_labels)
    added_confs = np.concatenate(added_confs)
    
    # Add selected samples to the training pool.
    pool.add_to_training(added_coords, added_labels, added_confs)
    
    return len(added_coords)

def fiu_step(model, pool, device, alpha=0.3, sample_size=5000, patch_size=11):
    """
    Update pseudo-labels and confidence scores with EMA smoothing.
    
    Args:
        model: Current model.
        pool: CPSTPool instance.
        device: Inference device.
        alpha: EMA smoothing coefficient.
        sample_size: Number of samples to update.
        patch_size: Spatial patch size.
    
    Returns:
        updated_count: Number of updated samples.
    """
    model.eval()
    
    # Randomly sample non-gold entries.
    non_gold_mask = pool.training_pool['status'] != 'gold'
    non_gold_indices = np.where(non_gold_mask)[0]
    
    if len(non_gold_indices) == 0:
        return 0
    
    sample_size = min(sample_size, len(non_gold_indices))
    sampled_indices = np.random.choice(non_gold_indices, sample_size, replace=False)
    sampled_coords = pool.training_pool['coords'][sampled_indices]
    old_confidences = pool.training_pool['confidences'][sampled_indices]
    
    # Extract patches and predict labels.
    predictions = []
    confidences = []
    
    with torch.no_grad():
        batch_size = 256
        for i in range(0, len(sampled_coords), batch_size):
            batch_coords = sampled_coords[i:i+batch_size]
            
            # Extract patches.
            hsi_patches = []
            lidar_patches = []
            
            pad = patch_size // 2
            hsi_padded = np.pad(pool.hsi_cube, ((pad, pad), (pad, pad), (0, 0)), 'constant')
            
            # Preserve the LiDAR channel dimension.
            lidar_map = pool.lidar_map
            if lidar_map.ndim == 2:
                lidar_map = np.expand_dims(lidar_map, axis=-1)
            lidar_padded = np.pad(lidar_map, ((pad, pad), (pad, pad), (0, 0)), 'constant')
            
            for coord in batch_coords:
                row, col = int(coord[0]), int(coord[1])
                hsi_patch = hsi_padded[row:row+patch_size, col:col+patch_size, :]
                lidar_patch = lidar_padded[row:row+patch_size, col:col+patch_size, :]
                
                hsi_patch = np.transpose(hsi_patch, (2, 0, 1))
                lidar_patch = np.transpose(lidar_patch, (2, 0, 1))
                
                hsi_patches.append(hsi_patch)
                lidar_patches.append(lidar_patch)
            
            hsi_batch = torch.FloatTensor(np.array(hsi_patches)).to(device)
            lidar_batch = torch.FloatTensor(np.array(lidar_patches)).to(device)
            
            outputs = model(hsi_batch, lidar_batch)
            probs = torch.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            confidences.extend(max_probs.cpu().numpy())
    
    predictions = np.array(predictions)
    new_confidences = np.array(confidences)
    
    # Apply EMA smoothing.
    smoothed_confidences = alpha * new_confidences + (1 - alpha) * old_confidences
    new_labels = predictions + 1  # Convert predictions to one-based labels.
    
    # Update the training pool.
    pool.update_training_pool(sampled_indices, new_labels, smoothed_confidences)
    
    return len(sampled_indices)



