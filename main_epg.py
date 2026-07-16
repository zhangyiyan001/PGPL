# -*- coding:utf-8 -*-
"""Three-stage EPG and CPST pipeline with dataset selection."""

from __future__ import annotations

import argparse
import os
import numpy as np
import json
import torch

from dataset_epg import load_data
from dataset_epg import create_test_dataloader
from epg_module import compute_normalization_stats, normalize_with_stats, epg_sample_selection
from generate_seeds import load_seeds_from_json
from test import result, evaluate_model_metrics, log_metrics_to_file
from train_epg import train_cpst, train_epg_warmup, train_refinement
from seed_utils import set_global_seed
from config_epg import (
    DATASET_CONFIG,
    get_dataset_config,
    get_epg_config,
    get_output_config,
    get_training_config,
    DEVICE_CONFIG,
    EXPERIMENT_CONFIG,
    print_config,
)


def full_inference(
    model: torch.nn.Module,
    hsi_cube: np.ndarray,
    lidar_map: np.ndarray,
    valid_mask: np.ndarray,
    gt_map: np.ndarray,
    device: torch.device,
    patch_size: int = 11,
    batch_size: int = 256,
    save_path: str = 'results/full_inference_map.npy'
):
    """Run full-scene inference and return label and probability maps."""

    print("\n" + "=" * 60)
    print("Full-scene inference")
    print("=" * 60)

    H, W, _ = hsi_cube.shape
    n_classes = int(np.max(gt_map))

    model.eval()
    valid_coords = np.argwhere(valid_mask)
    print(f"Valid pixels: {len(valid_coords)}")

    prob_map = np.zeros((H, W, n_classes), dtype=np.float32)

    pad = patch_size // 2
    hsi_padded = np.pad(hsi_cube, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
    if lidar_map.ndim == 2:
        lidar_map = np.expand_dims(lidar_map, axis=-1)
    lidar_padded = np.pad(lidar_map, ((pad, pad), (pad, pad), (0, 0)), mode='constant')

    with torch.no_grad():
        for start in range(0, len(valid_coords), batch_size):
            batch_coords = valid_coords[start:start + batch_size]
            hsi_patches, lidar_patches = [], []
            for row, col in batch_coords:
                row, col = int(row), int(col)
                hsi_patch = hsi_padded[row:row + patch_size, col:col + patch_size, :]
                lidar_patch = lidar_padded[row:row + patch_size, col:col + patch_size, :]
                hsi_patches.append(np.transpose(hsi_patch, (2, 0, 1)))
                lidar_patches.append(np.transpose(lidar_patch, (2, 0, 1)))

            hsi_batch = torch.from_numpy(np.array(hsi_patches, dtype=np.float32)).to(device)
            lidar_batch = torch.from_numpy(np.array(lidar_patches, dtype=np.float32)).to(device)
            outputs = model(hsi_batch, lidar_batch)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            for (row, col), prob in zip(batch_coords, probs):
                prob_map[int(row), int(col), :] = prob

            if (start // batch_size + 1) % 10 == 0:
                print(f"  Processed: {start + len(batch_coords)}/{len(valid_coords)}")

    result_map = np.zeros((H, W), dtype=int)
    result_map[valid_mask] = np.argmax(prob_map[valid_mask], axis=1) + 1
    result_map[gt_map == 0] = 0

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, result_map)
    np.save(save_path.replace('.npy', '_prob.npy'), prob_map)

    print("Full-scene inference complete!")
    print(f"Results saved to: {save_path}")
    return result_map, prob_map


def positive_odd_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('patch size must be an integer') from exc
    if parsed <= 0 or parsed % 2 == 0:
        raise argparse.ArgumentTypeError('patch size must be a positive odd integer')
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run EPG + CPST pipeline')
    parser.add_argument(
        '--dataset',
        choices=sorted(DATASET_CONFIG.keys()),
        help='Dataset to run (defaults to config_epg.CURRENT_DATASET)'
    )
    parser.add_argument(
        '--patch-size',
        type=positive_odd_int,
        help='Spatial patch size (recommended: Houston=11, Trento=7, MUUFL=9)'
    )
    return parser.parse_args()


def ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def main() -> None:
    args = parse_args()
    dataset_config = get_dataset_config(args.dataset)
    dataset_name = dataset_config['name']
    epg_config = get_epg_config(dataset_name)
    training_config = get_training_config(dataset_name)
    if args.patch_size is not None:
        training_config['patch_size'] = args.patch_size
    output_config = get_output_config(dataset_name)
    device_config = DEVICE_CONFIG
    balance_classes = EXPERIMENT_CONFIG.get('balance_classes', True)
    num_workers = device_config.get('num_workers', 0)
    save_intermediate = EXPERIMENT_CONFIG.get('save_intermediate', True)
    log_interval = EXPERIMENT_CONFIG.get('log_interval', 10)
    model_dropout = training_config.get('dropout', 0.0)

    ensure_parent_dir(output_config['training_pool_history_path'])
    ensure_parent_dir(output_config['log_dir'])
    os.makedirs(output_config['log_dir'], exist_ok=True)
    os.makedirs(os.path.join('results', dataset_name), exist_ok=True)

    hyperparams_snapshot = {
        'dataset_config': dataset_config,
        'epg_config': epg_config,
        'training_config': training_config,
        'experiment_config': EXPERIMENT_CONFIG,
        'device_config': device_config
    }
    hyperparams_text = json.dumps(hyperparams_snapshot, ensure_ascii=False, indent=2)

    print_config(dataset_name, training_config)

    # Set the random seed.
    seed = EXPERIMENT_CONFIG['random_seed']
    set_global_seed(seed, deterministic=False)

    use_gpu = device_config.get('use_gpu', True)
    gpu_id = device_config.get('gpu_id', 0)
    if use_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')
    print(f"\nDevice: {device}\n")

    # Load data.
    print("\n" + "=" * 60)
    print("Stage 0: Data Loading and Preparation")
    print("=" * 60)
    print("Loading data...")
    HSI_data, LiDAR_data, Train_data, Test_data, GT = load_data(dataset_name)
    print(f"HSI shape: {HSI_data.shape}")
    print(f"LiDAR shape: {LiDAR_data.shape}")
    print(f"GT shape: {GT.shape}")

    valid_mask = (GT != 0)
    original_gold_mask = (Train_data != 0)
    test_mask = (Test_data != 0)

    print(f"Valid pixels: {int(np.sum(valid_mask))}")
    print(f"Gold training samples (original TR): {int(np.sum(original_gold_mask))}")

    # Normalization statistics.
    stats_path = output_config['normalization_stats_path']
    ensure_parent_dir(stats_path)
    if not os.path.exists(stats_path):
        print("\nComputing normalization statistics...")
        compute_normalization_stats(HSI_data, LiDAR_data, valid_mask, stats_path)
    else:
        print(f"\nNormalization statistics already exist: {stats_path}")

    print("Applying z-score normalization...")
    HSI_norm, LiDAR_norm = normalize_with_stats(HSI_data, LiDAR_data, stats_path)
    print("Normalization complete")

    test_coords = np.argwhere(test_mask)
    test_labels = GT[test_mask]
    print(f"Test samples: {len(test_coords)}")
    test_loader = create_test_dataloader(
        test_coords=test_coords,
        test_labels=test_labels,
        hsi_cube=HSI_norm,
        lidar_map=LiDAR_norm,
        batch_size=64,
        patch_size=training_config['patch_size'],
        num_workers=num_workers
    )

    # Load seeds.
    print("\n" + "=" * 60)
    print("Stage 1: Build the Initial Training Pool with EPG")
    print("=" * 60)
    seeds_path = epg_config['seeds_path']
    if not os.path.exists(seeds_path):
        print(f"Error: seed file not found: {seeds_path}")
        print("Run generate_seeds.py first to create the seed file")
        return
    print(f"Loading seed file: {seeds_path}")
    seeds_dict = load_seeds_from_json(seeds_path)

    gold_train_mask = np.zeros_like(original_gold_mask, dtype=bool)
    seed_total = 0
    seed_class_counts = {}
    for class_key, coords in seeds_dict.items():
        if len(coords) == 0:
            continue
        coords_array = np.array(coords, dtype=int)
        if coords_array.ndim == 1:
            coords_array = coords_array.reshape(1, -1)
        rows, cols = coords_array[:, 0], coords_array[:, 1]
        gold_train_mask[rows, cols] = True
        seed_total += len(coords)
        seed_class_counts[class_key] = len(coords)

    if seed_total == 0:
        print("Warning: seeds.json contains no seed labels; fallback to original TR labels.")
        gold_train_mask = original_gold_mask.copy()
    else:
        print(f"Gold samples (seeds only): {seed_total}")
        for class_key in sorted(seed_class_counts.keys(), key=lambda k: int(k.split('_')[1])):
            print(f"  {class_key}: {seed_class_counts[class_key]}")

    print(f"Gold labeled pixels: {int(np.count_nonzero(gold_train_mask))}")

    exclude_test_from_candidates = (not epg_config.get('use_test_as_unlabeled', False))

    ensure_parent_dir(output_config['epg_selection_mask_path'])
    training_pool, preparation_mask = epg_sample_selection(
        hsi_cube=HSI_norm,
        lidar_map=LiDAR_norm,
        gt_map=GT,
        seeds_dict=seeds_dict,
        gold_train_mask=gold_train_mask,
        test_mask=test_mask,
        T_sam=epg_config['T_sam'],
        T_h=epg_config['T_h'],
        topk_per_class=epg_config['topk_per_class'],
        lambda_epg=epg_config['lambda_epg'],
        exclude_test_from_candidates=exclude_test_from_candidates,
        selection_mask_path=output_config['epg_selection_mask_path'],
        selection_labels_path=output_config['epg_labels_path'],
    )

    if epg_config.get('use_test_as_unlabeled', False) and os.path.exists(output_config['epg_selection_mask_path']):
        try:
            epg_mask = np.load(output_config['epg_selection_mask_path'])
            before = int(np.sum(test_mask))
            test_mask = np.logical_and(test_mask, np.logical_not(epg_mask))
            after = int(np.sum(test_mask))
            print(f"\nRemoved selected EPG samples from the test mask: {before - after} / {before} ({after} remaining)")
        except Exception as exc:
            print(f"Failed to update the EPG test mask: {exc}")

    # Stage 1.5: warm-up.
    warmup_cfg = training_config['warmup']
    for path in [
        warmup_cfg['save_path'],
        training_config['active_learning']['save_path'],
        training_config['refinement']['save_path'],
    ]:
        ensure_parent_dir(path)

    model = train_epg_warmup(
        training_pool=training_pool,
        hsi_cube=HSI_norm,
        lidar_map=LiDAR_norm,
        device=device,
        input_channels_hsi=dataset_config['input_channels_hsi'],
        input_channels_lidar=dataset_config['input_channels_lidar'],
        n_classes=dataset_config['n_classes'],
        patch_size=training_config['patch_size'],
        epochs=warmup_cfg['epochs'],
        batch_size=training_config['batch_size'],
        lr=warmup_cfg['lr'],
        dropout_rate=model_dropout,
        weight_decay=warmup_cfg['weight_decay'],
        save_path=warmup_cfg['save_path'],
        balance_classes=balance_classes,
        num_workers=num_workers,
        save_intermediate=save_intermediate
    )

    warmup_metrics = evaluate_model_metrics(
        model=model,
        test_iter=test_loader,
        dataset=dataset_name,
        device=device
    )
    print("\n" + "-" * 60)
    print("Test Evaluation: Stage 1.5 (EPG Warm-up)")
    print("-" * 60)
    print(f"Overall Accuracy: {warmup_metrics['oa']:.4f}")
    print(f"Average Accuracy: {warmup_metrics['aa']:.4f}")
    print(f"Kappa: {warmup_metrics['kappa']:.4f}")
    log_metrics_to_file(
        warmup_metrics,
        dataset_name,
        stage_name='Warmup',
        hyperparams=hyperparams_text
    )

    # Stage 2: collaborative progressive self-training.
    active_cfg = training_config['active_learning']
    model, final_pool = train_cpst(
        model=model,
        training_pool=training_pool,
        preparation_mask=preparation_mask,
        hsi_cube=HSI_norm,
        lidar_map=LiDAR_norm,
        gt_map=GT,
        device=device,
        input_channels_hsi=dataset_config['input_channels_hsi'],
        input_channels_lidar=dataset_config['input_channels_lidar'],
        n_classes=dataset_config['n_classes'],
        patch_size=training_config['patch_size'],
        epochs=active_cfg['epochs'],
        batch_size=training_config['batch_size'],
        lr=active_cfg['lr'],
        weight_decay=active_cfg['weight_decay'],
        cou_interval=active_cfg['cou_interval'],
        fiu_interval=active_cfg['fiu_interval'],
        cou_sample_size=active_cfg['cou_sample_size'],
        cou_quota_per_class=active_cfg['cou_quota_per_class'],
        fiu_sample_size=active_cfg['fiu_sample_size'],
        fiu_alpha=active_cfg['fiu_alpha'],
        save_path=active_cfg['save_path'],
        history_path=output_config['training_pool_history_path'],
        balance_classes=balance_classes,
        num_workers=num_workers,
        log_interval=log_interval,
        save_intermediate=save_intermediate,
        test_loader=test_loader,
        dataset_name=dataset_name,
        eval_interval=active_cfg.get('eval_interval')
    )


    # Stage 3: refinement.
    refine_cfg = training_config['refinement']
    model = train_refinement(
        model=model,
        training_pool=final_pool.training_pool,
        hsi_cube=HSI_norm,
        lidar_map=LiDAR_norm,
        device=device,
        input_channels_hsi=dataset_config['input_channels_hsi'],
        input_channels_lidar=dataset_config['input_channels_lidar'],
        n_classes=dataset_config['n_classes'],
        patch_size=training_config['patch_size'],
        epochs=refine_cfg['epochs'],
        batch_size=training_config['batch_size'],
        lr=refine_cfg['lr'],
        weight_decay=refine_cfg['weight_decay'],
        save_path=refine_cfg['save_path'],
        balance_classes=balance_classes,
        num_workers=num_workers,
        save_intermediate=save_intermediate
    )


    # Test evaluation.
    print("\n" + "=" * 60)
    print("Test Evaluation")
    print("=" * 60)
    print(f"Test samples: {len(test_coords)}")

    ensure_parent_dir(output_config['training_pool_history_path'])
    ensure_parent_dir(output_config['log_dir'])
    os.makedirs(output_config['log_dir'], exist_ok=True)
    os.makedirs(os.path.join('results', dataset_name), exist_ok=True)
    final_model_for_eval = f'./models/{dataset_name}.pt'
    ensure_parent_dir(final_model_for_eval)
    model_state_for_eval = model.state_dict()
    torch.save(model_state_for_eval, final_model_for_eval)
    model.load_state_dict(model_state_for_eval)
    result(test_loader, dataset_name, device, model, stage_name='Final', hyperparams=hyperparams_text)

    # Full-scene inference.
    result_map, prob_map = full_inference(
        model=model,
        hsi_cube=HSI_norm,
        lidar_map=LiDAR_norm,
        valid_mask=valid_mask,
        gt_map=GT,
        device=device,
        patch_size=training_config['patch_size'],
        batch_size=256,
        save_path=output_config['full_inference_map_path']
    )

    # Visualize the classification map when possible.
    try:
        from module import dis_groundtruth

        dis_groundtruth(
            dataset=dataset_name,
            num_class=dataset_config['n_classes'],
            gt=result_map,
            p=True
        )
    except Exception as exc:
        print(f"Visualization failed: {exc}")

    print("\n" + "=" * 60)
    print("Training pipeline complete!")
    print("=" * 60)
    print("Model paths:")
    print(f"  - Warm-up: {training_config['warmup']['save_path']}")
    print(f"  - CPST: {training_config['active_learning']['save_path']}")
    print(f"  - Refinement: {training_config['refinement']['save_path']}")
    print("Result paths:")
    print(f"  - Full-scene prediction: {output_config['full_inference_map_path']}")
    print(f"  - Training-pool history: {output_config['training_pool_history_path']}")
    print(f"  - EPG selection mask: {output_config['epg_selection_mask_path']}")
    print(f"  - EPG labels: {output_config['epg_labels_path']}")


if __name__ == '__main__':
    main()

