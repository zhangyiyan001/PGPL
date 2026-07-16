# -*- coding:utf-8 -*-

from __future__ import annotations

import copy

# ==================== Dataset Information ====================
DATASET_CONFIG = {
    'Houston': {
        'name': 'Houston',
        'input_channels_hsi': 144,
        'input_channels_lidar': 1,
        'n_classes': 15,
        'class_names': [
            'Healthy grass', 'Stressed grass', 'Synthetic grass', 'Tree',
            'Soil', 'Water', 'Residential', 'Commercial', 'Road', 'Highway',
            'Railway', 'Parking lot 1', 'Parking lot 2', 'Tennis court', 'Running track'
        ]
    },
    'Trento': {
        'name': 'Trento',
        'input_channels_hsi': 63,
        'input_channels_lidar': 1,
        'n_classes': 6,
        'class_names': [
            'Apple trees', 'Buildings', 'Ground', 'Wood', 'Vineyard', 'Roads'
        ]
    },
    'MUUFL': {
        'name': 'MUUFL',
        'input_channels_hsi': 64,
        'input_channels_lidar': 2,
        'n_classes': 11,
        'class_names': [
            'Trees', 'Mostly grass', 'Mixed ground surface', 'Dirt and sand',
            'Road', 'Water', 'Building shadow', 'Building', 'Sidewalk',
            'Yellow curb', 'Cloth panels'
        ]
    }
}

# Default dataset used when the CLI option is omitted
CURRENT_DATASET = 'MUUFL'

# ==================== EPG Parameters ====================
BASE_EPG_CONFIG = {}

EPG_CONFIG_OVERRIDES = {
    'Houston': {
        'T_sam': 0.15, #0.15
        'T_h': 0.3,  #0.3
        'topk_per_class': 0.1,
        'use_test_as_unlabeled': False,
        'seeds_path': 'seeds.json',
        'lambda_epg': 0.04,
        'seeds_per_class': 2
    },
    'Trento': {
        'T_sam': 0.35, #0.35
        'T_h': 0.6, #0.6
        'topk_per_class': 0.1,
        'use_test_as_unlabeled': False,
        'seeds_path': 'seeds_trento.json',
        'lambda_epg': 0.04,
        'seeds_per_class': 5
    },
    'MUUFL': {
        'T_sam': 0.24, #0.24
        'T_h': 2.0,  #2.0
        'topk_per_class': 0.1,
        'use_test_as_unlabeled': False,
        'seeds_path': 'seeds_muufl.json',
        'lambda_epg': 0.04,
        'seeds_per_class': 2
    }
}

# ==================== Training Parameters ====================
BASE_TRAINING_CONFIG = { #Houston 11, Trento 7, MUUFL 9
    'patch_size': 9,
    'batch_size': 32,
    'dropout': 0.1,
    'warmup': {
        'epochs': 100,
        'lr': 0.001,
        'weight_decay': 0.0001,
        'save_path': './models/{dataset}_epg_warmup.pt'
    },
    'active_learning': {
        'epochs': 200,
        'lr': 0.0005,
        'weight_decay': 0.0001,
        'cou_interval': 15,
        'fiu_interval': 15,
        'cou_sample_size': 300,
        'cou_quota_per_class': 100,
        'fiu_sample_size': 300,
        'fiu_alpha': 0.3,
        'save_path': './models/{dataset}_epg_active.pt',
        'eval_interval': 0
    },
    'refinement': {
        'epochs': 10,
        'lr': 0.0001,
        'weight_decay': 0.0001,
        'save_path': './models/{dataset}_epg_final.pt'
    }
}

# ==================== Output Paths ====================
BASE_OUTPUT_CONFIG = {
    'normalization_stats_path': 'results/{dataset}/normalization_stats.json',
    'epg_selection_mask_path': 'results/{dataset}/epg_selection_mask.npy',
    'epg_labels_path': 'results/{dataset}/epg_labels.npy',
    'training_pool_history_path': 'results/{dataset}/training_pool_history.json',
    'full_inference_map_path': 'results/{dataset}/full_inference_map.npy',
    'full_inference_prob_path': 'results/{dataset}/full_inference_map_prob.npy',
    'log_dir': 'results/{dataset}/logs'
}

# ==================== Device and Experiment Settings ====================
DEVICE_CONFIG = {
    'use_gpu': True,
    'gpu_id': 0,
    'num_workers': 0
}

EXPERIMENT_CONFIG = {
    'random_seed': 2021,
    'save_intermediate': True,
    'log_interval': 10,
    'balance_classes': True
}

# ==================== Utilities ====================

def _deep_merge(base: dict, overrides: dict | None) -> dict:
    result = copy.deepcopy(base)
    if not overrides:
        return result
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def get_dataset_config(dataset_name: str | None = None) -> dict:
    dataset_name = dataset_name or CURRENT_DATASET
    return DATASET_CONFIG[dataset_name]


def get_epg_config(dataset_name: str | None = None) -> dict:
    dataset_name = dataset_name or CURRENT_DATASET
    cfg = EPG_CONFIG_OVERRIDES.get(dataset_name)
    if cfg is None:
        raise KeyError(f"Missing dataset configuration in EPG_CONFIG_OVERRIDES: {dataset_name}")
    return copy.deepcopy(cfg)


def get_training_config(dataset_name: str | None = None) -> dict:
    dataset_name = dataset_name or CURRENT_DATASET
    config = _deep_merge(BASE_TRAINING_CONFIG, {})
    for section in config.values():
        if isinstance(section, dict):
            for key, value in section.items():
                if isinstance(value, str) and '{dataset}' in value:
                    section[key] = value.format(dataset=dataset_name)
    return config


def get_output_config(dataset_name: str | None = None) -> dict:
    dataset_name = dataset_name or CURRENT_DATASET
    config = {}
    for key, value in BASE_OUTPUT_CONFIG.items():
        config[key] = value.format(dataset=dataset_name)
    return config


def print_config(dataset_name: str | None = None, training_config: dict | None = None) -> None:
    dataset_name = dataset_name or CURRENT_DATASET
    dataset = get_dataset_config(dataset_name)
    epg_config = get_epg_config(dataset_name)
    training_config = training_config or get_training_config(dataset_name)

    total_epochs = (
        training_config['warmup']['epochs']
        + training_config['active_learning']['epochs']
        + training_config['refinement']['epochs']
    )

    print('=' * 60)
    print('Current Configuration')
    print('=' * 60)
    print(f"\nDataset: {dataset['name']}")
    print(f"  HSI channels: {dataset['input_channels_hsi']}")
    print(f"  LiDAR channels: {dataset['input_channels_lidar']}")
    print(f"  Classes: {dataset['n_classes']}")

    print('\nEPG parameters:')
    print(f"  T_sam: {epg_config['T_sam']:.3f} rad ({epg_config['T_sam']*180/3.14159:.1f} deg)")
    print(f"  T_h: {epg_config['T_h']:.2f}")
    print(f"  Top-K per class: {epg_config['topk_per_class']}")
    print(f"  Seeds file: {epg_config['seeds_path']}")

    print('\nTraining parameters:')
    print(f"  Patch size: {training_config['patch_size']}")
    print(f"  Batch size: {training_config['batch_size']}")
    print(f"  Warm-up epochs (Stage 1): {training_config['warmup']['epochs']}")
    print(f"  CPST epochs (Stage 2): {training_config['active_learning']['epochs']}")
    print(f"  Refinement epochs (Stage 3): {training_config['refinement']['epochs']}")
    print(f"  Total epochs: {total_epochs}")
    print('=' * 60)


if __name__ == '__main__':
    print_config()
