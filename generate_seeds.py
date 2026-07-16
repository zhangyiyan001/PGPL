# -*- coding:utf-8 -*-
"""Sample few-shot seeds from training masks for supported datasets."""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import scipy.io as sio

from config_epg import DATASET_CONFIG, EXPERIMENT_CONFIG, get_epg_config
from seed_utils import set_global_seed

# Update this path to the local dataset root.
DATA_ROOT = r'D:\Program Files (x86)\Anaconda\jupyter_path\dataset'


def _load_train_mask(dataset: str) -> tuple[np.ndarray, list[str]]:
    dataset = dataset.lower()
    if dataset == 'houston':
        mask = sio.loadmat(os.path.join(DATA_ROOT, 'Houston2013_Data/Houston2013_TR.mat'))['TR_map']
        names = DATASET_CONFIG['Houston']['class_names']
    elif dataset == 'trento':
        mask = sio.loadmat(os.path.join(DATA_ROOT, 'Trento/TRLabel.mat'))['TRLabel']
        names = DATASET_CONFIG['Trento']['class_names']
    elif dataset == 'muufl':
        mask = sio.loadmat(os.path.join(DATA_ROOT, 'MUUFL/mask_train_150.mat'))['mask_train']
        names = DATASET_CONFIG['MUUFL']['class_names']
    else:
        raise ValueError(f'Unsupported dataset: {dataset}')
    return mask, names


def generate_seeds(dataset: str, seeds_per_class: int) -> dict:
    train_mask, class_names = _load_train_mask(dataset)
    dataset_upper = dataset.capitalize()
    num_classes = len(class_names)

    print(f"Generating seeds from the {dataset_upper} training set...")
    print(f"Sampling {seeds_per_class} seeds per class\n")

    seeds = {}
    for class_id in range(1, num_classes + 1):
        coords = np.argwhere(train_mask == class_id)
        class_name = class_names[class_id - 1]
        if len(coords) == 0:
            print(f"Warning: class {class_id} ({class_name}) has no training samples")
            seeds[f'class_{class_id}'] = []
            continue
        if len(coords) <= seeds_per_class:
            selected = coords
            print(f"Warning: class {class_id} ({class_name}) has only {len(coords)} samples; using all of them")
        else:
            indices = random.sample(range(len(coords)), seeds_per_class)
            selected = coords[indices]
        seeds[f'class_{class_id}'] = selected.tolist()
        print(f"Class {class_id:2d} ({class_name:20s}): selected {len(selected)} seeds")

    return seeds


def save_seeds(seeds: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(seeds, f, indent=2, ensure_ascii=False)
    print(f"\nSeeds saved to: {output_path}")


def load_seeds(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_seeds_from_json(seeds_path: str = 'seeds.json') -> dict:
    """Load a seed JSON file through the legacy-compatible interface."""
    return load_seeds(seeds_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate EPG seeds for a supported dataset')
    parser.add_argument('--dataset', required=True, choices=[k for k in DATASET_CONFIG.keys()], help='Dataset name')
    parser.add_argument('--seeds-per-class', type=int, help='Seeds per class (defaults to config_epg.py)')
    parser.add_argument('--output', help='Output path (defaults to the configured seeds_path)')
    parser.add_argument(
        '--seed',
        type=int,
        default=EXPERIMENT_CONFIG.get('random_seed', 2021),
        help='Random seed (defaults to config_epg.EXPERIMENT_CONFIG)'
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    dataset_name = args.dataset
    epg_cfg = get_epg_config(dataset_name)
    seeds_per_class = args.seeds_per_class or epg_cfg.get('seeds_per_class', 5)
    output_path = args.output or epg_cfg.get('seeds_path', f'seeds_{dataset_name.lower()}.json')

    seeds = generate_seeds(dataset_name, seeds_per_class)
    save_seeds(seeds, output_path)

    loaded = load_seeds(output_path)
    total = sum(len(v) for v in loaded.values())
    print(f"Generated {total} seeds across {len(loaded)} classes")


if __name__ == '__main__':
    main()
