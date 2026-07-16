# -*- coding:utf-8 -*-
"""
EPG datasets with dynamic training pools and class-balanced sampling.
"""
import os
import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as Data
from typing import Dict

def load_data(dataset):
    #data_path = r'/home/ubuntu/dataset_RS/Multisource/data'
    data_path = r'D:\Program Files (x86)\Anaconda\jupyter_path\dataset'
    if dataset == 'Houston': #HSI.shape (349, 1905, 144), LiDAR.shape (349, 1905) gt.shape (349, 1905)=
        HSI_data = sio.loadmat(os.path.join(data_path, 'Houston2013/HSI.mat'))['HSI']
        LiDAR_data = sio.loadmat(os.path.join(data_path, 'Houston2013_Data/Houston2013_DSM.mat'))['DSM']
        LiDAR_data = np.expand_dims(LiDAR_data, axis=-1)
        Train_data = sio.loadmat(os.path.join(data_path, 'Houston2013_Data/Houston2013_TR.mat'))['TR_map']
        Test_data = sio.loadmat(os.path.join(data_path, 'Houston2013_Data/Houston2013_TE.mat'))['TE_map']
        GT = sio.loadmat(os.path.join(data_path, 'Houston2013/gt.mat'))['gt']

    if dataset == 'Trento':
        HSI_data = sio.loadmat(os.path.join(data_path, 'Trento/HSI.mat'))['HSI']
        LiDAR_data = sio.loadmat(os.path.join(data_path, 'Trento/LiDAR.mat'))['LiDAR']
        LiDAR_data = np.expand_dims(LiDAR_data, axis=-1)
        Train_data = sio.loadmat(os.path.join(data_path, 'Trento/TRLabel.mat'))['TRLabel']
        Test_data = sio.loadmat(os.path.join(data_path, 'Trento/TSLabel.mat'))['TSLabel']
        GT = sio.loadmat(os.path.join(data_path, 'Trento/gt.mat'))['gt']

    if dataset == 'MUUFL':
        HSI_data = sio.loadmat(os.path.join(data_path, 'MUUFL/HSI.mat'))['HSI']
        LiDAR_data = sio.loadmat(os.path.join(data_path, 'MUUFL/LiDAR.mat'))['LiDAR']
        Train_data = sio.loadmat(os.path.join(data_path, 'MUUFL/mask_train_150.mat'))['mask_train']
        Test_data = sio.loadmat(os.path.join(data_path, 'MUUFL/mask_test_150.mat'))['mask_test']
        GT = sio.loadmat(os.path.join(data_path, 'MUUFL/gt.mat'))['gt']
        GT[GT==-1] = 0

    return HSI_data, LiDAR_data, Train_data, Test_data, GT

class EPGDataset(Data.Dataset):
    """PyTorch dataset backed by a dynamic training pool."""
    
    def __init__(self, training_pool, hsi_cube, lidar_map, patch_size=11, balance_classes=True):
        """
        Args:
            training_pool: Dictionary containing coords, labels, confidences, and status.
            hsi_cube: Normalized HSI data with shape (H, W, D).
            lidar_map: Normalized LiDAR data with shape (H, W) or (H, W, C).
            patch_size: Spatial patch size.
            balance_classes: Whether to use class-balanced sampling.
        """
        self.training_pool = training_pool
        self.hsi_cube = hsi_cube
        self.lidar_map = lidar_map
        self.patch_size = patch_size
        self.balance_classes = balance_classes
        
        # Pad the data before extracting patches.
        pad = patch_size // 2
        self.hsi_padded = np.pad(hsi_cube, ((pad, pad), (pad, pad), (0, 0)), 'constant')
        # Expand a 2D LiDAR map to preserve a channel dimension.
        if lidar_map.ndim == 2:
            lidar_map = np.expand_dims(lidar_map, axis=-1)
        self.lidar_padded = np.pad(lidar_map, ((pad, pad), (pad, pad), (0, 0)), 'constant')
        
        # Build per-class indices for balanced sampling.
        if balance_classes:
            self.class_indices = {}
            for class_id in range(1, 16):  # Houston has 15 classes.
                mask = training_pool['labels'] == class_id
                indices = np.where(mask)[0]
                if len(indices) > 0:
                    self.class_indices[class_id] = indices
            
            self.available_classes = list(self.class_indices.keys())
            self.num_classes = len(self.available_classes)
    
    def __len__(self):
        return len(self.training_pool['coords'])
    
    def __getitem__(self, idx):
        """Return one HSI-LiDAR sample."""
        coord = self.training_pool['coords'][idx]
        label = self.training_pool['labels'][idx]
        
        row, col = coord
        pad = self.patch_size // 2
        
        # Extract the HSI patch.
        hsi_patch = self.hsi_padded[row:row+self.patch_size, col:col+self.patch_size, :]
        hsi_patch = np.transpose(hsi_patch, (2, 0, 1))  # (D, H, W)
        
        # Extract the LiDAR patch.
        lidar_patch = self.lidar_padded[row:row+self.patch_size, col:col+self.patch_size, :]  # (H, W, C)
        lidar_patch = np.transpose(lidar_patch, (2, 0, 1))  # (C, H, W)
        
        # Convert arrays to tensors.
        hsi_tensor = torch.FloatTensor(hsi_patch)
        lidar_tensor = torch.FloatTensor(lidar_patch)
        label_tensor = torch.LongTensor([label - 1])[0]  # Convert labels to zero-based indices.
        
        return hsi_tensor, lidar_tensor, label_tensor
    
    def get_balanced_indices(self, num_samples):
        """
        Generate class-balanced sample indices.
        
        Args:
            num_samples: Total number of samples.
        
        Returns:
            indices: Array of sample indices.
        """
        if not self.balance_classes or self.num_classes == 0:
            return np.random.choice(len(self), num_samples, replace=True)
        
        # Determine the number of samples per class.
        samples_per_class = num_samples // self.num_classes
        remainder = num_samples % self.num_classes
        
        indices = []
        for i, class_id in enumerate(self.available_classes):
            class_indices = self.class_indices[class_id]
            
            # Assign one extra sample to the first remainder classes.
            n_samples = samples_per_class + (1 if i < remainder else 0)
            
            # Sample from the class with replacement.
            sampled = np.random.choice(class_indices, n_samples, replace=True)
            indices.extend(sampled)
        
        # Shuffle the sampled indices.
        indices = np.array(indices)
        np.random.shuffle(indices)
        
        return indices  

class BalancedBatchSampler(Data.Sampler):
    """Batch sampler for class-balanced sampling."""
    
    def __init__(self, dataset, batch_size, num_batches_per_epoch=None):
        """
        Args:
            dataset: EPGDataset instance.
            batch_size: Number of samples in each batch.
            num_batches_per_epoch: Batches per epoch, inferred from the dataset when None.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        # Use at least one batch and round up for a partial batch.
        if num_batches_per_epoch is None:
            self.num_batches_per_epoch = max(1, (len(dataset) + batch_size - 1) // batch_size)
        else:
            self.num_batches_per_epoch = max(1, int(num_batches_per_epoch))

        self.total_samples = self.num_batches_per_epoch * batch_size
    
    def __iter__(self):
        # Generate indices with the balanced sampling strategy.
        indices = self.dataset.get_balanced_indices(self.total_samples)
        for i in range(self.num_batches_per_epoch):
            batch = indices[i * self.batch_size:(i + 1) * self.batch_size]
            yield batch.tolist()
    
    def __len__(self):
        return self.num_batches_per_epoch

def create_epg_dataloader(training_pool, hsi_cube, lidar_map, 
                         batch_size=64, patch_size=11, 
                         balance_classes=True, num_workers=0):
    """
    Create an EPG training data loader.
    
    Args:
        training_pool: Training-pool dictionary.
        hsi_cube: HSI data with shape (H, W, D).
        lidar_map: LiDAR data with shape (H, W) or (H, W, C).
        batch_size: Batch size.
        patch_size: Spatial patch size.
        balance_classes: Whether to balance classes.
        num_workers: Number of data-loading workers.
    
    Returns:
        dataloader: PyTorch DataLoader
    """
    dataset = EPGDataset(
        training_pool=training_pool,
        hsi_cube=hsi_cube,
        lidar_map=lidar_map,
        patch_size=patch_size,
        balance_classes=balance_classes
    )
    
    if balance_classes:
        # Use the class-balanced batch sampler.
        sampler = BalancedBatchSampler(dataset, batch_size)
        dataloader = Data.DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=num_workers
        )
    else:
        # Use standard random sampling.
        dataloader = Data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
    
    return dataloader

def create_test_dataloader(test_coords, test_labels, hsi_cube, lidar_map,
                          batch_size=64, patch_size=11, num_workers=0):
    """
    Create a test data loader.
    
    Args:
        test_coords: Test coordinates with shape (N, 2).
        test_labels: Test labels with shape (N,).
        hsi_cube: HSI data with shape (H, W, D).
        lidar_map: LiDAR data with shape (H, W) or (H, W, C).
        batch_size: Batch size.
        patch_size: Spatial patch size.
        num_workers: Number of data-loading workers.
    
    Returns:
        dataloader: PyTorch DataLoader
    """
    # Build a temporary training-pool structure.
    test_pool = {
        'coords': test_coords,
        'labels': test_labels,
        'confidences': np.ones(len(test_labels)),
        'status': np.array(['test'] * len(test_labels))
    }
    
    dataset = EPGDataset(
        training_pool=test_pool,
        hsi_cube=hsi_cube,
        lidar_map=lidar_map,
        patch_size=patch_size,
        balance_classes=False
    )
    
    dataloader = Data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return dataloader


