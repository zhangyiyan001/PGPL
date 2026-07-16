# -*- coding:utf-8 -*-
"""
Three-stage EPG and CPST training pipeline:
- Stage 1: Enhanced Prototype Guidance warm-up
- Stage 2: Collaborative Progressive Self-Training
- Stage 3: Refinement
"""
import copy
import json
import os
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from S2Enet import S2ENet
from module import weight_init
from dataset_epg import create_epg_dataloader
from epg_module import CPSTPool, ActiveLearningPool, cou_step, fiu_step, get_confidence_threshold
from test import evaluate_model_metrics

def create_model(input_channels_hsi, input_channels_lidar, n_classes, patch_size, device, dropout_rate=0.0):

    model = S2ENet(
        input_channels=input_channels_hsi,
        input_channels2=input_channels_lidar,
        n_classes=n_classes,
        patch_size=patch_size,
        dropout_rate=dropout_rate
    ).to(device)
    
    model.apply(weight_init)
    model.patch_size = patch_size
    model.dropout_rate = dropout_rate
    return model


def train_epg_warmup(training_pool, hsi_cube, lidar_map, device,
                     input_channels_hsi=144, input_channels_lidar=1,
                     n_classes=15, patch_size=11, dropout_rate=0.0,
                     epochs=40, batch_size=64, lr=0.001, weight_decay=0.0,
                     save_path='./models/Houston_epg_warmup.pt',
                     balance_classes=True, num_workers=0, save_intermediate=True):
    """
    Stage 1: train on high-quality samples selected by EPG.

    Args:
        training_pool: Training pool containing gold and EPG samples.
        hsi_cube: HSI data with shape (H, W, D).
        lidar_map: LiDAR data with shape (H, W) or (H, W, C).
        device: Training device.
        Other parameters configure the model and optimizer.

    Returns:
        model: Trained model.
    """
    print("\n" + "="*60)
    print("Stage 1: EPG Warm-up")
    print("="*60)
    print(f"Training samples: {len(training_pool['coords'])}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    
    # Create the model.
    model = create_model(input_channels_hsi, input_channels_lidar, n_classes, patch_size, device, dropout_rate=dropout_rate)
    
    # Create the optimizer and loss function.
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Configure the learning-rate scheduler.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Create the data loader.
    train_loader = create_epg_dataloader(
        training_pool=training_pool,
        hsi_cube=hsi_cube,
        lidar_map=lidar_map,
        batch_size=batch_size,
        patch_size=patch_size,
        balance_classes=balance_classes,
        num_workers=num_workers
    )
    
    best_loss = float('inf')
    best_state_dict = None
    
    # Training loop.
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        start_time = time.time()
        
        for batch_idx, (hsi_batch, lidar_batch, labels) in enumerate(train_loader):
            hsi_batch = hsi_batch.to(device)
            lidar_batch = lidar_batch.to(device)
            labels = labels.to(device)
            
            # Forward pass.
            outputs = model(hsi_batch, lidar_batch)
            loss = criterion(outputs, labels)
            
            # Backward pass.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics.
            epoch_loss += loss.item() * len(labels)
            _, predicted = torch.max(outputs, 1)
            epoch_correct += (predicted == labels).sum().item()
            epoch_total += len(labels)
        
        scheduler.step()
        
        # Compute epoch metrics.
        avg_loss = epoch_loss / epoch_total
        avg_acc = epoch_correct / epoch_total
        epoch_time = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {avg_loss:.6f} | Acc: {avg_acc:.4f} | "
              f"Time: {epoch_time:.2f}s | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save the best model.
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            if save_intermediate:
                torch.save(best_state_dict, save_path)
                print(f"  *** Saved best model (loss={best_loss:.6f}) ***")

    if not save_intermediate and best_state_dict is not None:
        torch.save(best_state_dict, save_path)
        print(f"  *** Saved best model (loss={best_loss:.6f}) ***")
    
    print(f"\nEPG warm-up complete! Best loss: {best_loss:.6f}")
    print(f"Model saved to: {save_path}\n")
    
    return model

def train_cpst(model, training_pool, preparation_mask,
               hsi_cube, lidar_map, gt_map, device,
               input_channels_hsi=144, input_channels_lidar=1,
               n_classes=15, patch_size=11,
               epochs=140, batch_size=64, lr=0.0005, weight_decay=0.0,
               cou_interval=5, fiu_interval=5,
               cou_sample_size=10000, cou_quota_per_class=200,
               fiu_sample_size=5000, fiu_alpha=0.3,
               save_path='./models/Houston_epg_active.pt',
               history_path='results/training_pool_history.json',
               balance_classes=True, num_workers=0,
               log_interval=10, save_intermediate=True,
               test_loader=None, dataset_name=None, eval_interval=None):
    """
    Stage 2: Collaborative Progressive Self-Training.
    Stage2: Collaborative Progressive Self-Training

    Combine COU sample expansion with FIU pseudo-label updates.

    Args:
        model: Warmed-up model.
        training_pool: Training-pool dictionary.
        preparation_mask: Preparation-pool mask.
        hsi_cube, lidar_map, gt_map: Input data.
        device: Training device.
        epochs: Number of training epochs.
        cou_interval: Run COU every N epochs.
        fiu_interval: Run FIU every N epochs.
        history_path: Output path for training-pool history.
        Other parameters configure training.

    Returns:
        model: Trained model.
    """
    print("\n" + "="*60)
    print("Stage 2: Collaborative Progressive Self-Training")
    print("="*60)
    print(f"Initial training samples: {len(training_pool['coords'])}")
    print(f"Initial preparation samples: {np.sum(preparation_mask)}")
    print(f"Epochs: {epochs}")
    print(f"COU interval: {cou_interval} epochs")
    print(f"FIU interval: {fiu_interval} epochs")
    
    enable_eval = (
        test_loader is not None
        and dataset_name is not None
        and eval_interval is not None
        and eval_interval > 0
    )
    
    # Create the CPST pool.
    pool = CPSTPool(training_pool, preparation_mask, hsi_cube, lidar_map, gt_map)
    
    
    # Create the optimizer and loss function.
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Configure the learning-rate scheduler.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_loss = float('inf')
    best_state_dict = None
    pool_history = []
    
    # Training loop.
    for epoch in range(epochs):
        # Run COU to select samples from the preparation pool.
        if epoch > 0 and epoch % cou_interval == 0:
            print(f"\n--- Running COU (Epoch {epoch}) ---")
            conf_threshold = get_confidence_threshold(epoch, epochs)
            added_count = cou_step(
                model=model,
                pool=pool,
                device=device,
                conf_threshold=conf_threshold,
                quota_per_class=cou_quota_per_class,
                patch_size=patch_size,
                sample_size=cou_sample_size
            )
            print(f"COU: added {added_count} samples to the training pool")
            print(f"  Training pool: {pool.get_training_size()} | Preparation pool: {pool.get_preparation_size()}")
        
        # Run FIU to update the training pool.
        if epoch > 0 and epoch % fiu_interval == 0:
            print(f"\n--- Running FIU (Epoch {epoch}) ---")
            updated_count = fiu_step(
                model=model,
                pool=pool,
                device=device,
                alpha=fiu_alpha,
                sample_size=fiu_sample_size,
                patch_size=patch_size
            )
            print(f"FIU: updated pseudo-labels for {updated_count} samples")
        
        # Record training-pool history.
        if log_interval and epoch % log_interval == 0:
            pool_history.append({
                'epoch': epoch,
                'training_size': pool.get_training_size(),
                'preparation_size': pool.get_preparation_size()
            })
        
        # Recreate the data loader to reflect pool changes.
        train_loader = create_epg_dataloader(
            training_pool=pool.training_pool,
            hsi_cube=hsi_cube,
            lidar_map=lidar_map,
            batch_size=batch_size,
            patch_size=patch_size,
            balance_classes=balance_classes,
            num_workers=num_workers
        )
        
        # Train for one epoch.
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        start_time = time.time()
        
        for batch_idx, (hsi_batch, lidar_batch, labels) in enumerate(train_loader):
            hsi_batch = hsi_batch.to(device)
            lidar_batch = lidar_batch.to(device)
            labels = labels.to(device)
            
            # Forward pass.
            outputs = model(hsi_batch, lidar_batch)
            loss = criterion(outputs, labels)
            
            # Backward pass.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics.
            epoch_loss += loss.item() * len(labels)
            _, predicted = torch.max(outputs, 1)
            epoch_correct += (predicted == labels).sum().item()
            epoch_total += len(labels)
        
        scheduler.step()
        
        # Compute epoch metrics.
        avg_loss = epoch_loss / epoch_total
        avg_acc = epoch_correct / epoch_total
        epoch_time = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {avg_loss:.6f} | Acc: {avg_acc:.4f} | "
              f"Pool: {pool.get_training_size()} | "
              f"Time: {epoch_time:.2f}s | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if enable_eval and ((epoch + 1) % eval_interval == 0):
            eval_metrics = evaluate_model_metrics(
                model=model,
                test_iter=test_loader,
                dataset=dataset_name,
                device=device
            )
            print(f"[Eval] Epoch {epoch+1} | OA: {eval_metrics['oa']:.4f} | "
                  f"AA: {eval_metrics['aa']:.4f} | Kappa: {eval_metrics['kappa']:.4f}")
        
        # Save the best model.
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            if save_intermediate:
                torch.save(best_state_dict, save_path)
                print(f"  *** Saved best model (loss={best_loss:.6f}) ***")
    
    # Record the final pool state.
    pool_history.append({
        'epoch': epochs,
        'training_size': pool.get_training_size(),
        'preparation_size': pool.get_preparation_size()
    })

    # Save training-pool history.
    if history_path:
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        with open(history_path, 'w') as f:
            json.dump(pool_history, f, indent=2)

    if not save_intermediate:
        if best_state_dict is not None:
            torch.save(best_state_dict, save_path)
        if best_state_dict is not None:
            print(f"  *** Saved best model (loss={best_loss:.6f}) ***")
    
    print(f"\nCPST complete! Best loss: {best_loss:.6f}")
    print(f"Final training-pool size: {pool.get_training_size()}")
    print(f"Model saved to: {save_path}")
    
    return model, pool


def train_refinement(model, training_pool, hsi_cube, lidar_map, device,
                    input_channels_hsi=144, input_channels_lidar=1,
                    n_classes=15, patch_size=11,
                    epochs=20, batch_size=64, lr=0.0001, weight_decay=0.0,
                    save_path='./models/Houston_epg_final.pt',
                    balance_classes=True, num_workers=0, save_intermediate=True):
    """
    Stage 3: fine-tune the model on a fixed training pool.

    Args:
        model: Model produced by CPST.
        training_pool: Final training pool.
        hsi_cube, lidar_map: Input data.
        device: Training device.
        epochs: Number of refinement epochs.
        Other parameters configure training.

    Returns:
        model: Refined model.
    """
    print("\n" + "="*60)
    print("Stage 3: Refinement")
    print("="*60)
    print(f"Training samples: {len(training_pool['coords'])}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    
    # Create the optimizer and loss function.
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Configure the learning-rate scheduler.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Create the data loader.
    train_loader = create_epg_dataloader(
        training_pool=training_pool,
        hsi_cube=hsi_cube,
        lidar_map=lidar_map,
        batch_size=batch_size,
        patch_size=patch_size,
        balance_classes=balance_classes,
        num_workers=num_workers
    )
    
    best_loss = float('inf')
    best_state_dict = None
    
    # Training loop.
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        start_time = time.time()
        
        for batch_idx, (hsi_batch, lidar_batch, labels) in enumerate(train_loader):
            hsi_batch = hsi_batch.to(device)
            lidar_batch = lidar_batch.to(device)
            labels = labels.to(device)
            
            # Forward pass.
            outputs = model(hsi_batch, lidar_batch)
            loss = criterion(outputs, labels)
            
            # Backward pass.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics.
            epoch_loss += loss.item() * len(labels)
            _, predicted = torch.max(outputs, 1)
            epoch_correct += (predicted == labels).sum().item()
            epoch_total += len(labels)
        
        scheduler.step()
        
        # Compute epoch metrics.
        avg_loss = epoch_loss / epoch_total
        avg_acc = epoch_correct / epoch_total
        epoch_time = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {avg_loss:.6f} | Acc: {avg_acc:.4f} | "
              f"Time: {epoch_time:.2f}s")
        
        # Save the best model.
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            if save_intermediate:
                torch.save(best_state_dict, save_path)
                print(f"  *** Saved best model (loss={best_loss:.6f}) ***")
    
    print(f"\nRefinement complete! Best loss: {best_loss:.6f}")
    if not save_intermediate and best_state_dict is not None:
        torch.save(best_state_dict, save_path)
        print(f"  *** Saved best model (loss={best_loss:.6f}) ***")
    print(f"Model saved to: {save_path}\n")
    
    return model






