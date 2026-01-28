import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from PSDDataset import PSDDataset
from Dataset import MTFDataset
from SplineEstimator import KernelEstimator
from utils import (
    generate_images, 
    get_torch_spline, 
    radial_mtf_to_2d_otf,
    save_checkpoint,
    load_checkpoint,
    compute_gradient_norm,
    plot_training_metrics, 
    validate,
    get_scipy_spline, plot_images_for_epoch, plot_splines_for_epoch, setup_logging
)
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import numpy as np
import logging
from datetime import datetime






def train_epoch(model, image_loader, mtf_loader, optimizer, scaler, l1_loss, alpha, device, logger):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_mtf_loss = 0.0
    total_grad_norm = 0.0
    num_batches = 0
    
    image_iter = iter(image_loader)
    mtf_iter = iter(mtf_loader)
    
    num_iters = max(len(image_loader), len(mtf_loader))
    
    # Variables to store last batch for visualization
    last_I_smooth = None
    last_I_sharp = None
    last_I_gen_sharp = None
    last_I_gen_smooth = None
    last_knots_smooth = None
    last_control_smooth = None
    last_knots_sharp = None
    last_control_sharp = None
    last_knots_phantom = None
    last_control_phantom = None
    last_target_mtfs = None
    
    for batch_idx in tqdm(range(num_iters), desc="Training", unit="batch"):
        # Get image batch
        try:
            I_smooth, I_sharp, psd_smooth, psd_sharp = next(image_iter)
        except StopIteration:
            image_iter = iter(image_loader)
            I_smooth, I_sharp, psd_smooth, psd_sharp = next(image_iter)
        
        I_smooth = I_smooth.to(device, non_blocking=True)
        I_sharp = I_sharp.to(device, non_blocking=True)
        psd_smooth = psd_smooth.to(device, non_blocking=True)
        psd_sharp = psd_sharp.to(device, non_blocking=True)
        
        # Forward pass with autocast
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device == "cuda")): #type: ignore
            # Image reconstruction
            knots_smooth, control_smooth = model(psd_smooth)
            knots_sharp, control_sharp = model(psd_sharp)
            
            mtf_smooth = get_torch_spline(knots_smooth, control_smooth)
            mtf_sharp = get_torch_spline(knots_sharp, control_sharp)
            
            otf_smooth_2d = radial_mtf_to_2d_otf(mtf_smooth, I_smooth.shape[-2:], device)
            otf_sharp_2d = radial_mtf_to_2d_otf(mtf_sharp, I_sharp.shape[-2:], device)
            
            I_gen_sharp, I_gen_smooth = generate_images(I_smooth, I_sharp, otf_smooth_2d, otf_sharp_2d)
            
            recon_loss_smooth = l1_loss(I_gen_smooth, I_smooth)
            recon_loss_sharp = l1_loss(I_gen_sharp, I_sharp)
            recon_loss = (recon_loss_smooth + recon_loss_sharp) / 2.0
            
            # Get MTF batch
            try:
                input_profiles, target_mtfs = next(mtf_iter)
            except StopIteration:
                mtf_iter = iter(mtf_loader)
                input_profiles, target_mtfs = next(mtf_iter)
            
            input_profiles = input_profiles.to(device, non_blocking=True).unsqueeze(1)
            target_mtfs = target_mtfs.to(device, non_blocking=True)
            
            # MTF supervision
            knots_phantom, control_phantom = model(input_profiles)
            mtf_phantom = get_torch_spline(knots_phantom, control_phantom).squeeze(1)
            
            mtf_loss = l1_loss(mtf_phantom, target_mtfs)
            
            # Combined loss
            batch_loss = alpha * recon_loss + (1 - alpha) * mtf_loss
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        
        if scaler:
            scaler.scale(batch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_norm = compute_gradient_norm(model)
            scaler.step(optimizer)
            scaler.update()
        else:
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_norm = compute_gradient_norm(model)
            optimizer.step()
        
        # Accumulate metrics
        total_loss += batch_loss.item()
        total_recon_loss += recon_loss.item()
        total_mtf_loss += mtf_loss.item()
        total_grad_norm += grad_norm
        num_batches += 1
        
        # Log batch details
        logger.info(f"  Batch [{batch_idx + 1}/{num_iters}] - "
                   f"Loss: {batch_loss.item():.6f}, "
                   f"Recon: {recon_loss.item():.6f}, "
                   f"MTF: {mtf_loss.item():.6f}, "
                   f"Grad: {grad_norm:.4f}")
        
        # Store last batch for visualization
        last_I_smooth = I_smooth.detach()
        last_I_sharp = I_sharp.detach()
        last_I_gen_sharp = I_gen_sharp.detach()
        last_I_gen_smooth = I_gen_smooth.detach()
        last_knots_smooth = knots_smooth.detach()
        last_control_smooth = control_smooth.detach()
        last_knots_sharp = knots_sharp.detach()
        last_control_sharp = control_sharp.detach()
        last_knots_phantom = knots_phantom.detach()
        last_control_phantom = control_phantom.detach()
        last_target_mtfs = target_mtfs.detach()
    
    # Return metrics and last batch data for visualization
    return {
        'total_loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'mtf_loss': total_mtf_loss / num_batches,
        'grad_norm': total_grad_norm / num_batches
    }, {
        'I_smooth': last_I_smooth,
        'I_sharp': last_I_sharp,
        'I_gen_sharp': last_I_gen_sharp,
        'I_gen_smooth': last_I_gen_smooth,
        'knots_smooth': last_knots_smooth,
        'control_smooth': last_control_smooth,
        'knots_sharp': last_knots_sharp,
        'control_sharp': last_control_sharp,
        'knots_phantom': last_knots_phantom,
        'control_phantom': last_control_phantom,
        'target_mtfs': last_target_mtfs
    }


def main():
    # ========================================
    # CONFIGURATION
    # ========================================
    IMAGE_ROOT_DIR = r"D:\Charan work file\KernelEstimator\Data_Root"
    MTF_DATASET_PATH = r"D:\Charan work file\PhantomTesting\training_dataset.npz"
    OUTPUT_DIR = Path("training_output")
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    VIS_IMAGES_DIR = OUTPUT_DIR / "visualization" / "images"
    VIS_SPLINES_DIR = OUTPUT_DIR / "visualization" / "splines"
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    LEARNING_RATE = 1e-4
    ALPHA = 0.5
    NUM_EPOCHS = 100
    RESUME = False
    OUTPUT_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    VIS_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    VIS_SPLINES_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(OUTPUT_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Print to terminal (concise)
    print("="*70)
    print(f"Training Configuration")
    print(f"Device: {device}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Alpha: {ALPHA}")
    print(f"Epochs: {NUM_EPOCHS}")
    print("="*70)
    logger.info("="*70)
    logger.info(f"Training Configuration")
    logger.info(f"Device: {device}")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    logger.info(f"Num Workers: {NUM_WORKERS}")
    logger.info(f"Learning Rate: {LEARNING_RATE}")
    logger.info(f"Alpha: {ALPHA}")
    logger.info(f"Epochs: {NUM_EPOCHS}")
    logger.info(f"Resume: {RESUME}")
    logger.info(f"Output Directory: {OUTPUT_DIR}")
    logger.info(f"Checkpoint Directory: {CHECKPOINT_DIR}")
    logger.info(f"Visualization - Images: {VIS_IMAGES_DIR}")
    logger.info(f"Visualization - Splines: {VIS_SPLINES_DIR}")
    logger.info("="*70)
    print("\nLoading datasets...")
    logger.info("\nLoading datasets...")
    
    # Image dataset
    image_dataset = PSDDataset(
        root_dir=r"D:\Charan work file\KernelEstimator\Data_Root",
        sampling_strategy='all',
        use_ct_windowing=True,
    )
    
    image_train_size = int(0.9 * len(image_dataset))
    image_val_size = len(image_dataset) - image_train_size
    
    image_train_dataset, image_val_dataset = random_split(
        image_dataset,
        [image_train_size, image_val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # MTF dataset
    mtf_dataset = MTFDataset(MTF_DATASET_PATH)
    
    mtf_train_size = int(0.8 * len(mtf_dataset))
    mtf_val_size = len(mtf_dataset) - mtf_train_size
    
    mtf_train_dataset, mtf_val_dataset = random_split(
        mtf_dataset,
        [mtf_train_size, mtf_val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    image_train_loader = DataLoader(
        image_train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=True
    )
    image_val_loader = DataLoader(
        image_val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    mtf_train_loader = DataLoader(
        mtf_train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    mtf_val_loader = DataLoader(
        mtf_val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    
    print(f"Image - Train: {len(image_train_dataset)} | Val: {len(image_val_dataset)}")
    print(f"MTF   - Train: {len(mtf_train_dataset)} | Val: {len(mtf_val_dataset)}")
    
    logger.info(f"Image Dataset - Train: {len(image_train_dataset)}, Val: {len(image_val_dataset)}")
    logger.info(f"MTF Dataset   - Train: {len(mtf_train_dataset)}, Val: {len(mtf_val_dataset)}")
    logger.info(f"Image Train Batches: {len(image_train_loader)}")
    logger.info(f"Image Val Batches: {len(image_val_loader)}")
    logger.info(f"MTF Train Batches: {len(mtf_train_loader)}")
    logger.info(f"MTF Val Batches: {len(mtf_val_loader)}")
    
    # ========================================
    # INITIALIZE MODEL
    # ========================================
    model = KernelEstimator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None #type: ignore
    l1_loss = nn.L1Loss()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\nModel Parameters:")
    logger.info(f"  Total: {total_params:,}")
    logger.info(f"  Trainable: {trainable_params:,}")
    
    # Metrics tracking
    metrics = {
        'epoch': [],
        'train_total_loss': [],
        'train_recon_loss': [],
        'train_mtf_loss': [],
        'train_grad_norm': [],
        'val_total_loss': [],
        'val_recon_loss': [],
        'val_mtf_loss': []
    }
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Resume from checkpoint if requested
    if RESUME:
        checkpoint_path = CHECKPOINT_DIR / "latest_checkpoint.pth"
        loaded = load_checkpoint(checkpoint_path, model, optimizer, scaler)
        if loaded:
            start_epoch = loaded['epoch'] + 1
            metrics = loaded['metrics']
            best_val_loss = loaded['best_val_loss']
            print(f"Resumed from epoch {loaded['epoch']}")
            print(f"Best validation loss: {best_val_loss:.6f}")
            logger.info(f"Resumed training from epoch {loaded['epoch']}")
            logger.info(f"Best validation loss so far: {best_val_loss:.6f}")
    
    # ========================================
    # TRAINING LOOP
    # ========================================
    print("\n" + "="*70)
    print(f"Starting training from epoch {start_epoch + 1}")
    print("="*70 + "\n")
    
    logger.info("\n" + "="*70)
    logger.info(f"Starting training from epoch {start_epoch + 1}")
    logger.info("="*70 + "\n")
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_num = epoch + 1
        
        # Terminal output (concise)
        print(f"Epoch [{epoch_num}/{NUM_EPOCHS}]")
        print("-"*70)
        
        # Log file output (detailed)
        logger.info(f"\n{'='*70}")
        logger.info(f"EPOCH {epoch_num}/{NUM_EPOCHS}")
        logger.info(f"{'='*70}")
        
        # Train
        logger.info("\nTraining Phase:")
        train_metrics, vis_data = train_epoch(
            model, image_train_loader, mtf_train_loader,
            optimizer, scaler, l1_loss, ALPHA, device, logger
        )
        
        # Validate
        logger.info("\nValidation Phase:")
        val_metrics = validate(
            model, image_val_loader, mtf_val_loader,
            l1_loss, ALPHA, device
        )
        
        logger.info(f"Validation Complete - Total Loss: {val_metrics['total_loss']:.6f}")
        
        # Update metrics
        metrics['epoch'].append(epoch_num)
        metrics['train_total_loss'].append(train_metrics['total_loss'])
        metrics['train_recon_loss'].append(train_metrics['recon_loss'])
        metrics['train_mtf_loss'].append(train_metrics['mtf_loss'])
        metrics['train_grad_norm'].append(train_metrics['grad_norm'])
        metrics['val_total_loss'].append(val_metrics['total_loss'])
        metrics['val_recon_loss'].append(val_metrics['recon_loss'])
        metrics['val_mtf_loss'].append(val_metrics['mtf_loss'])
        
        # Print summary to terminal (concise)
        print(f"[TRAIN] Total: {train_metrics['total_loss']:.6f} | "
              f"Recon: {train_metrics['recon_loss']:.6f} | "
              f"MTF: {train_metrics['mtf_loss']:.6f} | "
              f"Grad: {train_metrics['grad_norm']:.4f}")
        print(f"[VAL]   Total: {val_metrics['total_loss']:.6f} | "
              f"Recon: {val_metrics['recon_loss']:.6f} | "
              f"MTF: {val_metrics['mtf_loss']:.6f}")
        
        # Log summary to file (detailed)
        logger.info(f"\nEpoch {epoch_num} Summary:")
        logger.info(f"  [TRAIN] Total Loss: {train_metrics['total_loss']:.6f}")
        logger.info(f"  [TRAIN] Recon Loss: {train_metrics['recon_loss']:.6f}")
        logger.info(f"  [TRAIN] MTF Loss: {train_metrics['mtf_loss']:.6f}")
        logger.info(f"  [TRAIN] Grad Norm: {train_metrics['grad_norm']:.4f}")
        logger.info(f"  [VAL]   Total Loss: {val_metrics['total_loss']:.6f}")
        logger.info(f"  [VAL]   Recon Loss: {val_metrics['recon_loss']:.6f}")
        logger.info(f"  [VAL]   MTF Loss: {val_metrics['mtf_loss']:.6f}")
        
        # Visualizations
        logger.info("\nGenerating visualizations...")
        try:
            # Plot images
            img_path = plot_images_for_epoch(
                vis_data['I_smooth'],
                vis_data['I_sharp'],
                vis_data['I_gen_sharp'],
                vis_data['I_gen_smooth'],
                epoch_num,
                VIS_IMAGES_DIR
            )
            logger.info(f"  Images saved: {img_path}")
            
            # Plot splines
            spline_path = plot_splines_for_epoch(
                vis_data['knots_smooth'],
                vis_data['control_smooth'],
                vis_data['knots_sharp'],
                vis_data['control_sharp'],
                vis_data['knots_phantom'],
                vis_data['control_phantom'],
                vis_data['target_mtfs'],
                epoch_num,
                VIS_SPLINES_DIR
            )
            logger.info(f"  Splines saved: {spline_path}")
            
        except Exception as e:
            logger.error(f"  Error during visualization: {str(e)}")
            print(f"  Warning: Visualization failed - {str(e)}")
        
        # Save checkpoint
        is_best = val_metrics['total_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['total_loss']
            print(f"New best model! Val Loss: {best_val_loss:.6f}")
            logger.info(f"NEW BEST MODEL - Val Loss: {best_val_loss:.6f}")
        
        save_checkpoint(
            epoch_num, model, optimizer, scaler, metrics, 
            best_val_loss, ALPHA, LEARNING_RATE, CHECKPOINT_DIR, is_best
        )
        logger.info(f"Checkpoint saved (is_best={is_best})")
        
        print("-"*70 + "\n")
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    best_epoch = metrics['epoch'][metrics['val_total_loss'].index(min(metrics['val_total_loss']))]
    
    print("="*70)
    print("TRAINING COMPLETE!")
    print(f"Best Validation Loss: {best_val_loss:.6f}")
    print(f"Best Epoch: {best_epoch}")
    print("="*70)
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Best Validation Loss: {best_val_loss:.6f}")
    logger.info(f"Best Epoch: {best_epoch}")
    logger.info(f"Total Epochs Trained: {NUM_EPOCHS}")
    logger.info("="*70)
    
    # Save metrics
    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Plot metrics
    plot_training_metrics(metrics, ALPHA, LEARNING_RATE, OUTPUT_DIR)
    print(f"Plots saved to {OUTPUT_DIR / 'training_metrics.png'}")
    logger.info(f"Training metrics plot saved to {OUTPUT_DIR / 'training_metrics.png'}")


if __name__ == "__main__":
    main()