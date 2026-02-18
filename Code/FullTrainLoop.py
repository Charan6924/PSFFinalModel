import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from PSDDataset import PSDDataset
from Dataset import MTFPSDDataset
from SplineEstimator import KernelEstimator
from itertools import cycle
from utils import (
    generate_images,
    get_torch_spline,
    save_checkpoint,
    load_checkpoint,
    compute_gradient_norm,
    plot_training_metrics,
    validate, compute_psd, plot_images_for_epoch, plot_splines_for_epoch, setup_logging, spline_to_kernel)
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
import logging
from datetime import datetime
from itertools import cycle


def train_epoch(model, image_loader, mtf_loader, optimizer, scaler, l1_loss, alpha, device, logger):
    model.train()

    total_loss       = 0.0
    total_ft_loss    = 0.0
    total_recon_loss = 0.0
    total_mtf_loss   = 0.0
    total_grad_norm  = 0.0
    num_batches      = 0
    nan_batches      = 0

    mtf_cycle = cycle(mtf_loader)

    last_I_smooth        = None
    last_I_sharp         = None
    last_I_gen_sharp     = None
    last_I_gen_smooth    = None
    last_knots_smooth    = None
    last_control_smooth  = None
    last_knots_sharp     = None
    last_control_sharp   = None
    last_knots_phantom   = None
    last_control_phantom = None
    last_target_mtfs     = None

    for batch_idx, (I_smooth_1, I_sharp_1, I_smooth_2, I_sharp_2) in enumerate(
        tqdm(image_loader, desc="Training", unit="batch")
    ):
        I_smooth_1 = I_smooth_1.to(device, non_blocking=True)
        I_sharp_1  = I_sharp_1.to(device, non_blocking=True)
        I_smooth_2 = I_smooth_2.to(device, non_blocking=True)
        I_sharp_2  = I_sharp_2.to(device, non_blocking=True)

        with torch.no_grad():
            psd_smooth_1 = compute_psd(I_smooth_1, device='cuda').to(device, non_blocking=True)
            psd_sharp_1  = compute_psd(I_sharp_1,  device='cuda').to(device, non_blocking=True)
            psd_smooth_2 = compute_psd(I_smooth_2, device='cuda').to(device, non_blocking=True)
            psd_sharp_2  = compute_psd(I_sharp_2,  device='cuda').to(device, non_blocking=True)

            I_smooth_ft = torch.fft.fftshift(torch.fft.fft2(I_smooth_1, dim=(-2, -1)), dim=(-2, -1))
            I_sharp_ft  = torch.fft.fftshift(torch.fft.fft2(I_sharp_1,  dim=(-2, -1)), dim=(-2, -1))

            real_smooth_2_sharp = I_sharp_ft  / (I_smooth_ft + 1e-8)
            real_sharp_2_smooth = I_smooth_ft / (I_sharp_ft  + 1e-8)

            # clamp outliers then normalise so both targets live in [-1, 1]
            p99_s2sh = torch.quantile(
                torch.abs(real_smooth_2_sharp).flatten(1), 0.99, dim=1
            ).view(-1, 1, 1, 1)
            p99_sh2s = torch.quantile(
                torch.abs(real_sharp_2_smooth).flatten(1), 0.99, dim=1
            ).view(-1, 1, 1, 1)

            real_smooth_2_sharp = (
                torch.clamp(real_smooth_2_sharp.real, -p99_s2sh, p99_s2sh) +
                1j * torch.clamp(real_smooth_2_sharp.imag, -p99_s2sh, p99_s2sh)
            ) / (p99_s2sh + 1e-8)

            real_sharp_2_smooth = (
                torch.clamp(real_sharp_2_smooth.real, -p99_sh2s, p99_sh2s) +
                1j * torch.clamp(real_sharp_2_smooth.imag, -p99_sh2s, p99_sh2s)
            ) / (p99_sh2s + 1e-8)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device == "cuda")):
            smooth_knots_1, smooth_control_points_1 = model(psd_smooth_1)
            sharp_knots_2,  sharp_control_points_2  = model(psd_sharp_2)

            filter_smooth2sharp, filter_sharp2smooth = spline_to_kernel(
                smooth_knots=smooth_knots_1, smooth_control_points=smooth_control_points_1,
                sharp_knots=sharp_knots_2,   sharp_control_points=sharp_control_points_2,
                grid_size=512
            )

            # normalise predicted filters by same p99 scale as targets
            p99_s2sh_cast = p99_s2sh.to(filter_smooth2sharp.dtype)
            p99_sh2s_cast = p99_sh2s.to(filter_sharp2smooth.dtype)

            filter_smooth2sharp_norm = filter_smooth2sharp / (p99_s2sh_cast + 1e-8)
            filter_sharp2smooth_norm = filter_sharp2smooth / (p99_sh2s_cast + 1e-8)

            # your original formula, now on matched scales
            ft_loss = (
                torch.abs(real_smooth_2_sharp - filter_smooth2sharp_norm) +
                torch.abs(real_sharp_2_smooth - filter_sharp2smooth_norm)
            ).mean()

            I_generated_sharp, I_generated_smooth = generate_images(
                I_smooth=I_smooth_1, I_sharp=I_sharp_2,
                filter_smooth2sharp=filter_smooth2sharp,
                filter_sharp2smooth=filter_sharp2smooth,
                device=device
            )

            recon_loss_smooth = l1_loss(I_generated_smooth, I_smooth_2)
            recon_loss_sharp  = l1_loss(I_generated_sharp,  I_sharp_1)
            recon_loss        = (recon_loss_smooth + recon_loss_sharp) / 2.0

            input_profiles, target_mtfs = next(mtf_cycle)
            input_profiles = input_profiles.to(device, non_blocking=True)
            target_mtfs    = target_mtfs.to(device, non_blocking=True)

            knots_phantom, control_phantom = model(input_profiles)
            mtf_phantom = get_torch_spline(knots_phantom, control_phantom, num_points=64).squeeze(1)
            mtf_loss    = l1_loss(mtf_phantom, target_mtfs)

            batch_loss = alpha * recon_loss + (1 - alpha) * mtf_loss + 0.1 * ft_loss

        if batch_idx == 0:
            print(f"\n[DEBUG] Loss Breakdown (batch 0):")
            print(f"  recon_loss         : {recon_loss.item():.6f}  (weighted: {alpha * recon_loss.item():.6f})")
            print(f"  mtf_loss           : {mtf_loss.item():.6f}  (weighted: {(1 - alpha) * mtf_loss.item():.6f})")
            print(f"  ft_loss (raw)      : {ft_loss.item():.6f}  (weighted: {0.1 * ft_loss.item():.6f})")
            print(f"  batch_loss (total) : {batch_loss.item():.6f}")
            print(f"  filter_s2sh range  : [{filter_smooth2sharp_norm.real.min().item():.4f}, {filter_smooth2sharp_norm.real.max().item():.4f}]")
            print(f"  target_s2sh range  : [{torch.abs(real_smooth_2_sharp).min().item():.4f}, {torch.abs(real_smooth_2_sharp).max().item():.4f}]")

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

        total_loss       += batch_loss.item()
        total_recon_loss += recon_loss.item()
        total_mtf_loss   += mtf_loss.item()
        total_ft_loss    += ft_loss.item()
        total_grad_norm  += grad_norm
        num_batches      += 1

        last_I_smooth        = I_smooth_1.detach()
        last_I_sharp         = I_sharp_2.detach()
        last_I_gen_sharp     = I_generated_sharp.detach()
        last_I_gen_smooth    = I_generated_smooth.detach()
        last_knots_smooth    = smooth_knots_1.detach()
        last_control_smooth  = smooth_control_points_1.detach()
        last_knots_sharp     = sharp_knots_2.detach()
        last_control_sharp   = sharp_control_points_2.detach()
        last_knots_phantom   = knots_phantom.detach()
        last_control_phantom = control_phantom.detach()
        last_target_mtfs     = target_mtfs.detach()

    if nan_batches > 0:
        logger.warning(f"Epoch summary: {nan_batches} batches skipped due to NaN/Inf")

    return {
        'total_loss':  total_loss       / max(num_batches, 1),
        'recon_loss':  total_recon_loss  / max(num_batches, 1),
        'mtf_loss':    total_mtf_loss    / max(num_batches, 1),
        'ft_loss':     total_ft_loss     / max(num_batches, 1),
        'grad_norm':   total_grad_norm   / max(num_batches, 1),
        'nan_batches': nan_batches
    }, {
        'I_smooth':        last_I_smooth,
        'I_sharp':         last_I_sharp,
        'I_gen_sharp':     last_I_gen_sharp,
        'I_gen_smooth':    last_I_gen_smooth,
        'knots_smooth':    last_knots_smooth,
        'control_smooth':  last_control_smooth,
        'knots_sharp':     last_knots_sharp,
        'control_sharp':   last_control_sharp,
        'knots_phantom':   last_knots_phantom,
        'control_phantom': last_control_phantom,
        'target_mtfs':     last_target_mtfs
    }


def main():
    IMAGE_ROOT_DIR   = r"D:\Charan work file\KernelEstimator\Data_Root"
    MTF_DATASET_PATH = r"D:\Charan work file\PhantomTesting\training_dataset.npz"
    ALPHA            = 0.5
    OUTPUT_DIR       = Path(f"training_output_{ALPHA}")
    CHECKPOINT_DIR   = OUTPUT_DIR / "checkpoints"
    VIS_IMAGES_DIR   = OUTPUT_DIR / "visualization" / "images"
    VIS_SPLINES_DIR  = OUTPUT_DIR / "visualization" / "splines"
    BATCH_SIZE       = 16
    NUM_WORKERS      = 0
    LEARNING_RATE    = 1e-4
    NUM_EPOCHS       = 150
    RESUME           = False

    SCHEDULER_FACTOR   = 0.5
    SCHEDULER_PATIENCE = 5
    SCHEDULER_MIN_LR   = 1e-7

    OUTPUT_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    VIS_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    VIS_SPLINES_DIR.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(OUTPUT_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("="*70)
    logger.info("Training Configuration [FIXED FT LOSS VERSION]")
    logger.info(f"Device: {device}")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    logger.info(f"Num Workers: {NUM_WORKERS}")
    logger.info(f"Learning Rate: {LEARNING_RATE}")
    logger.info(f"LR Scheduler: ReduceLROnPlateau")
    logger.info(f"  - Factor: {SCHEDULER_FACTOR}")
    logger.info(f"  - Patience: {SCHEDULER_PATIENCE}")
    logger.info(f"  - Min LR: {SCHEDULER_MIN_LR}")
    logger.info(f"Alpha: {ALPHA}")
    logger.info(f"FT Loss Weight: 0.1 (normalised targets)")
    logger.info(f"Epochs: {NUM_EPOCHS}")
    logger.info(f"Resume: {RESUME}")
    logger.info("="*70)

    print("\nLoading datasets...")
    logger.info("\nLoading datasets...")

    data_root     = r"D:\Charan work file\KernelEstimator\Data_Root"
    image_dataset = PSDDataset(root_dir=data_root, preload=True)

    image_train_size = int(0.9 * len(image_dataset))
    image_val_size   = len(image_dataset) - image_train_size

    image_train_dataset, image_val_dataset = random_split(
        image_dataset,
        [image_train_size, image_val_size],
        generator=torch.Generator().manual_seed(42)
    )

    mtf_folder  = r"D:\Charan work file\PhantomTesting\MTF_Results_Output"
    psd_folder  = r"D:\Charan work file\PhantomTesting\PSD_Results_Output"
    mtf_dataset = MTFPSDDataset(mtf_folder, psd_folder, verbose=False)

    mtf_train_size = int(0.8 * len(mtf_dataset))
    mtf_val_size   = len(mtf_dataset) - mtf_train_size

    mtf_train_dataset, mtf_val_dataset = random_split(
        mtf_dataset,
        [mtf_train_size, mtf_val_size],
        generator=torch.Generator().manual_seed(42)
    )

    image_train_loader = DataLoader(
        image_train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True
    )
    image_val_loader = DataLoader(
        image_val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True
    )
    mtf_train_loader = DataLoader(
        mtf_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    mtf_val_loader = DataLoader(
        mtf_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    logger.info(f"Image Dataset - Train: {len(image_train_dataset)}, Val: {len(image_val_dataset)}")
    logger.info(f"MTF Dataset   - Train: {len(mtf_train_dataset)}, Val: {len(mtf_val_dataset)}")

    model     = KernelEstimator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=SCHEDULER_MIN_LR
    )

    scaler  = torch.amp.GradScaler('cuda') if device == "cuda" else None
    l1_loss = nn.L1Loss()

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\nModel Parameters:")
    logger.info(f"  Total: {total_params:,}")
    logger.info(f"  Trainable: {trainable_params:,}")

    metrics = {
        'epoch':            [],
        'train_total_loss': [],
        'train_recon_loss': [],
        'train_mtf_loss':   [],
        'train_ft_loss':    [],
        'train_grad_norm':  [],
        'val_total_loss':   [],
        'val_recon_loss':   [],
        'val_mtf_loss':     [],
        'learning_rate':    [],
        'nan_batches':      []
    }

    epoch_metrics = []
    start_epoch   = 0
    best_val_loss = float('inf')

    if RESUME:
        checkpoint_path = CHECKPOINT_DIR / "latest_checkpoint.pth"
        loaded = load_checkpoint(checkpoint_path, model, optimizer, scaler)
        if loaded:
            start_epoch   = loaded['epoch'] + 1
            metrics       = loaded['metrics']
            best_val_loss = loaded['best_val_loss']
            if 'scheduler_state_dict' in loaded:
                scheduler.load_state_dict(loaded['scheduler_state_dict'])
            print(f"Resumed from epoch {loaded['epoch']}")
            print(f"Best validation loss: {best_val_loss:.6f}")
            print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.2e}")

    logger.info("\n" + "="*70)
    logger.info(f"Starting training from epoch {start_epoch + 1}")
    logger.info("="*70 + "\n")

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_num  = epoch + 1
        current_lr = optimizer.param_groups[0]['lr']

        logger.info(f"\n{'='*70}")
        logger.info(f"EPOCH {epoch_num}/{NUM_EPOCHS}")
        logger.info(f"Learning Rate: {current_lr:.2e}")
        logger.info(f"{'='*70}")

        logger.info("\nTraining Phase:")
        train_metrics, vis_data = train_epoch(
            model, image_train_loader, mtf_train_loader,
            optimizer, scaler, l1_loss, ALPHA, device, logger
        )

        logger.info("\nValidation Phase:")
        val_metrics = validate(
            model, image_val_loader, mtf_val_loader,
            l1_loss, ALPHA, device
        )

        scheduler.step(val_metrics['total_loss'])

        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < current_lr:
            print(f"*** Learning rate reduced: {current_lr:.2e} -> {new_lr:.2e} ***")
            logger.info(f"*** Learning rate reduced: {current_lr:.2e} -> {new_lr:.2e} ***")

        metrics['epoch'].append(epoch_num)
        metrics['train_total_loss'].append(train_metrics['total_loss'])
        metrics['train_recon_loss'].append(train_metrics['recon_loss'])
        metrics['train_mtf_loss'].append(train_metrics['mtf_loss'])
        metrics['train_ft_loss'].append(train_metrics['ft_loss'])
        metrics['train_grad_norm'].append(train_metrics['grad_norm'])
        metrics['val_total_loss'].append(val_metrics['total_loss'])
        metrics['val_recon_loss'].append(val_metrics['recon_loss'])
        metrics['val_mtf_loss'].append(val_metrics['mtf_loss'])
        metrics['learning_rate'].append(new_lr)
        metrics['nan_batches'].append(train_metrics.get('nan_batches', 0))

        epoch_data = {
            'epoch':            epoch_num,
            'train_total_loss': train_metrics['total_loss'],
            'train_recon_loss': train_metrics['recon_loss'],
            'train_mtf_loss':   train_metrics['mtf_loss'],
            'train_ft_loss':    train_metrics['ft_loss'],
            'train_grad_norm':  train_metrics['grad_norm'],
            'val_total_loss':   val_metrics['total_loss'],
            'val_recon_loss':   val_metrics['recon_loss'],
            'val_mtf_loss':     val_metrics['mtf_loss'],
            'learning_rate':    new_lr,
            'nan_batches':      train_metrics.get('nan_batches', 0)
        }
        epoch_metrics.append(epoch_data)

        metrics_path = OUTPUT_DIR / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(epoch_metrics, f, indent=2)

        logger.info(f"\nEpoch {epoch_num} Summary:")
        logger.info(f"  [TRAIN] Total Loss: {train_metrics['total_loss']:.6f}")
        logger.info(f"  [TRAIN] Recon Loss: {train_metrics['recon_loss']:.6f}")
        logger.info(f"  [TRAIN] MTF Loss:   {train_metrics['mtf_loss']:.6f}")
        logger.info(f"  [TRAIN] FT Loss:    {train_metrics['ft_loss']:.6f}")
        logger.info(f"  [TRAIN] Grad Norm:  {train_metrics['grad_norm']:.4f}")
        logger.info(f"  [TRAIN] NaN Batches: {train_metrics.get('nan_batches', 0)}")
        logger.info(f"  [VAL]   Total Loss: {val_metrics['total_loss']:.6f}")
        logger.info(f"  [VAL]   Recon Loss: {val_metrics['recon_loss']:.6f}")
        logger.info(f"  [VAL]   MTF Loss:   {val_metrics['mtf_loss']:.6f}")

        logger.info("\nGenerating visualizations...")
        try:
            img_path = plot_images_for_epoch(
                vis_data['I_smooth'], vis_data['I_sharp'],
                vis_data['I_gen_sharp'], vis_data['I_gen_smooth'],
                epoch_num, VIS_IMAGES_DIR
            )
            logger.info(f"  Images saved: {img_path}")

            spline_path = plot_splines_for_epoch(
                vis_data['knots_smooth'], vis_data['control_smooth'],
                vis_data['knots_sharp'],  vis_data['control_sharp'],
                vis_data['knots_phantom'], vis_data['control_phantom'],
                vis_data['target_mtfs'],
                epoch_num, VIS_SPLINES_DIR
            )
            logger.info(f"  Splines saved: {spline_path}")

        except Exception as e:
            logger.error(f"  Error during visualization: {str(e)}")
            print(f"  Warning: Visualization failed - {str(e)}")

        is_best = val_metrics['total_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['total_loss']
            print(f"New best model! Val Loss: {best_val_loss:.6f}")
            logger.info(f"NEW BEST MODEL - Val Loss: {best_val_loss:.6f}")

        checkpoint = {
            'epoch':                epoch_num,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict':    scaler.state_dict() if scaler else None,
            'metrics':              metrics,
            'best_val_loss':        best_val_loss,
            'alpha':                ALPHA,
            'learning_rate':        LEARNING_RATE
        }

        torch.save(checkpoint, CHECKPOINT_DIR / f"epoch_{epoch + 1}_checkpoint.pth")
        if is_best:
            torch.save(checkpoint, CHECKPOINT_DIR / "best_checkpoint.pth")

        logger.info(f"Checkpoint saved (is_best={is_best})")
        print("-"*70 + "\n")

    best_epoch = metrics['epoch'][metrics['val_total_loss'].index(min(metrics['val_total_loss']))]
    final_lr   = optimizer.param_groups[0]['lr']

    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Best Validation Loss: {best_val_loss:.6f}")
    logger.info(f"Best Epoch: {best_epoch}")
    logger.info(f"Final Learning Rate: {final_lr:.2e}")
    logger.info("="*70)

    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    plot_training_metrics(metrics, ALPHA, LEARNING_RATE, OUTPUT_DIR)


if __name__ == "__main__":
    main()