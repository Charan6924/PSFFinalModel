import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import BSpline
import os
import matplotlib.pyplot as plt
from pathlib import Path
import logging, datetime

def plot_images_for_epoch(I_smooth, I_sharp, I_gen_sharp, I_gen_smooth, epoch, output_dir):
    """
    Plot and save image comparisons for a single epoch
    
    Args:
        I_smooth: [B, 1, H, W] smooth images
        I_sharp: [B, 1, H, W] sharp images
        I_gen_sharp: [B, 1, H, W] generated sharp images
        I_gen_smooth: [B, 1, H, W] generated smooth images
        epoch: current epoch number
        output_dir: directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Take first image from batch and convert to numpy
    I_smooth_np = I_smooth[0, 0].cpu().numpy()
    I_sharp_np = I_sharp[0, 0].cpu().numpy()
    I_gen_sharp_np = I_gen_sharp[0, 0].cpu().numpy()
    I_gen_smooth_np = I_gen_smooth[0, 0].cpu().numpy()
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Plot I_smooth
    im0 = axes[0, 0].imshow(I_smooth_np, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('I_smooth (Input)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    # Plot I_sharp
    im1 = axes[0, 1].imshow(I_sharp_np, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('I_sharp (Input)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # Plot I_generated_sharp
    im2 = axes[1, 0].imshow(I_gen_sharp_np, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('I_generated_sharp (from I_smooth)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)
    
    # Plot I_generated_smooth
    im3 = axes[1, 1].imshow(I_gen_smooth_np, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('I_generated_smooth (from I_sharp)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)
    
    plt.suptitle(f'Image Reconstruction - Epoch {epoch}', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save figure
    save_path = output_dir / f'epoch_{epoch:03d}_images.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path

def setup_logging(output_dir):
    """Setup logging configuration"""
    log_file = output_dir / f"training_{datetime.time().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    
    # File handler - detailed logs
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    
    return logger




def plot_splines_for_epoch(knots_smooth, control_smooth, knots_sharp, control_sharp,
                           knots_phantom, control_phantom, target_mtf, epoch, output_dir):
    """
    Plot and save spline comparisons for a single epoch
    
    Args:
        knots_smooth, control_smooth: smooth image MTF spline parameters
        knots_sharp, control_sharp: sharp image MTF spline parameters
        knots_phantom, control_phantom: phantom MTF spline parameters (predicted)
        target_mtf: ground truth MTF from phantom dataset
        epoch: current epoch number
        output_dir: directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get torch splines (first sample from batch)
    mtf_smooth_torch = get_torch_spline(knots_smooth, control_smooth)[0, 0].cpu().numpy()
    mtf_sharp_torch = get_torch_spline(knots_sharp, control_sharp)[0, 0].cpu().numpy()
    mtf_phantom_torch = get_torch_spline(knots_phantom, control_phantom)[0, 0].cpu().numpy()
    
    # Get scipy splines for comparison
    x_smooth_scipy, y_smooth_scipy = get_scipy_spline(knots_smooth[0], control_smooth[0])
    x_sharp_scipy, y_sharp_scipy = get_scipy_spline(knots_sharp[0], control_sharp[0])
    x_phantom_scipy, y_phantom_scipy = get_scipy_spline(knots_phantom[0], control_phantom[0])
    
    # Get ground truth
    target_mtf_np = target_mtf[0].cpu().numpy()
    
    # Create x-axis for torch splines
    x_torch = np.linspace(0, 1, len(mtf_smooth_torch))
    x_target = np.linspace(0, 1, len(target_mtf_np))
    
    # Create figure with 2 rows, 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ============ Plot 1: Smooth MTF (Torch vs SciPy) ============
    axes[0, 0].plot(x_torch, mtf_smooth_torch, 'b-', linewidth=2.5, label='Torch Spline', alpha=0.8)
    axes[0, 0].plot(x_smooth_scipy, y_smooth_scipy, 'r--', linewidth=2, label='SciPy Spline', alpha=0.7)
    axes[0, 0].set_xlabel('Normalized Frequency', fontsize=12)
    axes[0, 0].set_ylabel('MTF Value', fontsize=12)
    axes[0, 0].set_title('Smooth Image MTF', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim([0, 1])
    
    # ============ Plot 2: Sharp MTF (Torch vs SciPy) ============
    axes[0, 1].plot(x_torch, mtf_sharp_torch, 'b-', linewidth=2.5, label='Torch Spline', alpha=0.8)
    axes[0, 1].plot(x_sharp_scipy, y_sharp_scipy, 'r--', linewidth=2, label='SciPy Spline', alpha=0.7)
    axes[0, 1].set_xlabel('Normalized Frequency', fontsize=12)
    axes[0, 1].set_ylabel('MTF Value', fontsize=12)
    axes[0, 1].set_title('Sharp Image MTF', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim([0, 1])
    
    # ============ Plot 3: Phantom MTF - Predicted (Torch vs SciPy) ============
    axes[1, 0].plot(x_torch, mtf_phantom_torch, 'b-', linewidth=2.5, label='Torch Spline', alpha=0.8)
    axes[1, 0].plot(x_phantom_scipy, y_phantom_scipy, 'r--', linewidth=2, label='SciPy Spline', alpha=0.7)
    axes[1, 0].set_xlabel('Normalized Frequency', fontsize=12)
    axes[1, 0].set_ylabel('MTF Value', fontsize=12)
    axes[1, 0].set_title('Predicted MTF (from Phantom)', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim([0, 1])
    
    # ============ Plot 4: Predicted vs Ground Truth ============
    axes[1, 1].plot(x_target, target_mtf_np, 'g-', linewidth=2.5, label='Ground Truth MTF', alpha=0.8)
    axes[1, 1].plot(x_torch, mtf_phantom_torch, 'b--', linewidth=2, label='Predicted MTF (Torch)', alpha=0.7)
    axes[1, 1].set_xlabel('Normalized Frequency', fontsize=12)
    axes[1, 1].set_ylabel('MTF Value', fontsize=12)
    axes[1, 1].set_title('Predicted vs Ground Truth MTF', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim([0, 1])
    
    plt.suptitle(f'MTF Splines - Epoch {epoch}', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figure
    save_path = output_dir / f'epoch_{epoch:03d}_splines.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path

@torch.no_grad()
def validate(model, image_loader, mtf_loader, l1_loss, alpha, device):
    """Validation loop"""
    model.eval()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_mtf_loss = 0.0
    num_batches = 0
    
    image_iter = iter(image_loader)
    mtf_iter = iter(mtf_loader)
    
    num_iters = min(len(image_loader), len(mtf_loader))
    
    for batch_idx in range(num_iters):
        try:
            I_smooth, I_sharp, psd_smooth, psd_sharp = next(image_iter)
        except StopIteration:
            break
        
        I_smooth = I_smooth.to(device, non_blocking=True)
        I_sharp = I_sharp.to(device, non_blocking=True)
        psd_smooth = psd_smooth.to(device, non_blocking=True)
        psd_sharp = psd_sharp.to(device, non_blocking=True)
        
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
        
        try:
            input_profiles, target_mtfs = next(mtf_iter)
        except StopIteration:
            break
        
        input_profiles = input_profiles.to(device, non_blocking=True).unsqueeze(1)
        target_mtfs = target_mtfs.to(device, non_blocking=True)
        
        knots_phantom, control_phantom = model(input_profiles)
        mtf_phantom = get_torch_spline(knots_phantom, control_phantom).squeeze(1)
        
        mtf_loss = l1_loss(mtf_phantom, target_mtfs)
        
        batch_loss = alpha * recon_loss + (1 - alpha) * mtf_loss
        
        total_loss += batch_loss.item()
        total_recon_loss += recon_loss.item()
        total_mtf_loss += mtf_loss.item()
        num_batches += 1
    
    return {
        'total_loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'mtf_loss': total_mtf_loss / num_batches
    }

def save_checkpoint(epoch, model, optimizer, scaler, metrics, best_val_loss, 
                   alpha, learning_rate, checkpoint_dir, is_best=False):
    """
    Save training checkpoint
    
    Args:
        epoch: Current epoch number
        model: PyTorch model
        optimizer: Optimizer
        scaler: Gradient scaler (can be None)
        metrics: Dictionary of training metrics
        best_val_loss: Best validation loss so far
        alpha: Loss weighting parameter
        learning_rate: Learning rate
        checkpoint_dir: Directory to save checkpoints
        is_best: Whether this is the best model
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'metrics': metrics,
        'best_val_loss': best_val_loss,
        'alpha': alpha,
        'learning_rate': learning_rate
    }
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save latest checkpoint
    latest_path = checkpoint_dir / "latest_checkpoint.pth"
    torch.save(checkpoint, latest_path)
    
    # Save epoch checkpoint
    epoch_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, epoch_path)
    
    # Save best model
    if is_best:
        best_path = checkpoint_dir / "best_model.pth"
        torch.save(checkpoint, best_path)


def load_checkpoint(checkpoint_path, model, optimizer, scaler=None):
    """
    Load checkpoint to resume training
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model
        optimizer: Optimizer
        scaler: Gradient scaler (optional)
    
    Returns:
        Dictionary with epoch, metrics, and best_val_loss, or None if failed
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}")
        return None
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scaler and checkpoint.get('scaler_state_dict'):
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint['metrics'],
        'best_val_loss': checkpoint['best_val_loss']
    }


# ========================================
# GRADIENT NORM
# ========================================

def compute_gradient_norm(model):
    """
    Compute L2 norm of model gradients
    
    Args:
        model: PyTorch model
    
    Returns:
        float: L2 norm of gradients
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


# ========================================
# PLOTTING FUNCTIONS
# ========================================

def plot_training_metrics(metrics, alpha, learning_rate, output_dir):
    """
    Plot all training metrics
    
    Args:
        metrics: Dictionary containing all tracked metrics
        alpha: Loss weighting parameter
        learning_rate: Learning rate used
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    epochs = metrics['epoch']
    
    # Plot 1: Total Loss (Train vs Val)
    axes[0, 0].plot(epochs, metrics['train_total_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, metrics['val_total_loss'], 'r--', label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Total Loss', fontsize=12)
    axes[0, 0].set_title('Total Loss (Train vs Val)', fontweight='bold', fontsize=14)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Reconstruction Loss (Train vs Val)
    axes[0, 1].plot(epochs, metrics['train_recon_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, metrics['val_recon_loss'], 'r--', label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Reconstruction Loss', fontsize=12)
    axes[0, 1].set_title('Reconstruction Loss (L1)', fontweight='bold', fontsize=14)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: MTF Loss (Train vs Val)
    axes[0, 2].plot(epochs, metrics['train_mtf_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 2].plot(epochs, metrics['val_mtf_loss'], 'r--', label='Val', linewidth=2)
    axes[0, 2].set_xlabel('Epoch', fontsize=12)
    axes[0, 2].set_ylabel('MTF Loss', fontsize=12)
    axes[0, 2].set_title('MTF Loss (L1)', fontweight='bold', fontsize=14)
    axes[0, 2].legend(fontsize=11)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Gradient Norm
    axes[1, 0].plot(epochs, metrics['train_grad_norm'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Gradient Norm', fontsize=12)
    axes[1, 0].set_title('Gradient Norm (L2)', fontweight='bold', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Best Model Indicator
    best_idx = np.argmin(metrics['val_total_loss'])
    best_epoch = epochs[best_idx]
    best_loss = metrics['val_total_loss'][best_idx]
    
    axes[1, 1].plot(epochs, metrics['val_total_loss'], 'b-', linewidth=2)
    axes[1, 1].axvline(x=best_epoch, color='r', linestyle='--', linewidth=2, 
                      label=f'Best Epoch: {best_epoch}')
    axes[1, 1].scatter([best_epoch], [best_loss], color='r', s=150, zorder=5, marker='*')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Validation Total Loss', fontsize=12)
    axes[1, 1].set_title('Best Model', fontweight='bold', fontsize=14)
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Training Summary
    axes[1, 2].axis('off')
    summary_text = f"""
Training Summary
{'='*40}

Total Epochs: {len(epochs)}
Best Epoch: {best_epoch}

Best Val Loss: {best_loss:.6f}

Final Metrics:
  Train Loss: {metrics['train_total_loss'][-1]:.6f}
  Val Loss:   {metrics['val_total_loss'][-1]:.6f}

  Train Recon: {metrics['train_recon_loss'][-1]:.6f}
  Val Recon:   {metrics['val_recon_loss'][-1]:.6f}

  Train MTF: {metrics['train_mtf_loss'][-1]:.6f}
  Val MTF:   {metrics['val_mtf_loss'][-1]:.6f}

  Grad Norm: {metrics['train_grad_norm'][-1]:.6f}

Hyperparameters:
  Alpha: {alpha}
  Learning Rate: {learning_rate}
    """
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                   verticalalignment='center')
    
    plt.tight_layout()
    plot_path = output_dir / 'training_metrics.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training metrics plot saved to {plot_path}")

def radial_average(psd, return_freq=False):
    """
    Calculate radial average of 2D power spectral density
    
    Args:
        psd: 2D array of power spectral density
        return_freq: If True, also return frequency bins
    
    Returns:
        radial_profile: 1D radial average
        radial_freq (optional): corresponding frequency values
    """
    y, x = np.indices(psd.shape)
    center = np.array(psd.shape) / 2.0
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(np.int32)
    
    tbin = np.bincount(r.ravel(), psd.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / (nr + 1e-8)
    
    if return_freq:
        # Generate frequency bins (cycles per pixel)
        # Normalized frequency: 0 to 0.5 cycles/pixel
        max_radius = np.max(r)
        radial_freq = np.arange(len(radial_profile)) / (2.0 * max_radius)
        return radial_profile, radial_freq
    
    return radial_profile

def generate_images(I_smooth, I_sharp, otf_smooth_2d, otf_sharp_2d, epsilon=0.01):
    """
    Implements the exact formulation from your diagram:
    OTF_smooth2sharp = k_sharp / (k_smooth + epsilon)
    OTF_sharp2smooth = k_smooth / (k_sharp + epsilon)
    I_generated_smooth = F^-1[F[I_sharp] * OTF_sharp2smooth]
    I_generated_sharp = F^-1[F[I_smooth] * OTF_smooth2sharp]
    
    Args:
        I_smooth, I_sharp: [B, 1, H, W] - input images
        otf_smooth_2d, otf_sharp_2d: [B, 1, H, W] - 2D OTF maps (k_smooth, k_sharp)
        epsilon: regularization parameter
    
    Returns:
        I_gen_sharp, I_gen_smooth: [B, 1, H, W] - generated images
    """
    # Take FFT of input images
    F_smooth = torch.fft.fft2(I_smooth)  # F[I_smooth]
    F_sharp = torch.fft.fft2(I_sharp)    # F[I_sharp]
    
    # Compute transfer functions exactly as in diagram
    OTF_smooth2sharp = otf_sharp_2d / (otf_smooth_2d + epsilon)
    OTF_sharp2smooth = otf_smooth_2d / (otf_sharp_2d + epsilon)
    
    # Apply transfer functions in frequency domain
    F_gen_sharp = F_smooth * OTF_smooth2sharp   # F[I_smooth] * OTF_smooth2sharp
    F_gen_smooth = F_sharp * OTF_sharp2smooth   # F[I_sharp] * OTF_sharp2smooth
    
    # Take inverse FFT to get spatial domain images
    I_gen_sharp = torch.real(torch.fft.ifft2(F_gen_sharp))   # F^-1[...]
    I_gen_smooth = torch.real(torch.fft.ifft2(F_gen_smooth)) # F^-1[...]
    
    # Normalize to [0, 1] range
    I_gen_sharp = robust_normalize(I_gen_sharp, lower_percentile=2, upper_percentile=98)
    I_gen_smooth = robust_normalize(I_gen_smooth, lower_percentile=2, upper_percentile=98)
    
    return I_gen_sharp, I_gen_smooth


def robust_normalize(img, lower_percentile=2, upper_percentile=98):
    """
    Percentile-based normalization
    Handles both single images [C, H, W] and batches [B, C, H, W]
    """
    # Handle single image case
    if img.ndim == 3:
        img = img.unsqueeze(0)  # Add batch dimension [1, C, H, W]
        squeeze_output = True
    else:
        squeeze_output = False
    
    B, C, H, W = img.shape
    img_normalized = torch.zeros_like(img)
    
    for b in range(B):
        img_b = img[b]
        p_low = torch.quantile(img_b, lower_percentile / 100.0)
        p_high = torch.quantile(img_b, upper_percentile / 100.0)
        
        if p_high > p_low:
            img_normalized[b] = (img_b - p_low) / (p_high - p_low)
        else:
            img_normalized[b] = img_b * 0.5  # fallback to middle gray
    
    img_normalized = torch.clamp(img_normalized, 0, 1)
    
    # Remove batch dimension if input was single image
    if squeeze_output:
        img_normalized = img_normalized.squeeze(0)
    
    return img_normalized

def get_scipy_spline(knots_tensor, control_tensor, degree=3, num_points=363):

    t = knots_tensor.detach().cpu().numpy().flatten()
    c = control_tensor.detach().cpu().numpy().flatten()

    spl = BSpline(t, c, k=degree)

    x_axis = np.linspace(0, 1, num_points)
    y_axis = spl(x_axis)
    
    return x_axis, y_axis

def cox_de_boor(t, k, knots, degree):
    if degree == 0:
        k_start = knots[:, k].unsqueeze(1)
        k_end = knots[:, k+1].unsqueeze(1)
        mask = (t >= k_start) & (t < k_end)
        return mask.float()

    epsilon = 1e-6
    
    term1_num = (t - knots[:, k].unsqueeze(1))
    term1_den = (knots[:, k+degree].unsqueeze(1) - knots[:, k].unsqueeze(1))
    term1 = (term1_num / (term1_den + epsilon)) * cox_de_boor(t, k, knots, degree-1)
    
    term2_num = (knots[:, k+degree+1].unsqueeze(1) - t)
    term2_den = (knots[:, k+degree+1].unsqueeze(1) - knots[:, k+1].unsqueeze(1))
    term2 = (term2_num / (term2_den + epsilon)) * cox_de_boor(t, k+1, knots, degree-1)
    
    return term1 + term2

def get_torch_spline(knots, control, num_points=363, degree=3):
    batch_size = knots.shape[0]
    device = knots.device
    num_control = control.shape[1]
    
    t_eval = torch.linspace(0, 1, num_points, device=device).unsqueeze(0).repeat(batch_size, 1)
    
    final_curve = torch.zeros(batch_size, num_points, device=device)
    
    for i in range(num_control):
        basis_i = cox_de_boor(t_eval, i, knots, degree)
        final_curve += control[:, i].unsqueeze(1) * basis_i
        
    return final_curve.unsqueeze(1) # Shape [Batch, 1, 363]

class Preprocessor:
    def __init__(self, shape=(512, 512)):
        y, x = np.indices(shape)
        center = np.array(shape) / 2.0
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        self.r = r.astype(np.int32)
        self.nr = np.bincount(self.r.ravel())
        target_len = 363
        if len(self.nr) > target_len:
            self.nr = self.nr[:target_len]
        elif len(self.nr) < target_len:
            pad_size = target_len - len(self.nr)
            self.nr = np.pad(self.nr, (0, pad_size), 'constant', constant_values=1)

    def process_slice(self, slice_img):
        f = np.fft.fft2(slice_img)
        fshift = np.fft.fftshift(f)
        psd = np.abs(fshift) ** 2
        tbin = np.bincount(self.r.ravel(), psd.ravel())
        target_len = 363
        if len(tbin) > target_len:
            tbin = tbin[:target_len]
        elif len(tbin) < target_len:
            tbin = np.pad(tbin, (0, target_len - len(tbin)), 'constant')     
        radial_profile = tbin / (self.nr + 1e-8)
        radial_profile = np.log(radial_profile + 1e-8)
        
        return radial_profile

def profile_to_kernel(profile_1d, kernel_size=31):
    device = profile_1d.device
    range_vec = torch.linspace(-1, 1, kernel_size, device=device)
    y, x = torch.meshgrid(range_vec, range_vec, indexing='ij')
    r = torch.sqrt(x**2 + y**2)
    grid_coords = 2.0 * r - 1.0 
    zeros = torch.zeros_like(grid_coords)
    grid = torch.stack([grid_coords, zeros], dim=-1).unsqueeze(0) # [1, K, K, 2]

    inp = profile_1d.view(1, 1, 1, -1)
    
    kernel = F.grid_sample(inp, grid, align_corners=True, padding_mode='border')
    kernel = kernel.view(kernel_size, kernel_size)
    
    mask = (r <= 1.0).float()
    kernel = kernel * mask
    kernel = kernel / (kernel.sum() + 1e-8)
    
    return kernel

def radial_mtf_to_2d_otf(mtf_1d, image_shape, device):
    """
    Convert 1D radial MTF profile to 2D OTF in frequency space
    
    mtf_1d: [B, 1, L] - radial MTF profile (e.g., 363 points)
    image_shape: (H, W) - target image dimensions
    
    Returns: [B, 1, H, W] - 2D OTF ready for frequency domain operations
    """
    B, C, L = mtf_1d.shape
    H, W = image_shape
    
    # Create frequency space radial distance map (centered)
    y_freq = torch.fft.fftfreq(H, device=device).view(-1, 1)
    x_freq = torch.fft.fftfreq(W, device=device).view(1, -1)
    r_freq = torch.sqrt(y_freq**2 + x_freq**2)  # [H, W]
    
    # Normalize radius to [0, 1] range
    nyquist = 0.5  # Nyquist frequency
    r_norm = (r_freq / nyquist).clamp(0, 1)  # [H, W]
    
    # Map normalized radius to MTF profile indices
    indices = (r_norm * (L - 1)).long().clamp(0, L - 1)  # [H, W]
    
    # Build 2D OTF for each batch
    otf_2d = torch.zeros(B, 1, H, W, device=device)
    for b in range(B):
        otf_2d[b, 0] = mtf_1d[b, 0, indices]
    
    return otf_2d

def apply_window(kernel, device):
    """
    Multiplies the kernel by a 2D Hanning window to force edges to zero.
    This eliminates the "cliff" that causes ringing artifacts.
    """
    k_size = kernel.shape[0]
    # Create 1D window (bell curve shape)
    win1d = torch.hann_window(k_size, periodic=False, device=device)
    # Create 2D window by outer product
    win2d = win1d.unsqueeze(1) * win1d.unsqueeze(0)
    return kernel * win2d


