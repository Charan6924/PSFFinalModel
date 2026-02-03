import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from utils import get_torch_spline, compute_psd, compute_fft
from PSDDataset import PSDDataset
from SplineEstimator import KernelEstimator
from torch.utils.data import DataLoader, random_split

def spline_to_kernel(smooth_knots, smooth_control_points, sharp_knots, sharp_control_points, grid_size=512):
    batch_size = smooth_knots.shape[0]
    device = smooth_knots.device

    center = grid_size / 2.0
    y = torch.arange(grid_size, device=device, dtype=torch.float32) - center
    x = torch.arange(grid_size, device=device, dtype=torch.float32) - center
    y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')

    distance = torch.sqrt(x_grid**2 + y_grid**2)
    max_distance = center * np.sqrt(2)
    t = distance / max_distance
    t = torch.clamp(t, 0, 1)

    num_spline_points = 256
    smooth_spline_curve = get_torch_spline(smooth_knots, smooth_control_points, num_points=num_spline_points)
    sharp_spline_curve = get_torch_spline(sharp_knots, sharp_control_points, num_points=num_spline_points)
    
    smooth_spline_curve = smooth_spline_curve.view(batch_size, 1, num_spline_points, 1)
    sharp_spline_curve = sharp_spline_curve.view(batch_size, 1, num_spline_points, 1)
    otf_smooth_to_sharp = smooth_spline_curve / (sharp_spline_curve + 1e-10)
    otf_sharp_to_smooth = sharp_spline_curve / (smooth_spline_curve + 1e-10)

    grid_x = 2.0 * t - 1.0  # [H, W]
    grid_y = torch.zeros_like(grid_x)
    sampling_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    sampling_grid = sampling_grid.expand(batch_size, -1, -1, -1)

    otf_smooth_to_sharp_grid = F.grid_sample(
        otf_smooth_to_sharp,     
        sampling_grid,            
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    ).squeeze(1)  

    otf_sharp_to_smooth_grid = F.grid_sample(
        otf_sharp_to_smooth,      
        sampling_grid,          
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    ).squeeze(1) 

    return otf_smooth_to_sharp_grid, otf_sharp_to_smooth_grid





if __name__ == "__main__":
    BATCH_SIZE = 1
    device = 'cuda'
    data_root = r"D:\Charan work file\KernelEstimator\Data_Root"
    image_dataset = PSDDataset(root_dir=data_root, preload=True)
    
    image_train_size = int(0.9 * len(image_dataset))
    image_val_size = len(image_dataset) - image_train_size
    
    image_train_dataset, image_val_dataset = random_split(
        image_dataset,
        [image_train_size, image_val_size],
        generator=torch.Generator().manual_seed(42)
    )
    image_train_loader = DataLoader(
        image_train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True,
    )

    # Load model
    model = KernelEstimator().to(device)
    checkpoint_path = r'D:\Charan work file\PhantomTesting\Code\best_checkpoint.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to('cuda')
    model.eval()
    
    for index, (I_smooth, I_sharp) in enumerate(image_train_loader):
        with torch.no_grad():  
            I_smooth = I_smooth.to(device)
            I_sharp = I_sharp.to(device)
            psd_smooth = compute_psd(I_smooth, device=device)
            psd_sharp = compute_psd(I_sharp, device=device)
            smooth_knots, smooth_control = model(psd_smooth)
            sharp_knots, sharp_control = model(psd_sharp)
            otf_smooth_to_sharp_grid, otf_sharp_to_smooth_grid = spline_to_kernel(
                smooth_knots=smooth_knots,
                smooth_control_points=smooth_control,
                sharp_control_points=sharp_control,
                sharp_knots=sharp_knots,
                grid_size=I_smooth.shape[-1]
            )
            
            fft_smooth = compute_fft(I_smooth, device=device)
            fft_sharp = compute_fft(I_sharp, device=device)
            
            fft_generated_sharp = fft_smooth * otf_smooth_to_sharp_grid
            fft_generated_smooth = fft_sharp * otf_sharp_to_smooth_grid
            
            fft_generated_sharp_unshifted = torch.fft.ifftshift(fft_generated_sharp)
            fft_generated_smooth_unshifted = torch.fft.ifftshift(fft_generated_smooth)
            
            I_generated_sharp = torch.fft.ifft2(fft_generated_sharp_unshifted)
            I_generated_smooth = torch.fft.ifft2(fft_generated_smooth_unshifted)
            
            I_generated_sharp_real = I_generated_sharp.real
            I_generated_smooth_real = I_generated_smooth.real
        
            I_smooth_np = I_smooth[0, 0].cpu().numpy()
            I_sharp_np = I_sharp[0, 0].cpu().numpy()
            I_generated_sharp_np = I_generated_sharp_real[0].cpu().numpy()
            I_generated_smooth_np = I_generated_smooth_real[0].cpu().numpy()
            
            break