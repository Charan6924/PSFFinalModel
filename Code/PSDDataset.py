import os
import nibabel as nib
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def compute_psd_batch_gpu(image_batch, device):
    """
    Compute PSD for a batch of images on GPU.
    This should be called IN THE TRAINING LOOP, not in collate_fn.
    
    Args:
        image_batch: Tensor of shape (B, C, H, W) already on GPU
        device: Device to compute on
    
    Returns:
        psd: Normalized PSD tensor of shape (B, C, H, W)
    """
    if image_batch.device.type != device:
        image_batch = image_batch.to(device)
    
    # Compute FFT and shift to center
    freq_map = torch.fft.fftshift(torch.fft.fft2(image_batch), dim=(-2, -1))
    
    # Compute PSD
    psd = torch.abs(freq_map) ** 2
    psd = torch.log(psd + 1)
    
    # Normalize per image
    b, c, h, w = psd.shape
    psd_flat = psd.view(b, -1)
    p_min = psd_flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    p_max = psd_flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    
    psd = (psd - p_min) / (p_max - p_min + 1e-8)
    
    return psd


class PSDDataset(Dataset):
    def __init__(self, root_dir, min_slice_percentile=0.1, max_slice_percentile=0.9,
                 preload=True, seed=42):
        """
        Dataset that returns images only. PSD computation is done separately on GPU batches.
        
        Args:
            root_dir: Root directory containing trainA and trainB
            min_slice_percentile: Minimum slice percentile to use
            max_slice_percentile: Maximum slice percentile to use
            preload: Whether to preload all volumes into memory
            seed: Random seed for reproducibility
        """
        self.smooth_dir = os.path.join(root_dir, "trainA")
        self.sharp_dir = os.path.join(root_dir, "trainB")
        self.min_percentile = min_slice_percentile
        self.max_percentile = max_slice_percentile
        self.preload = preload
        np.random.seed(seed)
        
        print("Finding volume pairs...")
        volume_pairs = self._find_volume_pairs()
        print(f"Found {len(volume_pairs)} volume pairs")
        
        if self.preload:
            self.volume_cache = self._preload_all_volumes(volume_pairs)
            print(f"Cached {len(self.volume_cache)} unique volumes")
        else:
            self.volume_cache = {}
            print("Using lazy loading - first epoch will be slower")
        
        self.slice_data = self._build_slice_index(volume_pairs)
        print(f"Total slices: {len(self.slice_data)}")
        
        if self.preload and self.volume_cache:
            total_bytes = sum(v.nbytes for v in self.volume_cache.values())
            print(f"Memory usage: {total_bytes / (1024**3):.2f} GB")

    def _find_volume_pairs(self):
        smooth_files = sorted([f for f in os.listdir(self.smooth_dir) 
                              if f.endswith(('.nii', '.nii.gz'))])
        sharp_files = sorted([f for f in os.listdir(self.sharp_dir) 
                             if f.endswith(('.nii', '.nii.gz'))])
        
        sharp_dict = {
            (f.split("_filter_")[0] if "_filter_" in f else f.split(".")[0]): f 
            for f in sharp_files
        }
        
        volume_pairs = []
        for sfile in smooth_files:
            base_id = (sfile.split("_filter_")[0] if "_filter_" in sfile 
                      else sfile.split(".")[0])
            if base_id in sharp_dict:
                volume_pairs.append((sfile, sharp_dict[base_id]))
        
        return volume_pairs

    def _preload_all_volumes(self, volume_pairs):
        unique_paths = set()
        for sfile, shfile in volume_pairs:
            unique_paths.add(os.path.join(self.smooth_dir, sfile))
            unique_paths.add(os.path.join(self.sharp_dir, shfile))
        
        cache = {}
        for path in tqdm(sorted(unique_paths), desc="Loading volumes"):
            try:
                cache[path] = nib.load(path).get_fdata()
            except Exception as e:
                print(f"\nFailed to load {os.path.basename(path)}: {e}")
        
        return cache

    def _build_slice_index(self, volume_pairs):
        """Build index of all valid slices"""
        slice_data = []
        
        for sfile, shfile in tqdm(volume_pairs, desc="Indexing slices"):
            s_path = os.path.join(self.smooth_dir, sfile)
            sh_path = os.path.join(self.sharp_dir, shfile)
            
            try:
                if self.preload:
                    n_slices = self.volume_cache[s_path].shape[2]
                else:
                    n_slices = nib.load(s_path).shape[2]
                start_idx = int(n_slices * self.min_percentile)
                end_idx = int(n_slices * self.max_percentile)

                for z_idx in range(start_idx, end_idx):
                    slice_data.append({
                        'smooth_path': s_path,
                        'sharp_path': sh_path,
                        'slice_idx': z_idx
                    })
            
            except Exception as e:
                print(f"\nFailed to index {os.path.basename(s_path)}: {e}")
                continue
        
        return slice_data

    def _get_volume(self, path):
        """Get volume from cache or load on-demand"""
        if path in self.volume_cache:
            return self.volume_cache[path]
        
        vol = nib.load(path).get_fdata()
        if not self.preload:
            self.volume_cache[path] = vol
        return vol

    def __getitem__(self, idx):
        """
        Returns only the images. PSD will be computed on GPU after batching.
        
        Returns:
            I_smooth: Normalized smooth image (1, H, W)
            I_sharp: Normalized sharp image (1, H, W)
        """
        info = self.slice_data[idx]
        
        # Get volumes (from cache if preloaded)
        vol_s = self._get_volume(info['smooth_path'])
        vol_h = self._get_volume(info['sharp_path'])
        
        # Extract slice
        img_s = vol_s[:, :, info['slice_idx']].copy()
        img_h = vol_h[:, :, info['slice_idx']].copy()
        
        # Normalize
        img_s = (img_s - img_s.min()) / (img_s.max() - img_s.min() + 1e-8)
        img_h = (img_h - img_h.min()) / (img_h.max() - img_h.min() + 1e-8)
        
        # Convert to tensors
        I_smooth = torch.from_numpy(img_s).unsqueeze(0).float()
        I_sharp = torch.from_numpy(img_h).unsqueeze(0).float()
        
        return I_smooth, I_sharp

    def __len__(self):
        return len(self.slice_data)


def benchmark_dataloader(data_root):
    """
    Benchmark the fixed dataloader that computes PSD on GPU in the training loop.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*80)
    print("BENCHMARKING FIXED DATALOADER")
    print("="*80)
    print(f"Device: {device}")
    print()
    
    # Initialize dataset
    dataset = PSDDataset(root_dir=data_root, preload=True)
    
    # Create dataloader - NO custom collate needed, just default stacking
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    print("\nTesting dataloader output...")
    I_smooth, I_sharp = next(iter(dataloader))
    
    print(f"I_smooth shape: {I_smooth.shape}")
    print(f"I_sharp shape: {I_sharp.shape}")
    print(f"Tensors on device: {I_smooth.device} (CPU - will be moved to GPU in training loop)")
    
    # Move to GPU and compute PSD
    print("\nMoving to GPU and computing PSD...")
    transfer_start = time.time()
    I_smooth_gpu = I_smooth.to(device, non_blocking=True)
    I_sharp_gpu = I_sharp.to(device, non_blocking=True)
    if device == "cuda":
        torch.cuda.synchronize()
    transfer_time = time.time() - transfer_start
    
    psd_start = time.time()
    psd_smooth = compute_psd_batch_gpu(I_smooth_gpu, device)
    psd_sharp = compute_psd_batch_gpu(I_sharp_gpu, device)
    if device == "cuda":
        torch.cuda.synchronize()
    psd_time = time.time() - psd_start
    
    print(f"  Transfer to GPU: {transfer_time:.4f}s")
    print(f"  PSD computation: {psd_time:.4f}s")
    print(f"  Total: {transfer_time + psd_time:.4f}s")
    print(f"  psd_smooth shape: {psd_smooth.shape}")
    print(f"  psd_sharp shape: {psd_sharp.shape}")
    
    print("\n" + "="*80)
    print("Running 20 batch benchmark...")
    print("="*80)
    
    times = []
    transfer_times = []
    psd_times = []
    total_start = time.time()
    
    for i, (I_s, I_h) in enumerate(dataloader):
        if i >= 20:
            break
        
        batch_start = time.time()
        
        # Move to GPU
        transfer_start = time.time()
        I_s = I_s.to(device, non_blocking=True)
        I_h = I_h.to(device, non_blocking=True)
        if device == "cuda":
            torch.cuda.synchronize()
        transfer_time = time.time() - transfer_start
        transfer_times.append(transfer_time)
        
        # Compute PSD on GPU
        psd_compute_start = time.time()
        psd_s = compute_psd_batch_gpu(I_s, device)
        psd_h = compute_psd_batch_gpu(I_h, device)
        if device == "cuda":
            torch.cuda.synchronize()
        psd_compute_time = time.time() - psd_compute_start
        psd_times.append(psd_compute_time)
        
        # Dummy operation
        _ = I_s.mean()
        
        batch_time = time.time() - batch_start
        times.append(batch_time)
        
        if i == 0:
            print(f"\nFirst batch breakdown:")
            print(f"  Transfer: {transfer_time:.4f}s")
            print(f"  PSD:      {psd_compute_time:.4f}s")
            print(f"  Total:    {batch_time:.4f}s")
    
    total_time = time.time() - total_start
    avg_time = np.mean(times)
    avg_transfer = np.mean(transfer_times)
    avg_psd_time = np.mean(psd_times)
    
    print(f"\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Average batch time:      {avg_time:.4f}s")
    print(f"  - Transfer to GPU:     {avg_transfer:.4f}s ({avg_transfer/avg_time*100:.1f}%)")
    print(f"  - PSD computation:     {avg_psd_time:.4f}s ({avg_psd_time/avg_time*100:.1f}%)")
    print(f"Total time (20 batches): {total_time:.2f}s")
    print(f"Throughput:              {64 * 20 / total_time:.1f} images/sec")
    print("="*80)
    
    if device == "cuda":
        print(f"\nGPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    DATA_PATH = r"D:\Charan work file\KernelEstimator\Data_Root"
    benchmark_dataloader(DATA_PATH)