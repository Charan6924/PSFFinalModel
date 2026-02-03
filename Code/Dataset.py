import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class MTFPSDDataset(Dataset):
    """PyTorch Dataset class for paired MTF and PSD data"""
    
    def __init__(
        self, 
        mtf_folder: str, 
        psd_folder: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        return_paths: bool = False,
        verbose: bool = True
    ):
        """
        Initialize the dataset by pairing MTF and PSD files.
        
        Args:
            mtf_folder: Path to folder containing MTF .mat files
            psd_folder: Path to folder containing PSD .npy files
            transform: Optional transform to apply to MTF data
            target_transform: Optional transform to apply to PSD data
            return_paths: If True, also return file paths in __getitem__
            verbose: If True, print warnings about unmatched files
        """
        self.mtf_folder = Path(mtf_folder)
        self.psd_folder = Path(psd_folder)
        self.transform = transform
        self.target_transform = target_transform
        self.return_paths = return_paths
        self.verbose = verbose
        self.pairs = self._pair_files()
        
    def _extract_identifier(self, filename: str, file_type: str) -> str:
        """Extract the common identifier from a filename."""
        if file_type == 'mtf':
            return filename.replace('_MTF_Results_mat.mat', '')
        elif file_type == 'psd':
            return filename.replace('_PSD.npy', '')
        return filename
    
    def _pair_files(self) -> List[Dict[str, any]]:
        """Pair MTF and PSD files based on their identifiers."""
        mtf_files = list(self.mtf_folder.glob('*_MTF_Results_mat.mat'))
        psd_files = list(self.psd_folder.glob('*_PSD.npy'))
        
        psd_dict = {}
        for psd_file in psd_files:
            identifier = self._extract_identifier(psd_file.name, 'psd')
            psd_dict[identifier] = psd_file
        
        pairs = []
        unmatched_mtf = []
        
        for mtf_file in mtf_files:
            identifier = self._extract_identifier(mtf_file.name, 'mtf')
            
            if identifier in psd_dict:
                pairs.append({
                    'identifier': identifier,
                    'mtf_path': mtf_file,
                    'psd_path': psd_dict[identifier]
                })
            else:
                unmatched_mtf.append(mtf_file.name)
        
        matched_identifiers = {pair['identifier'] for pair in pairs}
        unmatched_psd = []
        for identifier, psd_path in psd_dict.items():
            if identifier not in matched_identifiers:
                unmatched_psd.append(psd_path.name)
        
        if self.verbose:
            if unmatched_mtf:
                print(f"Warning: {len(unmatched_mtf)} MTF file(s) without matching PSD")
            if unmatched_psd:
                print(f"Warning: {len(unmatched_psd)} PSD file(s) without matching MTF")
        
        pairs.sort(key=lambda x: x['identifier'])
        return pairs
    
    def __len__(self) -> int:
        """Return the number of paired samples"""
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a paired sample by index.
        
        Returns:
            Tuple of (input_profile, target_mtf):
                - input_profile: PSD data as 2D tensor with channel dimension (1, H, W)
                - target_mtf: MTF values as 1D tensor (N,)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        pair = self.pairs[idx]
        
        # Load MTF data
        mtf_data = loadmat(str(pair['mtf_path']))
        
        # Extract MTF values from the structured .mat file
        results = mtf_data['results']
        mtf_val = results['mtfVal'][0, 0][0]  # Shape: (64,) or similar
        mtf_val = np.nan_to_num(mtf_val, nan=0.0)
        
        # Load PSD data
        psd_data = np.load(str(pair['psd_path']))
        
        # Apply transforms if provided
        if self.transform:
            mtf_val = self.transform(mtf_val)
        
        if self.target_transform:
            psd_data = self.target_transform(psd_data)
        
        # Ensure proper dtypes
        # Keep PSD as 2D (H, W), don't flatten!
        psd_data = np.array(psd_data, dtype=np.float32)
        mtf_val = np.array(mtf_val, dtype=np.float32).flatten()
        
        # Convert to tensors
        # Add channel dimension to PSD: (H, W) -> (1, H, W)
        if psd_data.ndim == 2:
            input_profile = torch.from_numpy(psd_data).unsqueeze(0)  # (1, H, W)
        elif psd_data.ndim == 3 and psd_data.shape[0] == 1:
            input_profile = torch.from_numpy(psd_data)  # Already (1, H, W)
        else:
            # Handle unexpected shapes
            input_profile = torch.from_numpy(psd_data.reshape(1, *psd_data.shape[-2:]))
        
        target_mtf = torch.from_numpy(mtf_val)
        
        return input_profile, target_mtf
    
    def get_sample_dict(self, idx: int) -> Dict[str, any]:
        """
        Get a sample as a dictionary (for visualization purposes).
        This is the old behavior, useful for plotting.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        pair = self.pairs[idx]
        
        mtf_data = loadmat(str(pair['mtf_path']))
        psd_data = np.load(str(pair['psd_path']))
        
        sample = {
            'identifier': pair['identifier'],
            'mtf': mtf_data,
            'psd': psd_data,
        }
        
        if self.return_paths:
            sample['mtf_path'] = str(pair['mtf_path'])
            sample['psd_path'] = str(pair['psd_path'])
        
        return sample
    
    def get_identifiers(self) -> List[str]:
        """Return list of all identifiers in the dataset"""
        return [pair['identifier'] for pair in self.pairs]


def extract_mtf_data(mtf_dict: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract MTF axis, values, and error from the .mat file structure.
    
    Args:
        mtf_dict: Dictionary loaded from .mat file
    
    Returns:
        Tuple of (mtf_axis, mtf_values, mtf_error) as numpy arrays
    """
    results = mtf_dict['results']
    
    # Extract the fields from the structured array
    mtf_axis = results['mtfAxis'][0, 0][0]  # Shape: (64,)
    mtf_val = results['mtfVal'][0, 0][0]    # Shape: (64,)
    mtf_error = results['mtfError'][0, 0][0]  # Shape: (64,)
    
    return mtf_axis, mtf_val, mtf_error


def visualize_sample(sample: dict, save_path: Optional[str] = None):
    """
    Visualize a single MTF-PSD pair.
    
    Args:
        sample: Sample from the dataset containing 'mtf' and 'psd'
        save_path: Optional path to save the figure
    """
    # Extract MTF data
    mtf_axis, mtf_val, mtf_error = extract_mtf_data(sample['mtf'])
    
    # Get PSD data
    psd = sample['psd']
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    
    # Plot MTF curve
    ax1 = plt.subplot(gs[0])
    ax1.plot(mtf_axis, mtf_val, 'b-', linewidth=2, label='MTF')
    ax1.fill_between(mtf_axis, 
                      mtf_val - mtf_error, 
                      mtf_val + mtf_error, 
                      alpha=0.3, 
                      label='±Error')
    ax1.set_xlabel('Spatial Frequency (cycles/mm)', fontsize=12)
    ax1.set_ylabel('MTF', fontsize=12)
    ax1.set_title(f'MTF Curve - {sample["identifier"]}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0, 1.1])
    
    # Plot PSD image
    ax2 = plt.subplot(gs[1])
    # Use log scale for better visualization of PSD
    psd_display = np.log10(psd + 1e-10)  # Add small value to avoid log(0)
    im = ax2.imshow(psd_display, cmap='hot', aspect='auto')
    ax2.set_xlabel('Frequency X', fontsize=12)
    ax2.set_ylabel('Frequency Y', fontsize=12)
    ax2.set_title(f'PSD (log scale) - {sample["identifier"]}', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Log10(PSD)', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()


def visualize_multiple_samples(dataset: MTFPSDDataset, 
                               num_samples: int = 4, 
                               indices: Optional[List[int]] = None,
                               save_path: Optional[str] = None):
    if indices is None:
        # Select evenly spaced samples
        indices = np.linspace(0, len(dataset)-1, num_samples, dtype=int)
    
    num_samples = len(indices)
    
    # Create figure
    fig = plt.figure(figsize=(14, 4*num_samples))
    gs = gridspec.GridSpec(num_samples, 2, width_ratios=[1, 1], hspace=0.3)
    
    for i, idx in enumerate(indices):
        sample = dataset.get_sample_dict(idx)  # Use get_sample_dict for visualization
        
        # Extract MTF data
        mtf_axis, mtf_val, mtf_error = extract_mtf_data(sample['mtf'])
        psd = sample['psd']
        
        # Plot MTF curve
        ax1 = plt.subplot(gs[i, 0])
        ax1.plot(mtf_axis, mtf_val, 'b-', linewidth=2)
        ax1.fill_between(mtf_axis, 
                         mtf_val - mtf_error, 
                         mtf_val + mtf_error, 
                         alpha=0.3)
        ax1.set_xlabel('Spatial Frequency (cycles/mm)', fontsize=10)
        ax1.set_ylabel('MTF', fontsize=10)
        ax1.set_title(f'MTF - {sample["identifier"]}', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.1])
        
        # Plot PSD image
        ax2 = plt.subplot(gs[i, 1])
        psd_display = np.log10(psd + 1e-10)
        im = ax2.imshow(psd_display, cmap='hot', aspect='auto')
        ax2.set_xlabel('Frequency X', fontsize=10)
        ax2.set_ylabel('Frequency Y', fontsize=10)
        ax2.set_title(f'PSD (log scale) - {sample["identifier"]}', fontsize=12, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Log10(PSD)', fontsize=8)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()


def compare_kernels(dataset: MTFPSDDataset, 
                   base_name: str = "I100",
                   save_path: Optional[str] = None):
    identifiers = dataset.get_identifiers()
    matching_samples = []
    
    for identifier in identifiers:
        if identifier.startswith(base_name + "_"):
            idx = identifiers.index(identifier)
            matching_samples.append(dataset.get_sample_dict(idx))  # Use get_sample_dict
    
    if not matching_samples:
        print(f"No samples found with base name '{base_name}'")
        return
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot MTF curves
    colors = plt.cm.tab10(np.linspace(0, 1, len(matching_samples)))
    
    for i, sample in enumerate(matching_samples):
        mtf_axis, mtf_val, mtf_error = extract_mtf_data(sample['mtf'])
        kernel = sample['identifier'].split('_')[-1]
        
        ax1.plot(mtf_axis, mtf_val, linewidth=2, color=colors[i], 
                label=f'Kernel {kernel}')
        ax1.fill_between(mtf_axis, 
                        mtf_val - mtf_error, 
                        mtf_val + mtf_error, 
                        alpha=0.2,
                        color=colors[i])
    
    ax1.set_xlabel('Spatial Frequency (cycles/mm)', fontsize=12)
    ax1.set_ylabel('MTF', fontsize=12)
    ax1.set_title(f'MTF Comparison - {base_name} (All Kernels)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    ax1.set_ylim([0, 1.1])
    
    # Plot PSD images in a grid
    n_samples = len(matching_samples)
    n_cols = min(4, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    # Remove ax2 and create gridspec for PSDs
    ax2.remove()
    gs = gridspec.GridSpec(n_rows, n_cols, 
                          left=0.55, right=0.98, 
                          top=0.95, bottom=0.05,
                          hspace=0.3, wspace=0.3)
    
    for i, sample in enumerate(matching_samples):
        psd = sample['psd']
        kernel = sample['identifier'].split('_')[-1]
        
        row = i // n_cols
        col = i % n_cols
        ax = plt.subplot(gs[row, col])
        
        psd_display = np.log10(psd + 1e-10)
        im = ax.imshow(psd_display, cmap='hot', aspect='auto')
        ax.set_title(f'Kernel {kernel}', fontsize=10)
        ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("="*80)
    print("MTF-PSD VISUALIZATION")
    print("="*80)
    
    # Define paths
    mtf_folder = r"D:\Charan work file\PhantomTesting\MTF_Results_Output"
    psd_folder = r"D:\Charan work file\PhantomTesting\PSD_Results_Output"
    
    # Create dataset
    print("\nLoading dataset...")
    dataset = MTFPSDDataset(mtf_folder, psd_folder, verbose=False)
    print(f"✓ Loaded {len(dataset)} paired samples")
    
    # Show available identifiers
    identifiers = dataset.get_identifiers()
    print(f"\nFirst 10 identifiers:")
    for i, identifier in enumerate(identifiers[:10]):
        print(f"  {i}: {identifier}")
    
    # Visualize single sample
    print("\n" + "="*80)
    print("1. Visualizing first sample...")
    print("="*80)
    sample = dataset.get_sample_dict(0)
    visualize_sample(sample)
    
    # Visualize multiple samples
    print("\n" + "="*80)
    print("2. Visualizing 4 samples...")
    print("="*80)
    visualize_multiple_samples(dataset, num_samples=4, indices=[0, 10, 20, 30])
    
    # Compare kernels
    print("\n" + "="*80)
    print("3. Comparing different kernels for I100...")
    print("="*80)
    compare_kernels(dataset, base_name="I100")
    
    print("\n✓ Visualization complete!")