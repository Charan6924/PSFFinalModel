import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import os
from collections import defaultdict

# Configuration
output_dir = r'D:\Charan work file\PhantomTesting\PSD_Results_Output'

# Load ALL files and organize by kernel
kernel_data = defaultdict(list)

print("Loading PSD files...")
for npy_file in Path(output_dir).glob('*_PSD.npy'):
    meta_file = str(npy_file).replace('.npy', '_metadata.json')
    if Path(meta_file).exists():
        with open(meta_file, 'r') as f:
            meta = json.load(f)
            kernel = meta['kernel']
            psd = np.load(npy_file)
            kernel_data[kernel].append({
                'psd': psd,
                'file': npy_file.name,
                'meta': meta
            })

# Print statistics
print("\n" + "="*60)
print("KERNEL STATISTICS")
print("="*60)
for kernel in sorted(kernel_data.keys()):
    count = len(kernel_data[kernel])
    print(f"Kernel {kernel:8s}: {count:4d} files")
print("="*60)

# ============================================
# VISUALIZATION: One sample per kernel
# ============================================
print("\nCreating kernel comparison (1 sample per kernel)...")

kernels = sorted(kernel_data.keys())
n_kernels = len(kernels)

# Determine grid layout
n_cols = min(4, n_kernels)
n_rows = (n_kernels + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
axes = axes.flatten() if n_kernels > 1 else [axes]

for i, kernel in enumerate(kernels):
    data_list = kernel_data[kernel]
    
    # Get one random sample
    data = np.random.choice(data_list)
    psd = data['psd']
    
    ax = axes[i]
    im = ax.imshow(psd, cmap='hot', aspect='auto')
    ax.set_title(f'Kernel {kernel}', 
                fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Power', rotation=270, labelpad=15, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

# Hide unused subplots
for i in range(n_kernels, len(axes)):
    axes[i].axis('off')

plt.suptitle('Fourier transform of each kernel)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'kernel_comparison_one_each.png'), 
            dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: kernel_comparison_one_each.png")
plt.show()

print("\n" + "="*60)
print("✓ Analysis complete!")
print("="*60)
print(f"\nSaved to: {output_dir}")