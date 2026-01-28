import pydicom
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Define your paths
input_dir = r"D:\Charan work file\PhantomTesting\MTF-20260127T164932Z-3-001\MTF\kernels"
output_dir = r"D:\Charan work file\PhantomTesting\DICOM_PNG_Exports"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Recursively find all files
files = list(Path(input_dir).rglob('*'))

print(f"Found {len(files)} files. Starting export...")

for file_path in files:
    try:
        # Attempt to read as DICOM
        ds = pydicom.dcmread(str(file_path))
        pixel_array = ds.pixel_array
        
        # Apply Hounsfield scaling if available
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        
        # Determine a filename (e.g., Kernel_FileName.png)
        kernel = getattr(ds, 'FilterType', 'Unknown').replace('/', '_')
        file_name = f"{kernel}_{file_path.name}.png"
        save_path = os.path.join(output_dir, file_name)
        
        # Save without axes/margins for a clean training dataset
        plt.imsave(save_path, pixel_array, cmap='gray')
        print(f"Saved: {file_name}")
        
    except Exception:
        # Skip non-DICOM files
        continue

print(f"Export complete. Images stored in: {output_dir}")