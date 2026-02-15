import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PSDDataset import PSDDataset
from Dataset import MTFPSDDataset
from SplineEstimator import KernelEstimator
from utils import (
    generate_images, 
    get_torch_spline, 
    compute_psd,
    spline_to_kernel
)
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np

def analyze_data_ranges(model, image_loader, mtf_loader, device):
    """Analyze data ranges for inputs and outputs"""
    model.eval()
    
    # Storage for statistics
    stats = {
        'input_images': {
            'I_smooth_1': {'min': [], 'max': [], 'mean': [], 'std': []},
            'I_sharp_1': {'min': [], 'max': [], 'mean': [], 'std': []},
            'I_smooth_2': {'min': [], 'max': [], 'mean': [], 'std': []},
            'I_sharp_2': {'min': [], 'max': [], 'mean': [], 'std': []},
        },
        'psd_inputs': {
            'psd_smooth_1': {'min': [], 'max': [], 'mean': [], 'std': []},
            'psd_sharp_1': {'min': [], 'max': [], 'mean': [], 'std': []},
            'psd_smooth_2': {'min': [], 'max': [], 'mean': [], 'std': []},
            'psd_sharp_2': {'min': [], 'max': [], 'mean': [], 'std': []},
        },
        'generated_images': {
            'I_generated_smooth': {'min': [], 'max': [], 'mean': [], 'std': []},
            'I_generated_sharp': {'min': [], 'max': [], 'mean': [], 'std': []},
        },
        'spline_outputs': {
            'smooth_knots': {'min': [], 'max': [], 'mean': [], 'std': []},
            'smooth_control': {'min': [], 'max': [], 'mean': [], 'std': []},
            'sharp_knots': {'min': [], 'max': [], 'mean': [], 'std': []},
            'sharp_control': {'min': [], 'max': [], 'mean': [], 'std': []},
        },
        'kernels': {
            'otf_smooth_to_sharp': {'min': [], 'max': [], 'mean': [], 'std': []},
            'otf_sharp_to_smooth': {'min': [], 'max': [], 'mean': [], 'std': []},
        },
        'mtf': {
            'input_profiles': {'min': [], 'max': [], 'mean': [], 'std': []},
            'target_mtfs': {'min': [], 'max': [], 'mean': [], 'std': []},
            'predicted_mtfs': {'min': [], 'max': [], 'mean': [], 'std': []},
        }
    }
    
    num_batches = min(50, len(image_loader))  # Sample 50 batches
    
    with torch.no_grad():
        # Analyze image reconstruction
        print("Analyzing image data...")
        for idx, (I_smooth_1, I_sharp_1, I_smooth_2, I_sharp_2) in enumerate(
            tqdm(image_loader, desc="Image batches", total=num_batches)
        ):
            if idx >= num_batches:
                break
                
            I_smooth_1 = I_smooth_1.to(device)
            I_sharp_1 = I_sharp_1.to(device)
            I_smooth_2 = I_smooth_2.to(device)
            I_sharp_2 = I_sharp_2.to(device)
            
            # Collect input image stats
            for name, tensor in [
                ('I_smooth_1', I_smooth_1),
                ('I_sharp_1', I_sharp_1),
                ('I_smooth_2', I_smooth_2),
                ('I_sharp_2', I_sharp_2)
            ]:
                stats['input_images'][name]['min'].append(tensor.min().item())
                stats['input_images'][name]['max'].append(tensor.max().item())
                stats['input_images'][name]['mean'].append(tensor.mean().item())
                stats['input_images'][name]['std'].append(tensor.std().item())
            
            # Compute PSDs
            psd_sharp_1 = compute_psd(I_sharp_1, device=device)
            psd_smooth_1 = compute_psd(I_smooth_1, device=device)
            psd_sharp_2 = compute_psd(I_sharp_2, device=device)
            psd_smooth_2 = compute_psd(I_smooth_2, device=device)
            
            # Collect PSD stats
            for name, tensor in [
                ('psd_smooth_1', psd_smooth_1),
                ('psd_sharp_1', psd_sharp_1),
                ('psd_smooth_2', psd_smooth_2),
                ('psd_sharp_2', psd_sharp_2)
            ]:
                stats['psd_inputs'][name]['min'].append(tensor.min().item())
                stats['psd_inputs'][name]['max'].append(tensor.max().item())
                stats['psd_inputs'][name]['mean'].append(tensor.mean().item())
                stats['psd_inputs'][name]['std'].append(tensor.std().item())
            
            # Forward pass
            smooth_knots_1, smooth_control_points_1 = model(psd_smooth_1)
            sharp_knots_2, sharp_control_points_2 = model(psd_sharp_2)
            
            # Collect spline output stats
            for name, tensor in [
                ('smooth_knots', smooth_knots_1),
                ('smooth_control', smooth_control_points_1),
                ('sharp_knots', sharp_knots_2),
                ('sharp_control', sharp_control_points_2)
            ]:
                stats['spline_outputs'][name]['min'].append(tensor.min().item())
                stats['spline_outputs'][name]['max'].append(tensor.max().item())
                stats['spline_outputs'][name]['mean'].append(tensor.mean().item())
                stats['spline_outputs'][name]['std'].append(tensor.std().item())
            
            # Generate kernels
            otf_smooth_to_sharp_grid, otf_sharp_to_smooth_grid = spline_to_kernel(
                smooth_knots=smooth_knots_1,
                smooth_control_points=smooth_control_points_1,
                sharp_control_points=sharp_control_points_2,
                sharp_knots=sharp_knots_2
            )
            
            # Collect kernel stats
            for name, tensor in [
                ('otf_smooth_to_sharp', otf_smooth_to_sharp_grid),
                ('otf_sharp_to_smooth', otf_sharp_to_smooth_grid)
            ]:
                stats['kernels'][name]['min'].append(tensor.min().item())
                stats['kernels'][name]['max'].append(tensor.max().item())
                stats['kernels'][name]['mean'].append(tensor.mean().item())
                stats['kernels'][name]['std'].append(tensor.std().item())
            
            # Generate images
            I_generated_sharp, I_generated_smooth = generate_images(
                I_smooth=I_smooth_1,
                I_sharp=I_sharp_2, 
                otf_sharp_to_smooth_grid=otf_sharp_to_smooth_grid,
                otf_smooth_to_sharp_grid=otf_smooth_to_sharp_grid
            )
            
            # Collect generated image stats
            for name, tensor in [
                ('I_generated_smooth', I_generated_smooth),
                ('I_generated_sharp', I_generated_sharp)
            ]:
                stats['generated_images'][name]['min'].append(tensor.min().item())
                stats['generated_images'][name]['max'].append(tensor.max().item())
                stats['generated_images'][name]['mean'].append(tensor.mean().item())
                stats['generated_images'][name]['std'].append(tensor.std().item())
        
        # Analyze MTF data
        print("\nAnalyzing MTF data...")
        num_mtf_batches = min(50, len(mtf_loader))
        for idx, (input_profiles, target_mtfs) in enumerate(
            tqdm(mtf_loader, desc="MTF batches", total=num_mtf_batches)
        ):
            if idx >= num_mtf_batches:
                break
                
            input_profiles = input_profiles.to(device)
            target_mtfs = target_mtfs.to(device)
            
            # Collect MTF input stats
            for name, tensor in [
                ('input_profiles', input_profiles),
                ('target_mtfs', target_mtfs)
            ]:
                stats['mtf'][name]['min'].append(tensor.min().item())
                stats['mtf'][name]['max'].append(tensor.max().item())
                stats['mtf'][name]['mean'].append(tensor.mean().item())
                stats['mtf'][name]['std'].append(tensor.std().item())
            
            # Forward pass
            knots_phantom, control_phantom = model(input_profiles)
            mtf_phantom = get_torch_spline(knots_phantom, control_phantom, num_points=64).squeeze(1)
            
            # Collect predicted MTF stats
            stats['mtf']['predicted_mtfs']['min'].append(mtf_phantom.min().item())
            stats['mtf']['predicted_mtfs']['max'].append(mtf_phantom.max().item())
            stats['mtf']['predicted_mtfs']['mean'].append(mtf_phantom.mean().item())
            stats['mtf']['predicted_mtfs']['std'].append(mtf_phantom.std().item())
    
    return stats


def compute_aggregate_stats(stats):
    """Compute aggregate statistics from collected data"""
    aggregated = {}
    
    for category in stats:
        aggregated[category] = {}
        for name in stats[category]:
            data = stats[category][name]
            aggregated[category][name] = {
                'global_min': float(np.min(data['min'])),
                'global_max': float(np.max(data['max'])),
                'avg_mean': float(np.mean(data['mean'])),
                'avg_std': float(np.mean(data['std'])),
                'range': float(np.max(data['max']) - np.min(data['min']))
            }
    
    return aggregated


def print_analysis(aggregated_stats):
    """Print formatted analysis"""
    
    print("\n" + "="*80)
    print("DATA RANGE ANALYSIS")
    print("="*80)
    
    # Input Images
    print("\n### INPUT IMAGES ###")
    for name in ['I_smooth_1', 'I_sharp_1', 'I_smooth_2', 'I_sharp_2']:
        s = aggregated_stats['input_images'][name]
        print(f"{name:15s}: [{s['global_min']:8.4f}, {s['global_max']:8.4f}]  "
              f"Mean={s['avg_mean']:7.4f}  Std={s['avg_std']:7.4f}  Range={s['range']:7.4f}")
    
    # PSD Inputs
    print("\n### PSD INPUTS ###")
    for name in ['psd_smooth_1', 'psd_sharp_1', 'psd_smooth_2', 'psd_sharp_2']:
        s = aggregated_stats['psd_inputs'][name]
        print(f"{name:15s}: [{s['global_min']:8.4f}, {s['global_max']:8.4f}]  "
              f"Mean={s['avg_mean']:7.4f}  Std={s['avg_std']:7.4f}  Range={s['range']:7.4f}")
    
    # Spline Outputs
    print("\n### SPLINE OUTPUTS (Model) ###")
    for name in ['smooth_knots', 'smooth_control', 'sharp_knots', 'sharp_control']:
        s = aggregated_stats['spline_outputs'][name]
        print(f"{name:15s}: [{s['global_min']:8.4f}, {s['global_max']:8.4f}]  "
              f"Mean={s['avg_mean']:7.4f}  Std={s['avg_std']:7.4f}  Range={s['range']:7.4f}")
    
    # Kernels
    print("\n### KERNELS (OTF) ###")
    for name in ['otf_smooth_to_sharp', 'otf_sharp_to_smooth']:
        s = aggregated_stats['kernels'][name]
        print(f"{name:20s}: [{s['global_min']:8.4f}, {s['global_max']:8.4f}]  "
              f"Mean={s['avg_mean']:7.4f}  Std={s['avg_std']:7.4f}  Range={s['range']:7.4f}")
    
    # Generated Images
    print("\n### GENERATED IMAGES ###")
    for name in ['I_generated_smooth', 'I_generated_sharp']:
        s = aggregated_stats['generated_images'][name]
        print(f"{name:20s}: [{s['global_min']:8.4f}, {s['global_max']:8.4f}]  "
              f"Mean={s['avg_mean']:7.4f}  Std={s['avg_std']:7.4f}  Range={s['range']:7.4f}")
    
    # Compare targets vs generated
    print("\n### TARGET vs GENERATED COMPARISON ###")
    smooth_target = aggregated_stats['input_images']['I_smooth_2']
    smooth_gen = aggregated_stats['generated_images']['I_generated_smooth']
    sharp_target = aggregated_stats['input_images']['I_sharp_1']
    sharp_gen = aggregated_stats['generated_images']['I_generated_sharp']
    
    print(f"Smooth - Target: [{smooth_target['global_min']:.4f}, {smooth_target['global_max']:.4f}]  "
          f"Mean={smooth_target['avg_mean']:.4f}")
    print(f"Smooth - Generated: [{smooth_gen['global_min']:.4f}, {smooth_gen['global_max']:.4f}]  "
          f"Mean={smooth_gen['avg_mean']:.4f}")
    print(f"Smooth - Range Mismatch: {abs(smooth_target['range'] - smooth_gen['range']):.4f}")
    print()
    print(f"Sharp - Target: [{sharp_target['global_min']:.4f}, {sharp_target['global_max']:.4f}]  "
          f"Mean={sharp_target['avg_mean']:.4f}")
    print(f"Sharp - Generated: [{sharp_gen['global_min']:.4f}, {sharp_gen['global_max']:.4f}]  "
          f"Mean={sharp_gen['avg_mean']:.4f}")
    print(f"Sharp - Range Mismatch: {abs(sharp_target['range'] - sharp_gen['range']):.4f}")
    
    # MTF Analysis
    print("\n### MTF DATA ###")
    for name in ['input_profiles', 'target_mtfs', 'predicted_mtfs']:
        s = aggregated_stats['mtf'][name]
        print(f"{name:20s}: [{s['global_min']:8.4f}, {s['global_max']:8.4f}]  "
              f"Mean={s['avg_mean']:7.4f}  Std={s['avg_std']:7.4f}  Range={s['range']:7.4f}")
    
    # Flag potential issues
    print("\n" + "="*80)
    print("POTENTIAL ISSUES")
    print("="*80)
    
    issues = []
    
    # Check for range mismatches
    if abs(smooth_target['range'] - smooth_gen['range']) > 0.1:
        issues.append("⚠️ SMOOTH: Large range mismatch between target and generated")
    
    if abs(sharp_target['range'] - sharp_gen['range']) > 0.1:
        issues.append("⚠️ SHARP: Large range mismatch between target and generated")
    
    # Check for collapsed outputs
    if smooth_gen['range'] < 0.01:
        issues.append("🔴 CRITICAL: Generated smooth images have collapsed range (nearly constant)")
    
    if sharp_gen['range'] < 0.01:
        issues.append("🔴 CRITICAL: Generated sharp images have collapsed range (nearly constant)")
    
    # Check for extreme values
    for name, s in aggregated_stats['generated_images'].items():
        if s['global_min'] < -1.0 or s['global_max'] > 2.0:
            issues.append(f"⚠️ {name} has extreme values outside expected range")
    
    # Check kernel ranges
    for name, s in aggregated_stats['kernels'].items():
        if s['global_min'] < 0:
            issues.append(f"⚠️ {name} has negative values (unexpected for OTF)")
    
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("✓ No obvious issues detected")
    
    print("="*80)


def main():
    # Configuration
    CHECKPOINT_PATH = r"D:\Charan work file\PhantomTesting\Code\training_output_0.5\checkpoints\best_checkpoint.pth"
    IMAGE_ROOT_DIR = r"D:\Charan work file\KernelEstimator\Data_Root"
    MTF_FOLDER = r"D:\Charan work file\PhantomTesting\MTF_Results_Output"
    PSD_FOLDER = r"D:\Charan work file\PhantomTesting\PSD_Results_Output"
    OUTPUT_DIR = Path("data_analysis")
    BATCH_SIZE = 16
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*80)
    print("DATA RANGE ANALYSIS TOOL")
    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print("="*80)
    
    # Load datasets
    print("\nLoading datasets...")
    image_dataset = PSDDataset(root_dir=IMAGE_ROOT_DIR, preload=True)
    mtf_dataset = MTFPSDDataset(MTF_FOLDER, PSD_FOLDER, verbose=False)
    
    image_loader = DataLoader(
        image_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    mtf_loader = DataLoader(
        mtf_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Load model
    print("Loading model...")
    model = KernelEstimator().to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}\n")
    
    # Analyze data
    stats = analyze_data_ranges(model, image_loader, mtf_loader, device)
    aggregated = compute_aggregate_stats(stats)
    
    # Print analysis
    print_analysis(aggregated)
    
    # Save to JSON
    output_file = OUTPUT_DIR / "data_range_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()