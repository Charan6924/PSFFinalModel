import json
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import nibabel as nib
import os
import numpy as np

output_dir = r"D:\Charan work file\PhantomTesting\test_reconstructions"
os.makedirs(output_dir, exist_ok=True)

I_smooth = nib.load(r"D:\Charan work file\PhantomTesting\testA\M0RGZYGQ6366_filter_CB.nii")  # type: ignore
I_sharp  = nib.load(r"D:\Charan work file\PhantomTesting\testB\M0RGZYGQ6366_filter_YA.nii")  # type: ignore
device = 'cuda'


def compute_psd(image):
    """Returns log-normalized PSD tensor [B, 1, H, W] (real-valued)."""
    with torch.no_grad():
        batch_psd = []
        for b in range(image.shape[0]):
            slice_data = image[b, 0, :, :]
            slice_ft   = torch.fft.fftshift(torch.fft.fft2(slice_data))
            slice_psd  = torch.abs(slice_ft) ** 2
            slice_psd  = torch.log(slice_psd + 1)
            psd_min    = slice_psd.min()
            psd_max    = slice_psd.max()
            slice_psd  = (slice_psd - psd_min) / (psd_max - psd_min + 1e-10)
            batch_psd.append(slice_psd)
        psd = torch.stack(batch_psd, dim=0).unsqueeze(1)  # [B, 1, H, W]
    return psd  # FIX 1: was missing return


def compute_fft(image):
    """Returns shifted FFT [B, H, W] (complex-valued)."""
    with torch.no_grad():
        image = image.to(device)
        if image.dim() == 4:
            image = image[:, 0, :, :]          # [B, H, W]
        fft_shifted = torch.fft.fftshift(torch.fft.fft2(image))
    return fft_shifted                          # [B, H, W] complex


def symmetric_average_2d(psd_2d):
    """Symmetrize a [B, 1, H, W] PSD along the frequency axis."""
    B, C, H, W = psd_2d.shape
    center = W // 2

    left  = psd_2d[:, :, :, :center].flip(dims=[-1])  # [B, 1, H, center]
    right = psd_2d[:, :, :, center + 1:]               # [B, 1, H, W-center-1]

    min_len = min(left.shape[-1], right.shape[-1])
    left    = left[:,  :, :, :min_len]
    right   = right[:, :, :, :min_len]

    averaged = (left + right) / 2.0                    # [B, 1, H, min_len]
    dc       = psd_2d[:, :, :, center].unsqueeze(-1)   # [B, 1, H, 1]

    smoothed = torch.cat([averaged.flip(dims=[-1]), dc, averaged], dim=-1)
    return smoothed  # [B, 1, H, 2*min_len+1]


def gaussian_blur_2d(x, kernel_size=21, sigma=5.0):
    pad    = kernel_size // 2
    coords = torch.arange(kernel_size, dtype=torch.float32) - pad
    g      = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g      = g / g.sum()
    kernel = g.outer(g).unsqueeze(0).unsqueeze(0).to(x.device)
    kernel = kernel.expand(x.shape[1], 1, -1, -1)
    return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode='reflect'),
                    kernel, groups=x.shape[1])


num_slices   = I_smooth.shape[2]
smooth_recon = np.zeros(I_smooth.shape[:3], dtype=np.float32)
sharp_recon  = np.zeros(I_sharp.shape[:3],  dtype=np.float32)

for k in range(num_slices):
    s_slice = I_smooth.dataobj[:, :, k].copy()
    h_slice = I_sharp.dataobj[:,  :, k].copy()

    s_slice = np.clip(s_slice, -1000, 3000)
    h_slice = np.clip(h_slice, -1000, 3000)

    s_slice_norm = (s_slice + 1000) / 4000
    h_slice_norm = (h_slice + 1000) / 4000

    I_smooth_tensor = torch.from_numpy(s_slice_norm).float().unsqueeze(0).unsqueeze(0).to(device)
    I_sharp_tensor  = torch.from_numpy(h_slice_norm).float().unsqueeze(0).unsqueeze(0).to(device)

    # FIX 2: compute PSD (for filter estimation) AND raw FFT (for reconstruction) separately
    I_smooth_psd = compute_psd(I_smooth_tensor)  # [1, 1, H, W] real
    I_sharp_psd  = compute_psd(I_sharp_tensor)   # [1, 1, H, W] real

    I_smooth_fft = compute_fft(I_smooth_tensor)  # [1, H, W] complex
    I_sharp_fft  = compute_fft(I_sharp_tensor)   # [1, H, W] complex

    # FIX 3: derive filters from PSDs (real), shape [1, 1, H, W]
    smooth_to_sharp_filter = I_sharp_fft  / (I_smooth_fft + 1)
    sharp_to_smooth_filter = I_smooth_fft / (I_sharp_fft  + 1)

    # FIX 4: squeeze filter to [1, H, W] so it broadcasts with complex FFT [1, H, W]
    smooth_to_sharp_filter = smooth_to_sharp_filter.squeeze(1)  # [1, H, W]
    sharp_to_smooth_filter = sharp_to_smooth_filter.squeeze(1)  # [1, H, W]

    print(f"Slice {k}: filter shape {smooth_to_sharp_filter.shape}, "
          f"FFT shape {I_smooth_fft.shape}")

    # FIX 5: apply filter in frequency domain using the raw complex FFT
    sharp_fft_filtered  = I_smooth_fft * smooth_to_sharp_filter  # smooth→sharp
    smooth_fft_filtered = I_sharp_fft  * sharp_to_smooth_filter  # sharp→smooth

    # Reconstruct spatial domain
    smooth_recon_slice = torch.fft.ifft2(torch.fft.ifftshift(smooth_fft_filtered))
    sharp_recon_slice  = torch.fft.ifft2(torch.fft.ifftshift(sharp_fft_filtered))

    # Take real part (tiny imaginary residual expected from float arithmetic)
    smooth_recon_slice = smooth_recon_slice.real.squeeze().cpu().numpy() * 4000 - 1000
    sharp_recon_slice  = sharp_recon_slice.real.squeeze().cpu().numpy()  * 4000 - 1000

    smooth_recon[:, :, k] = smooth_recon_slice
    sharp_recon[:,  :, k] = sharp_recon_slice

smooth_out = nib.Nifti1Image(smooth_recon, affine=I_smooth.affine, header=I_smooth.header)
sharp_out  = nib.Nifti1Image(sharp_recon,  affine=I_sharp.affine,  header=I_sharp.header)

nib.save(smooth_out, os.path.join(output_dir, "smooth_reconstructed.nii"))
nib.save(sharp_out,  os.path.join(output_dir, "sharp_reconstructed.nii"))

print("Saved reconstructed volumes.")