# Spline-Based Kernel Estimation for Image Transformation

## Overview

This project implements a deep learning approach to estimate and apply optical transfer functions (OTFs) for transforming images between sharp and smooth (blurred) representations. The system uses spline-based kernel estimation to model the frequency-domain characteristics that distinguish sharp from smooth images.

## Key Components

### 1. **KernelEstimator Model**
A neural network that analyzes the Power Spectral Density (PSD) of images and outputs spline parameters (knots and control points) that characterize the image's frequency response.

### 2. **Spline-to-Kernel Conversion**
Converts learned spline parameters into 2D Optical Transfer Functions (OTFs) that can be applied in the frequency domain to transform images:
- **OTF smooth→sharp**: Sharpens blurry images
- **OTF sharp→smooth**: Smooths/blurs sharp images

### 3. **Frequency Domain Processing**
Uses Fast Fourier Transform (FFT) operations to apply transformations:
```
Sharp Image = IFFT(FFT(Smooth Image) × OTF_smooth_to_sharp)
Smooth Image = IFFT(FFT(Sharp Image) × OTF_sharp_to_smooth)
```

## Workflow

1. **Input**: Paired smooth and sharp images from medical imaging dataset
2. **PSD Computation**: Calculate power spectral density for each image
3. **Spline Estimation**: Model predicts spline parameters characterizing each image's frequency content
4. **OTF Generation**: Convert splines to radially-symmetric 2D transfer functions
5. **Image Transformation**: Apply OTFs via FFT multiplication and inverse FFT
6. **Visualization**: Compare original and generated images

## Technical Details

### Frequency Domain Operations
- **FFT (Forward)**: Converts spatial domain → frequency domain (preserves magnitude + phase)
- **IFFT (Inverse)**: Converts frequency domain → spatial domain
- **OTF Application**: Multiplicative in frequency domain = convolution in spatial domain

### Key Insight
The project uses **complex FFT** (not PSD) for image transformation to preserve phase information, which is critical for accurate reconstruction.

## File Structure

- `SplineEstimator.py`: Neural network model definition
- `PSDDataset.py`: Dataset loader for image pairs
- `utils.py`: Helper functions (spline generation, PSD computation)
- `plot_images_fixed.py`: Main script for inference and visualization

## Usage

```python
# Load trained model
model = KernelEstimator().to(device)
checkpoint = torch.load('best_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Process images
fft_smooth = torch.fft.fft2(smooth_image)
fft_sharp = torch.fft.fft2(sharp_image)

# Apply transformations
sharp_generated = torch.fft.ifft2(fft_smooth * otf_smooth_to_sharp)
smooth_generated = torch.fft.ifft2(fft_sharp * otf_sharp_to_smooth)
```

## Applications

- Medical image enhancement
- Image quality assessment
- Deblurring and restoration
- Understanding imaging system characteristics

## Requirements

- PyTorch
- NumPy
- Matplotlib
- CUDA-capable GPU (recommended)
