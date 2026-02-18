import nibabel as nib
import torch

original = nib.load(r"D:\Charan work file\PhantomTesting\testA\M0RGZYGQ6366_filter_CB.nii")
reconstructed = nib.load(r"D:\Charan work file\PhantomTesting\reconstructions\M0RGZYGQ6366_YA_to_CB.nii.gz")

original_data = original.get_fdata()
reconstructed_data = reconstructed.get_fdata()

original_slice = torch.from_numpy(original_data[:, :, 1])
reconstructed_slice = torch.from_numpy(reconstructed_data[:, :, 1])

original_slice_fft = torch.fft.fftshift(torch.fft.fft2(original_slice))
reconstructed_slice_fft = torch.fft.fftshift(torch.fft.fft2(reconstructed_slice))

print(abs(original_slice_fft.real - reconstructed_slice_fft.real).mean().item())
print(abs(original_slice_fft.imag - reconstructed_slice_fft.imag).mean().item())