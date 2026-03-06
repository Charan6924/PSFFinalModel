from SplineEstimator import KernelEstimator
from utils import spline_to_kernel, get_torch_spline, generate_images, compute_fft, compute_psd
from PSDDataset import PSDDataset
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

device = 'cuda'
model = KernelEstimator()
checkpoint = torch.load("/home/cxv166/PhantomTesting/Code/training_output_0.5/checkpoints/best_checkpoint.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval() 
checkpoint = torch
dataset = PSDDataset(root_dir=r"/home/cxv166/PhantomTesting/Data_Root")
loader = DataLoader(dataset=dataset,batch_size=32)
I_smooth,I_sharp, _,_ = next(iter(loader))
psd_smooth = compute_psd(I_smooth, device='cuda').to(device, non_blocking=True)
psd_sharp  = compute_psd(I_sharp,  device='cuda').to(device, non_blocking=True)
I_smooth_fft = compute_fft(I_smooth)
I_sharp_fft = compute_fft(I_sharp)

smooth_k, smooth_c = model(psd_smooth)
sharp_k,sharp_c = model(psd_sharp)

otf_smooth,otf_sharp = spline_to_kernel(smooth_knots=smooth_k,smooth_control_points=smooth_c,sharp_control_points=sharp_c,sharp_knots=sharp_k)

smooth2sharp = otf_smooth/(otf_sharp + 1e-10)
real = I_sharp_fft/(I_smooth_fft + 1e-10)

plt.plot(real[0,255,:].detach().to('cpu'))
plt.ylim(0,10)
plt.savefig('comparison3')


