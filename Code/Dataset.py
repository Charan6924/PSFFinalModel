import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MTFDataset(Dataset):
    def __init__(self, npz_path):
        # Load the compressed data
        loaded = np.load(npz_path, allow_pickle=True)
        # Access the 'data' array we saved earlier
        self.samples = loaded['data']
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load tensors
        input_profile = torch.tensor(sample['input_profile'], dtype=torch.float32)
        label_mtf = torch.tensor(sample['label_mtf'], dtype=torch.float32)
        
        # --- FIX: Handle NaNs and Infs ---
        # Replace NaN with 0 and Pos/Neg Inf with large/small finite numbers
        input_profile = torch.nan_to_num(input_profile, nan=0.0)
        label_mtf = torch.nan_to_num(label_mtf, nan=0.0)
        
        # Optional: Clip MTF to physical range [0, 1.2] 
        # (Since you saw values like 1.104 earlier)
        label_mtf = torch.clamp(label_mtf, 0.0, 1.2)
        
        kernel = sample['kernel']
        return input_profile, label_mtf
# ---------------------------------------------------------
# 2. Testing Code
# ---------------------------------------------------------
if __name__ == "__main__":
    print('loading data')
    dataset_path = r"D:\Charan work file\PhantomTesting\training_dataset.npz"
    
    try:
        # Initialize Dataset
        ds = MTFDataset(dataset_path)
        print(f"Successfully loaded dataset with {len(ds)} samples.")
        
        # Use a DataLoader to test batching
        batch_size = 4
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        
        # Pull one batch
        profiles, mtfs = next(iter(loader))
        
        print("-" * 30)
        print("BATCH SHAPES:")
        print(f"Input Profile Shape: {profiles.shape}") # Expect: [batch, length]
        print(f"Target MTF Shape:    {mtfs.shape}")    # Expect: [batch, length]
        print("-" * 30)
        
        # Check a single sample's range
        print("SAMPLE VALUES CHECK:")
        print(f"Max MTF Value: {torch.max(mtfs):.4f}")
        print(f"Min MTF Value: {torch.min(mtfs):.4f}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")