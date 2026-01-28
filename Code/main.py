import torch
from torch.utils.data import DataLoader, random_split
from Dataset import MTFDataset

# Load full MTF dataset
mtf_dataset = MTFDataset(r"D:\Charan work file\PhantomTesting\training_dataset.npz")

# Split into train (80%) and val (20%)
train_size = int(0.8 * len(mtf_dataset))
val_size = len(mtf_dataset) - train_size

mtf_train_dataset, mtf_val_dataset = random_split(
    mtf_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # For reproducibility
)

# Create dataloaders
mtf_train_loader = DataLoader(mtf_train_dataset, batch_size=16, shuffle=True)
mtf_val_loader = DataLoader(mtf_val_dataset, batch_size=16, shuffle=False)

print(f"MTF Training samples: {len(mtf_train_dataset)}")
print(f"MTF Validation samples: {len(mtf_val_dataset)}")