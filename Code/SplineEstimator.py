import torch
import torch.nn as nn
import torch.nn.functional as F

class FixedSplineLayer(nn.Module):
    def __init__(self, degree=3):
        super().__init__()
        self.degree = degree

    def forward(self, knot_params, batch_size, device):
        deltas = F.softplus(knot_params) + 1e-3
        internal_knots = torch.cumsum(deltas, dim=1)
        internal_knots = internal_knots / (internal_knots[:, -1].unsqueeze(1) + 1e-6)
        internal_knots = internal_knots * 0.90  
        
        padding = self.degree + 1
        zeros = torch.zeros(batch_size, padding, device=device)
        ones = torch.ones(batch_size, padding, device=device)
        full_knots = torch.cat([zeros, internal_knots, ones], dim=1)

        return full_knots

class ResBlock2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class KernelEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: [Batch, 1, 512, 512]
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),    # 256x256
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            ResBlock2D(32),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # 128x128
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            ResBlock2D(64),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 64x64
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            ResBlock2D(128),
            nn.AdaptiveAvgPool2d((8, 8)) 
        )
        
        self.flatten = nn.Flatten() 
        self.fc_head = nn.Linear(128 * 8 * 8, 16)  
        self.knot_layer = FixedSplineLayer(degree=3)
        self.control_scale = nn.Parameter(torch.tensor(1.5))

    def forward(self, psd):
        x = self.features(psd)
        x = self.flatten(x)
        raw_out = self.fc_head(x)
        
        raw_control = raw_out[:, :10]
        raw_knots = raw_out[:, 10:]
        control = F.softplus(raw_control, beta=1.0) * self.control_scale
        
        full_knots = self.knot_layer(raw_knots, control.shape[0], control.device)
        return full_knots, control