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
        internal_knots = internal_knots / internal_knots[:, -1].unsqueeze(1) 

        zeros = torch.zeros(batch_size, self.degree + 1, device=device)
        ones = torch.ones(batch_size, self.degree + 1, device=device)

        valid_internal = internal_knots[:, 2:3] 
        
        full_knots = torch.cat([zeros, valid_internal, ones], dim=1)

        return full_knots

class ResBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm1d(channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout1d(0.1)
    
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
        # Input: PSD [Batch, 1, 363]
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1),    
            nn.InstanceNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(0.1),
            
            ResBlock1D(32),
            
            nn.Conv1d(32, 64, 3, padding=1),   
            nn.InstanceNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(0.1),
            
            ResBlock1D(64),
            
            nn.Conv1d(64, 128, 3, padding=1),  
            nn.InstanceNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(0.1),
            
            ResBlock1D(128),
            
            nn.Conv1d(128, 64, 3, padding=1),  
            nn.InstanceNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(0.1),
            
            ResBlock1D(64),
            
            nn.Conv1d(64, 32, 3, padding=1),   
            nn.InstanceNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(0.1),
            
            ResBlock1D(32)
        )
        
        self.flatten = nn.Flatten() 
        self.fc_head = nn.Linear(32 * 363, 10)  #10 points
        self.knot_layer = FixedSplineLayer(degree=3)

    def forward(self, psd):
        x = self.features(psd)
        x = self.flatten(x)
        raw_out = self.fc_head(x)
        
        raw_control = raw_out[:, :5]
        raw_knots = raw_out[:, 5:]
        
        control = F.softplus(raw_control, beta=2.0)
        control = control / (control.max(dim=1, keepdim=True).values + 1e-6)
        control[:, 0] = 1.0 
        
        full_knots = self.knot_layer(raw_knots, control.shape[0], control.device)
        return full_knots, control