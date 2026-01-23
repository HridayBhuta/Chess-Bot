import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ChessStyleBot(nn.Module):
    def __init__(self, num_res_blocks=10):
        super().__init__()
        self.start_conv = nn.Conv2d(13, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
            
        self.res_blocks = nn.ModuleList([ResBlock(128) for _ in range(num_res_blocks)])
        self.flatten_dim = 128 * 8 * 8 
            
        self.policy_head = nn.Linear(self.flatten_dim, 4096)
        self.value_head = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.start_conv(x)))
        for block in self.res_blocks:
            x = block(x)
            
        x = x.view(-1, self.flatten_dim)
            
        policy = self.policy_head(x)
        value = self.value_head(x)
            
        return policy, value