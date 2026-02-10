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
        
        # Policy head (matches saved model weights)
        self.policy_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2 * 8 * 8, 4096)

    def forward(self, x):
        x = F.relu(self.bn1(self.start_conv(x)))
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 2 * 8 * 8)
        policy = self.fc(policy)
            
        return policy