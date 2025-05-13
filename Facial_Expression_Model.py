import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

# ----------------------
# Convolutional Block Attention Module (CBAM)
# ----------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# ----------------------
# ResNet18 with CBAM
# ----------------------
class ResNet18_CBAM(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18_CBAM, self).__init__()
        self.backbone = resnet18(pretrained=pretrained)
        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.cbam1(x)
        x = self.backbone.layer2(x)
        x = self.cbam2(x)
        x = self.backbone.layer3(x)
        x = self.cbam3(x)
        x = self.backbone.layer4(x)
        x = self.cbam4(x)

        return x

# ----------------------
# Barlow Twins SSL Module
# ----------------------
class BarlowTwinsHead(nn.Module):
    def __init__(self, in_dim, out_dim=2048):
        super(BarlowTwinsHead, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, out_dim)
        )

    def forward(self, x):
        return self.projector(x)

class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambd=5e-3):
        super(BarlowTwinsLoss, self).__init__()
        self.lambd = lambd

    def forward(self, z1, z2):
        N, D = z1.size()
        z1_norm = (z1 - z1.mean(0)) / z1.std(0)
        z2_norm = (z2 - z2.mean(0)) / z2.std(0)
        c = torch.matmul(z1_norm.T, z2_norm) / N
        on_diag = torch.diagonal(c).add_(-1).pow(2).sum()
        off_diag = self.off_diagonal(c).pow(2).sum()
        return on_diag + self.lambd * off_diag

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

# ----------------------
# Diffusion-based CARD model block (simplified form)
# ----------------------
class SimpleDiffusionDecoder(nn.Module):
    def __init__(self, feature_dim, out_dim):
        super(SimpleDiffusionDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        return self.decoder(x)

# ----------------------
# Combined Model
# ----------------------
class FullExpressionModel(nn.Module):
    def __init__(self, num_classes):
        super(FullExpressionModel, self).__init__()
        self.feature_extractor = ResNet18_CBAM(pretrained=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.barlow_projector = BarlowTwinsHead(in_dim=512)
        self.diffusion_decoder = SimpleDiffusionDecoder(2048, num_classes)

    def forward(self, x1, x2=None, mode="train"):
        x1 = self.feature_extractor(x1)
        x1 = self.pool(x1).view(x1.size(0), -1)

        if mode == "train" and x2 is not None:
            x2 = self.feature_extractor(x2)
            x2 = self.pool(x2).view(x2.size(0), -1)
            z1 = self.barlow_projector(x1)
            z2 = self.barlow_projector(x2)
            return z1, z2
        else:
            z1 = self.barlow_projector(x1)
            logits = self.diffusion_decoder(z1)
            return logits

