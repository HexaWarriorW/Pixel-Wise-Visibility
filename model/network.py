import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from model.basic import *

class VitResNet50(nn.Module):
    def __init__(self, cross_num):
        super(VitResNet50, self).__init__()
        rgb_stream = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.rgb_stream_backbone = nn.Sequential(*[
            rgb_stream.conv1,
            rgb_stream.bn1,
            rgb_stream.relu,
            rgb_stream.maxpool,
            rgb_stream.layer1,
            rgb_stream.layer2,
            rgb_stream.layer3]
        )
        
        depth_stream = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.depth_stream = nn.Sequential(*[
            depth_stream.conv1,
            depth_stream.bn1,
            depth_stream.relu,
            depth_stream.maxpool,
            depth_stream.layer1,
            depth_stream.layer2,
            depth_stream.layer3]
        )
        self.depth_stream[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv_rgb = convbnrelu(1024, 256, kernel_size=1, stride=1, padding=0)
        self.conv_depth = convbnrelu(256, 256, kernel_size=1, stride=1, padding=0)
        
        self.fusion_transformer = nn.ModuleList(
            [
                CrossFusionBlock(dim=256, num_heads=8) for _ in range(cross_num)
            ]
        )
        self.pos_encoding = SinusoidalPositionalEncoding2D(d_model=256, max_h=64, max_w=64)
        self.depth_proj1 = convbnrelu(64, 256, kernel_size=3, stride=1, padding=1)
        self.depth_proj2 = convbnrelu(128, 256, kernel_size=3, stride=1, padding=1)
        self.fusion1_block = FusionBlock(dim=256)
        self.fusion2_block = FusionBlock(dim=256)
        self.fusion3_block = FusionBlock(dim=256)
        self.adapt_fusion_dim = nn.Sequential(
            *convbnrelu(256, 256, kernel_size=3, stride=1, padding=1),
            *convbnrelu(256, 256, kernel_size=3, stride=1, padding=1),
            *convbnrelu(256, 128, kernel_size=3, stride=1, padding=1),
            *convbnrelu(128, 128, kernel_size=3, stride=1, padding=1),
            *convbnrelu(128, 64, kernel_size=3, stride=1, padding=1),
        )
        self.head_t = nn.Sequential(
            convbnrelu(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.head_beta = nn.Sequential(
            convbnrelu(65, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.ReLU()  # enforce β ≥ 0
        )

        self.patch_fog_detector = PatchyFogDetection()

    def forward(self, x_rgb, x_depth):
        # RGB Stream
        x_rgb = self.rgb_stream_backbone[0](x_rgb)
        x_rgb = self.rgb_stream_backbone[1](x_rgb)
        x_rgb = self.rgb_stream_backbone[2](x_rgb)
        x_rgb = self.rgb_stream_backbone[3](x_rgb)

        x_rgb = self.rgb_stream_backbone[4](x_rgb)
        x_rgb = self.rgb_stream_backbone[5](x_rgb)
        feat_rgb = self.rgb_stream_backbone[6](x_rgb)
        feat_rgb = self.conv_rgb(feat_rgb)
        feat_rgb = self.pos_encoding(feat_rgb)  # (B, 256, H, W)
        b, c, h, w = feat_rgb.shape
        feat_rgb = feat_rgb.view(b, c, h*w).permute(0, 2, 1)

        # Depth Stream
        x_depth = self.depth_stream[0](x_depth)      
        x_depth = self.depth_stream[1](x_depth)
        x_depth = self.depth_stream[2](x_depth)
        x_depth = self.depth_stream[3](x_depth)    
        x_depth1 = self.depth_stream[4](x_depth)     
        x_depth2 = self.depth_stream[5](x_depth1)    
        feat_depth = self.depth_stream[6](x_depth2)
        feat_depth = self.conv_depth(feat_depth)
        feat_depth = self.pos_encoding(feat_depth)  # (B, 256, H, W)
        b, c, h, w = feat_depth.shape
        feat_depth = feat_depth.view(b, c, h*w).permute(0, 2, 1)
        
        # Transformer Fusion
        for fusion_layer in self.fusion_transformer:
            feat_rgb = fusion_layer(feat_rgb, feat_depth)
            feat_depth = fusion_layer(feat_depth, feat_rgb)
        
        feat_rgb = feat_rgb.permute(0, 2, 1).view(b, c, h, w)
        feat_depth = feat_depth.permute(0, 2, 1).view(b, c, h, w)

        # Feature Projection and Fusion
        x_depth1 = self.depth_proj1(x_depth1)
        x_depth2 = self.depth_proj2(x_depth2)
        fusion1 = self.fusion1_block(feat_rgb, feat_depth)
        fusion2 = self.fusion2_block(fusion1, x_depth2)
        fusion3 = self.fusion3_block(fusion2, x_depth1)
        fusion3 = self.adapt_fusion_dim(fusion3)
        t = self.head_t(fusion3)
        fusion3 = torch.cat([fusion3, t], dim=1)  # Concatenate t for beta prediction
        beta = self.head_beta(fusion3)
        t = F.interpolate(t, scale_factor=2.0, mode='bilinear', align_corners=True)
        beta = F.interpolate(beta, scale_factor=2.0, mode='bilinear', align_corners=True)
        patch_fog = self.patch_fog_detector(t, beta)
        return t, beta, patch_fog

class PatchyFogDetection(nn.Module):
    def __init__(self):
        super(PatchyFogDetection, self).__init__()
        self.features = nn.Sequential(
            convbnrelu(2, 16, 5, 2, 2),
            nn.MaxPool2d(kernel_size=4, stride=4),  # 640 -> 80

            convbnrelu(16, 32, 3, 1, 1),
            nn.MaxPool2d(kernel_size=4, stride=4),  # 80 -> 20

            convbnrelu(32, 64, 3, 1, 1),
            nn.MaxPool2d(kernel_size=4, stride=4),  # 20 -> 5
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x, y):
        x = self.features(torch.cat([x, y], dim=1))
        x = self.classifier(x)
        return x