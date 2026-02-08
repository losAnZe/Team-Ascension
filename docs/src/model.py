"""
Lightweight CNN model for wafer defect classification.
Uses a simplified architecture optimized for edge deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution - much more efficient than standard conv."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=kernel_size, stride=stride, padding=padding,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class WaferDefectCNN(nn.Module):
    """
    Lightweight CNN for wafer defect classification.
    Designed for edge deployment with ~500K parameters.
    
    Architecture:
        - Initial Conv block
        - 4 Depthwise Separable Conv blocks with pooling
        - Global Average Pooling
        - Fully Connected classifier
    """
    
    def __init__(self, num_classes=8, in_channels=1, dropout=0.3):
        super().__init__()
        
        # Initial convolution (grayscale input)
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Depthwise separable blocks with increasing channels
        self.block1 = nn.Sequential(
            DepthwiseSeparableConv(32, 64),
            nn.MaxPool2d(2, 2)  # 64x64 -> 32x32
        )
        
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(64, 128),
            nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        )
        
        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(128, 256),
            nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        )
        
        self.block4 = nn.Sequential(
            DepthwiseSeparableConv(256, 512),
            nn.MaxPool2d(2, 2)  # 8x8 -> 4x4
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.initial(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def get_features(self, x):
        """Extract features before classifier (for visualization)."""
        x = self.initial(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return x


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(num_classes=8, pretrained=False):
    """Create and return the model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights (not implemented)
    
    Returns:
        WaferDefectCNN model
    """
    model = WaferDefectCNN(num_classes=num_classes)
    
    # Initialize weights
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    return model


if __name__ == "__main__":
    # Test model
    print("Testing WaferDefectCNN model...")
    
    model = create_model(num_classes=8)
    
    # Count parameters
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
    
    # Test forward pass
    x = torch.randn(4, 1, 64, 64)  # Batch of 4 grayscale 64x64 images
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Estimate model size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = (param_size + buffer_size) / 1024 / 1024
    print(f"Estimated model size: {total_size:.2f} MB")
