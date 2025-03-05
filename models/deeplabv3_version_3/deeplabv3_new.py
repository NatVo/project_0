import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module.
    """
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.atrous_conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.atrous_conv2 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.atrous_conv3 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18
        )
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1x1_pool = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn5 = nn.BatchNorm2d(out_channels)

        self.conv1x1_out = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)
        self.bn_out = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        size = x.shape[2:]

        # 1x1 Convolution
        conv1x1_1 = self.relu(self.bn1(self.conv1x1_1(x)))

        # Atrous convolutions with different dilation rates
        atrous1 = self.relu(self.bn2(self.atrous_conv1(x)))
        atrous2 = self.relu(self.bn3(self.atrous_conv2(x)))
        atrous3 = self.relu(self.bn4(self.atrous_conv3(x)))

        # Global average pooling
        global_avg = self.global_avg_pool(x)
        global_avg = self.relu(self.bn5(self.conv1x1_pool(global_avg)))
        global_avg = F.interpolate(global_avg, size=size, mode='bilinear', align_corners=False)

        # Concatenate all ASPP branches
        x = torch.cat([conv1x1_1, atrous1, atrous2, atrous3, global_avg], dim=1)
        x = self.relu(self.bn_out(self.conv1x1_out(x)))
        return x

class DeepLabDecoder(nn.Module):
    """
    Decoder module to refine the output and produce segmentation maps.
    """
    def __init__(self, low_level_channels, num_classes):
        super(DeepLabDecoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_channels, 48, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(48 + 256, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv_out = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, low_level_features, high_level_features):
        size = low_level_features.shape[2:]

        # Process low-level features
        low_level_features = self.relu(self.bn1(self.conv1(low_level_features)))

        # Upsample high-level features
        high_level_features = F.interpolate(
            high_level_features, size=size, mode='bilinear', align_corners=False
        )

        # Concatenate low-level and high-level features
        x = torch.cat([low_level_features, high_level_features], dim=1)

        # Further convolution
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        # Output segmentation map
        x = self.conv_out(x)
        return x

class DeepLabV3(nn.Module):
    """
    End-to-end DeepLabV3 model with ResNet-101 backbone, ASPP (encoder), and decoder.
    """
    def __init__(self, num_classes, output_stride=16):
        super(DeepLabV3, self).__init__()

        # Output stride
        if output_stride == 16:
            replace_stride_with_dilation = [False, True, True]  # Dilation in last two blocks
        elif output_stride == 8:
            replace_stride_with_dilation = [True, True, True]  # Dilation in all three blocks
        else:
            raise ValueError("Output stride must be 8 or 16")

        # Load ResNet-101 backbone
        resnet = resnet101(pretrained=True, replace_stride_with_dilation=replace_stride_with_dilation)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        # Extract low-level features from layer1
        self.low_level_features = resnet.layer1

        # ASPP module (Encoder)
        in_channels = 2048  # ResNet-101 final output channels
        self.aspp = ASPP(in_channels, 256)

        # Decoder
        self.decoder = DeepLabDecoder(low_level_channels=256, num_classes=num_classes)

    def forward(self, x):
        # Backbone
        size = x.shape[2:]
        x = self.backbone[:4](x)  # Initial conv1, bn1, relu, maxpool layers
        low_level_features = self.low_level_features(x)  # Low-level features from layer1
        x = self.backbone[4:](x)  # Remaining layers (layer2, layer3, and layer4)

        # ASPP module (Encoder)
        high_level_features = self.aspp(x)

        # Decoder
        x = self.decoder(low_level_features, high_level_features)

        # Upsample to original image size
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return x

# Instantiate the model
if __name__ == "__main__":
    model = DeepLabV3(num_classes=21, output_stride=16)  # Example for 21 classes, e.g., in Pascal VOC
    print(model)

    # Test the model with a dummy input
    dummy_input = torch.randn(1, 3, 512, 512)  # Batch size 1, 3-channel, 512x512 input
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Should match (1, num_classes, 512, 512)