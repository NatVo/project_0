import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet

# Custom ResNet101 Backbone
class ResNetBackbone(nn.Module):
    def __init__(self, resnet_model):
        super(ResNetBackbone, self).__init__()
        # Extract layers up to the fourth block
        resnet_model.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(resnet_model.conv1.weight, mode='fan_out', nonlinearity='relu')

        self.initial = nn.Sequential(
            resnet_model.conv1,
            resnet_model.bn1,
            resnet_model.relu,
            resnet_model.maxpool,
        )
        self.layer1 = resnet_model.layer1
        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4

    def forward(self, x):
        x = self.initial(x)
        low_level_features = self.layer1(x)  # Feature map from early layers
        x = self.layer2(low_level_features)
        x = self.layer3(x)
        x = self.layer4(x)  # Final feature map for encoder
        return low_level_features, x


# ASPP: Atrous Spatial Pyramid Pooling
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.aspp1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)
        self.aspp2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6, padding_mode='zeros')
        self.aspp3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.aspp4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        )
        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        return self.relu(x)


# Decoder
class DeepLabDecoder(nn.Module):
    def __init__(self, low_level_channels, num_classes):
        super(DeepLabDecoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_channels, 48, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.relu = nn.ReLU()

    def forward(self, low_level_features, high_level_features):
        low_level_features = self.conv1(low_level_features)
        high_level_features = F.interpolate(high_level_features, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((low_level_features, high_level_features), dim=1)
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x


# DeepLabV3 Model
class DeepLabV3(nn.Module):
    def __init__(self, num_classes, backbone):
        super(DeepLabV3, self).__init__()
        self.backbone = backbone
        self.aspp = ASPP(in_channels=2048, out_channels=256)
        self.decoder = DeepLabDecoder(low_level_channels=256, num_classes=num_classes)

    def forward(self, x):
        low_level_features, high_level_features = self.backbone(x)
        x = self.aspp(high_level_features)
        x = self.decoder(low_level_features, x)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
        return x


# Instantiate the model
if __name__ == "__main__":
    # Load pre-trained ResNet-101 from torchvision
    resnet101 = resnet.resnet101(pretrained=True)
    backbone = ResNetBackbone(resnet101)

    # Create the DeepLabV3 model
    model = DeepLabV3(num_classes=8, backbone=backbone)

    # Test the model with a random input
    input_tensor = torch.randn(8, 5, 256, 256)  # Batch size = 8, Channels = 5, Height/Width = 256x256
    output = model(input_tensor)
    print("Output shape:", output.shape)  # Expected output shape: [8, 21, 256, 256]