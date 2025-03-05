import torch
import torch.nn as nn
import torch.nn.functional as F
from models.deeplabv3_version_3.resnet import ResNet
from models.deeplabv3_version_3.aspp import _ASPP
from models.deeplabv3_version_3.component import _Stem, _ResLayer, _ConvBnReLU

ch = [64 * 2 ** p for p in range(6)]
# ch = [64, 128, 256, 512, 1024, 2048]

class DeepLabV3(nn.Module):
    def __init__(self, num_classes=16, n_blocks=[3,4,6,3], atrous_rates=[6,12,18], multi_grids=[1,2,1], output_stride=16,
                 resnet_backnone = 'ResNet50', use_dropout=False):
        super(DeepLabV3, self).__init__()
        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]
        # print(ch)
        self.num_classes = num_classes
        self.use_dropout = use_dropout


        print('\nUSING BACKBONE: ', resnet_backnone)
        self.backbone = ResNet(n_blocks, multi_grids, output_stride, resnet_backnone)
        self.backbone.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if (self.use_dropout):
            print('USING DROPOUT\n')
            self.dropout = self.dropout = nn.Dropout2d(0.2)

        self.conv = nn.Conv2d(256, 2048, kernel_size=1)
        #self.transposed_conv = nn.ConvTranspose2d(256, 2048, kernel_size=4, stride=4, padding=0)

        self.add_module("aspp", _ASPP(ch[5], 256, atrous_rates))
        concat_ch = 256 * (len(atrous_rates) + 2)
        self.add_module('layer0', _Stem(ch[0]))
        self.add_module("layer1", _ResLayer(n_blocks[0], ch[0], ch[2], s[0], d[0]))
        self.add_module("layer2", _ResLayer(n_blocks[1], ch[2], ch[3], s[1], d[1]))
        self.add_module("layer3", _ResLayer(n_blocks[2], ch[3], ch[4], s[2], d[2]))
        self.add_module("layer4", _ResLayer(n_blocks[3], ch[4], ch[5], s[3], d[3], multi_grids))
        self.add_module("fc1", _ConvBnReLU(concat_ch, 256, 1, 1, 0, 1))
        self.add_module("fc2", nn.Conv2d(256, num_classes, kernel_size=1))

    # def forward(self, x, y):
    #
    #     fea = torch.cat([x, y], dim=1)
    #     feature_map = self.backbone.conv1(fea)
    #
    #     if(self.use_dropout):
    #         feature_map_drop_out = self.dropout(feature_map)
    #         x1 = self.layer1(feature_map_drop_out)
    #     else:
    #         x1 = self.layer1(feature_map)
    #     #print(feature_map)
    #     #x0 = self.layer0(fea)
    #
    #     x2 = self.layer2(x1)
    #     x3 = self.layer3(x2)
    #     feature_map = self.layer4(x3)
    #     print('OUTPUT feature_map:', feature_map.size())
    #     output = self.aspp(feature_map)
    #     print('OUTPUT LAYERS SIZE:', output.size())
    #     output = self.fc1(output)
    #     output = self.fc2(output)
    #     output = F.interpolate(output, size=(x.size()[2], x.size()[3]), mode="bilinear", align_corners=False)
    #
    #     return  output

    def forward(self, x, y):

        fea = torch.cat([x, y], dim=1)
        feature_map = self.backbone.conv1(fea)

        if(self.use_dropout):
            feature_map_drop_out = self.dropout(feature_map)
            x1 = self.layer1(feature_map_drop_out)
        else:
            x1 = self.layer1(feature_map)

        #print('OUTPUT X1 LAYER SIZE:', x1.size())
        x1 = self.conv(x1)
        #x1 = x1.reshape(8, 2048, 32, 32)
        output = self.aspp(x1)

        output = self.fc1(output)
        output = self.fc2(output)
        output = F.interpolate(output, size=(x.size()[2], x.size()[3]), mode="bilinear", align_corners=False)
        return  output

if __name__ == "__main__":
    model = DeepLabV3()
    model.train()
    image = torch.randn(1, 5, 256, 256)
    print(model)
    print("input IMAGE SHAPE:", image.shape)
    print("output:", model(image).shape)



