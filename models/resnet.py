import torch
import torch.nn as nn
import torchvision.models as models
from models.component import _ResLayer


class ResNet152(nn.Module):
    def __init__(self, n_blocks, multi_grids, output_stride):
        #super(ResNet50, self).__init__()
        super(ResNet152, self).__init__()

        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]

        ch = [64 * 2 ** p for p in range(6)]
        #ch = [64, 128, 256, 512, 1024, 2048]

        #resnet = models.resnet50()
        resnet = models.resnet152()
        #resnet.load_state_dict(torch.load(r"C:\Users\hekai\Desktop\github-repo\High-Resolution-Remote-Sensing-Semantic-Segmentation-PyTorch\models\pretrained_model\resnet50-19c8e357.pth"))
        self.resnet = nn.Sequential(*list(resnet.children())[:-3])
        print("You are using resnet152!")
        self.add_module("layer5", _ResLayer(n_blocks[3], ch[4], ch[5], s[3], d[3], multi_grids))

    def forward(self, x):
        out = self.resnet(x)
        out = self.layer5(out)
        return out

if __name__ == "__main__":
    model = ResNet152([3,4,6,3], [1,2,4], 16)
    model.eval()
    image = torch.randn(1, 3, 512, 512)
    print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)