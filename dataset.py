from torch.utils.data import Dataset
import cv2
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import torch

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


img_opt_transform = transforms.Compose([
    transforms.ToTensor(),
    ])

# img_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

img_sar_transform = transforms.Compose([
    transforms.ToTensor(),
    ])

mask_transform = MaskToTensor()

class WHUOPTSARDataset(Dataset):
    def __init__(self, class_name, root, mode=None, img_sar_transform=img_sar_transform,
                 img_opt_transform=img_opt_transform, mask_transform=mask_transform, sync_transforms=None,
                 open_with_PIL = None,
                 normalize=None, augmentations=None):

        self.class_names = class_name
        self.mode = mode
        self.img_sar_transform = img_sar_transform
        self.img_opt_transform = img_opt_transform
        self.mask_transform = mask_transform
        self.sync_transform = sync_transforms
        self.sync_img_mask = []
        self.normalize = normalize
        self.augmentations = augmentations
        self.open_with_PIL = open_with_PIL

        img_sar_dir = os.path.join(root, 'sar')
        img_opt_dir = os.path.join(root, 'opt')
        mask_dir = os.path.join(root, 'lbl')

        for img_filename in os.listdir(img_sar_dir):
            img_mask_pair = (os.path.join(img_sar_dir, img_filename),
                             os.path.join(img_opt_dir, img_filename),
                             os.path.join(mask_dir, img_filename))
            self.sync_img_mask.append(img_mask_pair)
        # print(self.sync_img_mask)

        if (len(self.sync_img_mask)) == 0:
            print("Found 0 data, please check your dataset!")

    def __getitem__(self, index):
        img_sar_path, img_opt_path, mask_path = self.sync_img_mask[index]

        if self.open_with_PIL:
            img_sar = Image.open(img_sar_path)
            img_opt = Image.open(img_opt_path)
            mask = Image.open(mask_path).convert('L')

            # # print(img_sar.mode, img_opt.mode, mask.mode)
        else:
            img_sar = cv2.imread(img_sar_path, cv2.IMREAD_GRAYSCALE)
            img_opt = cv2.imread(img_opt_path, cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if self.normalize:
                img_sar = img_sar.astype(np.float32)
                img_opt = cv2.cvtColor(img_opt, cv2.COLOR_BGRA2RGBA).astype(np.float32) / 255.0
                mask = mask.astype(np.float32)

            if self.augmentations:
                augmented = self.augmentations(image=img_opt, mask_2=img_sar, mask=mask)

                img_opt = augmented["image"]
                mask = augmented["mask"]
                img_sar = augmented["mask_2"]


        if self.sync_transform is not None:
            img_sar, img_opt, mask = self.sync_transform(img_sar, img_opt, mask)
        if self.img_sar_transform is not None:
            img_sar = self.img_sar_transform(img_sar)
            img_opt = self.img_opt_transform(img_opt)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        #print(img_sar.shape, img_opt.shape, mask.shape)
        return img_sar, img_opt, mask

    def __len__(self):
        return len(self.sync_img_mask)

    def classes(self):
        return self.class_names

if __name__ ==  "__main__":
    pass