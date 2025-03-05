import os
import cv2
import math
import shutil

import numpy as np

import torchvision.transforms as transforms


from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma
)

def create_file(file_path):
    if not os.path.exists(file_path):
        f = open(file_path, 'w+')
        f.close()

def copy_file(from_path, dest_path):
    shutil.copy(from_path, dest_path)

def remove_file(file_path):
    try:
        os.remove(file_path)
    except:
        print('Unable to remove {} file!'.format(file_path))
        pass

def create_directory(src_path):
    if not os.path.exists(src_path):
        os.makedirs(src_path)
    else:
        print('Unable create path: {}, it\'s already exsists'.format(src_path))

def delete_directory(dir_path):
    try:
        shutil.rmtree(dir_path)
    except:
        pass

def clean_directory(src_path):
    try:
        for f in os.listdir(src_path):
            if os.path.isfile(src_path + '/' + f):
                os.remove(src_path + '/' + f)
            if os.path.isdir(src_path + '/' + f):
                shutil.rmtree(src_path + '/' + f)
    except:
        pass

def get_all_subdirs(src_path):
    pass


def get_all_filenames_in_dir(src_path, ext=None):

    try:

        files_list = []

        for element in os.listdir(src_path):
            element_path = os.path.join(src_path, element)

            if (os.path.isfile(element_path)):
                if ext:
                    if element_path.endswith(ext):
                        files_list.append(element_path)
                else:
                    files_list.append(element_path)
        return files_list
    except:
        raise Exception('invalid format of extention!')



def change_img_brightness(input_img, value=30, norm=True):
    hsv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    result_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    if (norm):
        return result_img / 255.0
    else:
        return result_img

def change_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=-10.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))

    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def gamma_correction(input_img, gamma=1.0, norm=True):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    result_img = cv2.LUT(input_img, table)

    if (norm):
        return result_img / 255.0
    else:
        return result_img


def make_tiles(out_path, input_img, img_num=0, img_name='img.png', subfolder=False, patch_size=256, step=256):

    print('width: ', input_img.shape[1], 'height: ', input_img.shape[0])
    x_count = math.ceil((input_img.shape[1] - patch_size) / step)
    y_count = math.ceil((input_img.shape[0] - patch_size) / step)

    print('x_count: ', x_count, 'y_count: ', y_count)

    create_directory(out_path)

    # for x in tqdm(range(0, step * x_count, step)):
    for x in range(0, step * x_count, step):
        for y in range(0, step * y_count, step):
            patch = input_img[y:y + patch_size, x:x + patch_size]

            if subfolder:
                folder_name = out_path + '/F' + str(img_num) + '_X' + str(x) + '_Y' + str(y)
                create_directory(folder_name)
                #print(folder_name + '/' + img_name)
                cv2.imwrite(folder_name + '/' + img_name, patch)
            else:
                print(out_path + '/X_' + str(x) + '_Y_' + str(y) + '.png')
                cv2.imwrite(out_path + '/X_' + str(x) + '_Y_' + str(y) + '.png', patch)
            del(patch)

def make_tiles_from_all_imgs(imgs_path, class_name='forest'):

    out_path = './result/out_patches'

    for sub_dir in os.listdir(imgs_path):
        img_path = imgs_path + '/' + sub_dir + '/img.png'
        mask_path = imgs_path + '/' + sub_dir + '/' + class_name + '.png'

        if os.path.exists(mask_path):
            print('sub_dir: ', sub_dir)
            print('img_path: ', img_path)
            print('mask_path: ', mask_path)

            img = cv2.imread(img_path, 1)
            mask = cv2.imread(mask_path, 1)

            make_tiles(out_path, img, subfolder=True, img_num=int(sub_dir), patch_size=512, step=128,
                       img_name='img.png')
            make_tiles(out_path, mask, subfolder=True, img_num=int(sub_dir), patch_size=512, step=128,
                       img_name=class_name + '.png')

            print('-' * 50)


def create_csv_from_folder(out_path, file_path):
    create_directory(out_path)
    create_file(file_path)

def create_csv_for_datasets(src_path):

    for sub_dir in os.path.listdir(src_path):
        pass

def test_albumentations(input_path, out_path, class_name='forest', limit=150):
    clean_directory(out_path)
    create_directory(out_path)
    aug = Compose([
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        RandomRotate90(p=0.5),
        OneOf([
            #ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            GridDistortion(p=0.5),
            OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
        ], p=0.6),
        CLAHE(p=0.6),
        RandomBrightnessContrast(p=0.6),
        RandomGamma(p=0.6)])
    counter = 0

    for subdir in os.listdir(input_path):
        img_path = input_path + '/' + subdir + '/' + 'img.png'
        mask_path = input_path + '/' + subdir + '/' + class_name + '.png'

        img = cv2.imread(img_path, 1)
        mask = cv2.imread(mask_path, 0)
        print('img_path: ', img_path, 'mask_path: ', mask_path, 'img_dtype: ', img.dtype, ' mask_dtype: ', mask.dtype)

        augmented = aug(image=img, mask=mask)

        image_heavy = augmented['image']
        mask_heavy = augmented['mask']
        print(image_heavy.dtype, ' ', mask_heavy.dtype)

        cv2.imwrite(out_path + '/test_img_' + str(counter) + '.png', image_heavy)
        cv2.imwrite(out_path + '/test_mask_' + str(counter) + '.png', mask_heavy)
        counter += 1

if __name__ == '__main__':
    #make_tiles_from_all_imgs('./datasets/imgs/train/Datasets_gib_checked')
    #create_csv_for_datasets()
    #test_albumentations(input_path='./datasets/imgs/train/forest/egor_forest_cutted/cutted', out_path='./result/datagen_test', limit=333)

    # img = cv2.imread('/home/nv/Documents/datasets/imgs/custom_maps_400/train/574.png')
    # print(img)
    # img = img / 255.
    # print(img)
    #
    # transforms = transforms.Compose([
    #                                  # transforms.CenterCrop(size=(img_size, img_size*2)),
    #                                  transforms.ToTensor()
    #                                  ])
    # print('-'*150)
    # img = transforms(img)
    #
    # print(img)


    input_img = cv2.imread('/home/nv/Documents/datasets/imgs_test/test/test_yoshkar_ola_4/test_yoshkar_ola_4.png')
    make_tiles(input_img=input_img,
               out_path='/home/nv/Documents/result/patches',
               subfolder=False,
               patch_size=400,
               step=256)
