import time

import argparse
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# from models.deeplabv3_version_1.deeplabv3 import DeepLabV3

from torchinfo import summary
from torchsummary import summary as t_summary

from torch.autograd import Variable
import torch
import os
import pandas as pd
from PIL import Image
import cv2 as cv
from collections import OrderedDict
import torch.nn as nn
from dataset import WHUOPTSARDataset
from dataset import img_sar_transform, img_opt_transform, mask_transform
from models.deeplabv3_version_3.deeplabv3 import DeepLabV3 as SOSeg
from models.deeplabv3_version_3.deeplabv3 import DeepLabV3 as deeplabv3
from palette import colorize_mask
from torchvision import transforms
from libs import average_meter, metric
# from models.SOLC.solc import SOLC
# from models.SOLCV2.solcv2 import SOLCV2
# from models.SOLCV5.solcv5 import SOLCV5
from models.SOLCV7.solcv7 import SOLCV7
from models.MCANet.mcanet import MCANet
from models.u_net import UNet
from models.seg_net import SegNet

from ptflops import get_model_complexity_info

img_transform = transforms.Compose([
    transforms.ToTensor()])
    
resore_transform = transforms.Compose([
    transforms.ToPILImage()
])

from class_names import eight_classes
class_name = eight_classes()


def snapshot_forward(model, dataloader, save_path, num_classes, output_path):
    model.eval()
    total_mIoU = 0
    preprocessing_time = 0
    output_time = 0
    postprocessing_time = 0
    total_time = 0

    for index, data in enumerate(dataloader):

        start_total_time = time.time()
        start_preprocessing_time = time.time()

        imgs_sar = Variable(data[0])
        imgs_opt = Variable(data[1])
        masks = Variable(data[2])
        # print(imgs_sar.shape, imgs_opt.shape, masks.shape)

        imgs_sar = imgs_sar.cuda()
        imgs_opt = imgs_opt.cuda()
        masks = masks.cuda()

        preprocessing_time += time.time() - start_preprocessing_time
        print(f"Preprocessing time: {(time.time() - start_preprocessing_time) * 1000:.4f} mseconds")
        #-----------------------------------------------------------------------------
        start_output_time = time.time()

        outputs = model(imgs_sar, imgs_opt)

        output_time += time.time() - start_output_time
        print(f"Output time: {(time.time() - start_output_time) * 1000:.4f} mseconds")
        #-----------------------------------------------------------------------------
        start_postprocessing_time = time.time()

        preds = torch.argmax(outputs, 1)
        preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)
        masks = masks.data.cpu().numpy().squeeze().astype(np.uint8)

        postprocessing_time += time.time() - start_postprocessing_time
        print(f"Postprocessing time: {(time.time() - start_postprocessing_time) * 1000:.4f} mseconds")

        total_time += time.time() - start_total_time
        print(f"Total time: {(time.time() - start_total_time) * 1000:.4f} mseconds")
        print('-----------------------------------------------------')

        conf_mat = np.zeros((num_classes, num_classes)).astype(np.int64)

        for i in range(masks.shape[0]):
            
            img_pil = resore_transform(imgs_opt[i])
            img_sar = resore_transform(imgs_sar[i])
            preds_pil = Image.fromarray(preds[i].astype(np.uint8)).convert('L')
            pred_vis_pil = colorize_mask(preds[i])
            gt_vis_pil = colorize_mask(masks[i])
            data = np.array(img_pil)[:, :, :3]
            img_pil = Image.fromarray(np.uint8(data[:, :]))
            img_sar = Image.fromarray(np.uint8(img_sar))

            dir_list = ['opt', 'label', 'sar', 'predict', 'gt']
            rgb_save_path = os.path.join(save_path, dir_list[0] )
            sar_save_path = os.path.join(save_path, dir_list[2] )
            label_save_path = os.path.join(save_path, dir_list[1])
            vis_save_path = os.path.join(save_path, dir_list[3])
            gt_save_path = os.path.join(save_path, dir_list[4])
            #print('SAR PATH: ', sar_save_path)

            path_list = [rgb_save_path, label_save_path, sar_save_path, vis_save_path, gt_save_path]
            for path in range(5):
                if not os.path.exists(path_list[path]):
                    os.makedirs(path_list[path])
            img_pil.save(os.path.join(path_list[0], 'opt_%d_%d.png' % (index, i)))
            img_sar.save(os.path.join(path_list[2], 'sar_%d_%d.png' % (index, i)))
            preds_pil.save(os.path.join(path_list[1], 'label_%d_%d.png' % (index, i)))
            pred_vis_pil.save(os.path.join(path_list[3], 'predict_%d_%d.png' % (index, i)))
            gt_vis_pil.save(os.path.join(path_list[4], 'gt_%d_%d.png' % (index, i)))

            conf_mat += metric.confusion_matrix(pred=preds.flatten(),
                                                label=masks.flatten(),
                                                num_classes=num_classes)

        test_acc, test_acc_per_class, test_acc_cls, test_IoU, test_mean_IoU, test_kappa = metric.evaluate(conf_mat)
        total_mIoU += test_mean_IoU

    # for i in range(num_classes):
    #     print(i, eight_classes()[i], test_acc_per_class[i], test_IoU[i])
    print('dataloader length: {}'.format(len(dataloader)))
    print('mean test IoU: {}:', total_mIoU / len(dataloader))

    print(f"Preprocessing time: {(preprocessing_time / len(dataloader)) * 1000:.4f} mseconds")
    print(f"Output time: {(output_time / len(dataloader)) * 1000:.4f} mseconds")
    print(f"Postprocessing time: {(postprocessing_time / len(dataloader)) * 1000:.4f} mseconds")
    print(f"Totaaal time: {(total_time / len(dataloader)) * 1000:.4f} mseconds")

    with open(os.path.join(output_path, "result_test.txt"), "a") as file:
        # file.write("test_acc: {}: \n".format(test_acc))
        file.write("test_mean_IoU: {}: \n".format(total_mIoU / len(dataloader)))
        # file.write("test kappa: {}: \n".format(test_kappa))


def parse_args():
    parser = argparse.ArgumentParser(description="predict")
    parser.add_argument('--test-data-root', type=str, default='/home/natvo/Documents/Semantic_Segmentation_Projects/tmp_projects/SOLC/data/test')
    parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                        help='batch size for testing (default:16)')
    parser.add_argument('--num_workers', type=int, default=8)
    #parser.add_argument('--model', type=str, default='deeplabv3_version_3', help='model name')
    #parser.add_argument('--model', type=str, default='deeplabv3_version_3', help='model name')
    parser.add_argument("--output-path", type=str,
                        default='/home/natvo/Documents/Semantic_Segmentation_Projects/tmp_projects/SOLC/result_models/deeplabv3_version_3/02-27-16:56:37')
    parser.add_argument("--model-name", type=str,
                        default='epoch_179_oa_0.80721_kappa_0.73025_latest.pth')
    parser.add_argument("--config-file", type=str,
                        default='config.csv')
    parser.add_argument("--pred-path", type=str, default="/home/natvo/Documents/Semantic_Segmentation_Projects/tmp_projects/SOLC/result")
    parser.add_argument('--n-blocks', type=str, default='3, 4, 23, 3', help='')
    parser.add_argument('--output-stride', type=int, default=16, help='') # len=16
    parser.add_argument('--multi-grids', type=str, default='1, 1, 1', help='')
    parser.add_argument('--deeplabv3-atrous-rates', type=str, default='6, 12, 18', help='')
    args = parser.parse_args()
    return args


def reference():
    args = parse_args()

    df = pd.read_csv(os.path.join(args.output_path, args.config_file))
    print(int(df.at[0, 'val_batch_size']))

    n_blocks = args.n_blocks
    n_blocks = [int(b) for b in n_blocks.split(',')]
    atrous_rates = args.deeplabv3_atrous_rates
    atrous_rates = [int(s) for s in atrous_rates.split(',')]
    multi_grids = args.multi_grids
    multi_grids = [int(g) for g in multi_grids.split(',')]



    dataset = WHUOPTSARDataset(class_name=class_name,
                               root=args.test_data_root,
                               img_sar_transform=img_sar_transform,
                               img_opt_transform=img_opt_transform,
                               mask_transform=mask_transform,
                               open_with_PIL=df.at[0, 'open_with_PIL'],
                               normalize=df.at[0, 'normalize_images'],
                               sync_transforms=None
                              )

    dataloader = DataLoader(dataset=dataset, batch_size=int(df.at[0, 'val_batch_size']), shuffle=False, num_workers=8)



    print(class_name, len(class_name))
    """
    model = SOSeg(num_classes=len(class_name),
                          n_blocks=n_blocks,
                          atrous_rates=atrous_rates,
                          multi_grids=multi_grids,
                          output_stride=args.output_stride)
    """
    
    #model = MCANet(num_classes=len(class_name))


    if df.at[0, 'model'] == 'solcv7':
        model = SOLCV7(num_classes=len(class_name))

    if df.at[0, 'model'] == 'deeplabv3_version_3':
        model = deeplabv3(num_classes=len(class_name),
                          n_blocks=n_blocks,
                          atrous_rates=atrous_rates,
                          multi_grids=multi_grids,
                          output_stride=args.output_stride,
                          resnet_backnone=df.at[0, 'model_backbone'],
                          use_dropout=df.at[0, 'use_dropout'])
    # model = MCANet(num_classes=len(class_name))

    # model = SOSeg(num_classes=len(class_name),
     #                     n_blocks=n_blocks,
     ##                     atrous_rates=atrous_rates,
     #                     multi_grids=multi_grids,
      #                    output_stride=args.output_stride)

    # print(model)
    # summary(model, input_size=((8, 4, 256, 256), (8, 1, 256, 256)))
    #
    # print('\n\n')

    # with torch.cuda.device(0):  # Указываем устройство (0 - GPU, если доступен)
    #     flops, params = get_model_complexity_info(model, input_res = ((1, 4, 256, 256), (1, 1, 256, 256)), as_strings=True, print_per_layer_stat=True)
    #     print(f"FLOPS: {flops}")
    #     print(f"Параметры: {params}")

    print('\n\n')

    state_dict = torch.load(os.path.join(args.output_path, args.model_name))

    with open(os.path.join(args.output_path, "result_test.txt"), "w") as file:
        file.write("MODEL: {}: \n".format(os.path.join(args.output_path, args.model_name)))
        file.write("----------------------------------------------- \n")

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    print('=========> load model success', os.path.join(args.output_path, args.model_name))
    model = model.cuda()
    #model = nn.DataParallel(model, device_ids=[0, 1])

    with open("result_test.txt", "w") as file:
        file.write("Model_path: {}: \n\n".format(os.path.join(args.output_path, args.model_name)))

    snapshot_forward(model, dataloader, args.pred_path, len(class_name), output_path = args.output_path)
    print('test done........')
if __name__ == '__main__':


    reference()