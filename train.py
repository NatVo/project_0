import argparse
import time
import os
import csv
import pandas as pd
import json

from dataset import WHUOPTSARDataset
import sync_transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from models.deeplabv3_version_1.deeplabv3 import DeepLabV3 as model1
from models.deeplabv3_version_2.deeplabv3 import DeepLabV3 as model2
from models.deeplabv3_version_3.deeplabv3 import DeepLabV3 as deeplabv3
from models.deeplabv3 import DeepLabV3 as DL3
from models.custom_losses import combined_loss
from libs import average_meter, metric
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import torchvision
from torchvision import transforms
from palette import colorize_mask
from PIL import Image
from collections import OrderedDict
from tensorboardX import SummaryWriter

from torchinfo import summary
from torch.optim import lr_scheduler
# from models.SOLC.solc import SOLC
# from models.SOLCV2.solcv2 import SOLCV2
# from models.SOLCV3.solcv3 import SOLCV3_res50
# from models.SOLCV4.solcv4 import SOLCV4
# from models.SOLCV5.solcv5 import SOLCV5
from models.SOLCV7.solcv7 import SOLCV7
from models.MCANet.mcanet import MCANet
from torch.optim.lr_scheduler import StepLR

import albumentations as A
from albumentations.pytorch import ToTensorV2

alb_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Rotate(limit=40, p=0.5),
        A.GaussNoise(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05,
                           scale_limit=0.05,
                           rotate_limit=15, p=0.5),
        # A.Normalize(mean=(0.485, 0.456, 0.406),
        #             std=(0.229, 0.224, 0.225)),
        #ToTensorV2()
    ],
    additional_targets = {
        'mask_2': 'mask'
    }
)

def make_config_csv(args, file_path):
    config_dict = {}

    config_dict['model'] = [args.model]
    config_dict['model_backbone'] = [args.model_backbone]
    config_dict['optimizer'] = [args.optimizer_name]
    config_dict['total_epochs'] = [args.total_epochs]
    config_dict['train_batch_size'] = [args.train_batch_size]
    config_dict['val_batch_size'] = [args.train_batch_size]
    config_dict['use_dropout'] = [args.use_dropout]
    config_dict['open_with_PIL'] = [args.open_with_PIL]
    config_dict['normalize_images'] = [args.normalize_images]
    config_dict['scheduler_type'] = [args.scheduler_type]
    config_dict['loss_name'] = [args.loss_name]

    df = pd.DataFrame(config_dict)
    df.to_csv(os.path.join(file_path, 'config.csv'), index=False)

    # config_list = []
    #
    # config_list.append(['model', args.model])
    # config_list.append(['model_backbone', args.model_backbone])
    # config_list.append(['optimizer', args.optimizer_name])
    # config_list.append(['total_epochs', args.total_epochs])
    # config_list.append(['train_batch_size', args.train_batch_size])
    # config_list.append(['val_batch_size', args.val_batch_size])
    # config_list.append(['use-dropout', args.use_dropout])
    # config_list.append(['open_with_PIL', args.open_with_PIL])
    # config_list.append(['normalize_images', args.normalize_images])
    # config_list.append(['scheduler_type', args.scheduler_type])
    # config_list.append(['loss_name', args.loss_name])
    #
    # with open(os.path.join(file_path, 'config.csv'), mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(config_list)

    print(f"CSV file '{os.path.join(file_path, 'config.csv')}' created successfully.")


def parse_args():
    parser = argparse.ArgumentParser(description="Remote Sensing Segmentation by PyTorch")
    # dataset
    parser.add_argument('--dataset-name', type=str, default='eight')

    parser.add_argument('--train-data-root', type=str, default='/home/natvo/Documents/Semantic_Segmentation_Projects/tmp_projects/SOLC/data/train')
    parser.add_argument('--val-data-root', type=str, default='/home/natvo/Documents/Semantic_Segmentation_Projects/tmp_projects/SOLC/data/val')
    parser.add_argument('--save_root', type=str, default='/home/natvo/Documents/Semantic_Segmentation_Projects/tmp_projects/SOLC/result_models')
    parser.add_argument('--gpu_ids', type=list, default=[0])
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M', help='weight-decay (default:1e-4)')
    # output_save_path 
    parser.add_argument('--experiment-start-time', type=str, default=time.strftime('%m-%d-%H:%M:%S', time.localtime(time.time())))
    # learning_rate 
    parser.add_argument('--base-lr', type=float, default=1e-3, metavar='M', help='')

    parser.add_argument('--model', type=str, default='deeplabv3_version_3', help='model name')
    # if using deeplabv3
    parser.add_argument('--model-backbone', type=str, default='ResNet101', help='model backbone')
    
    parser.add_argument('--total-epochs', type=int, default=180, metavar='N', help='number of epochs to train (default: 120)')
    parser.add_argument('--train-batch-size', type=int, default=8, metavar='N', help='batch size for training (default:16)')
    parser.add_argument('--val-batch-size', type=int, default=8, metavar='N', help='batch size for testing (default:16)')

    parser.add_argument('--use-dropout', type=bool, default=True)
    parser.add_argument('--open-with-PIL', type=bool, default=True)
    parser.add_argument('--normalize-images', type=bool, default=False)
    parser.add_argument('--scheduler-type', type=str, default='ReduceLROnPlateau')
    #parser.add_argument('--scheduler-type', type=str, default='StepLR')
    
    parser.add_argument('--step_size', type=int, default=80)
    parser.add_argument('--gamma', type=float, default=0.1)

    #parser.add_argument('--model', type=str, default='solcv7', help='model name')
    
    # -===================！！！！！！！
    parser.add_argument('--save-pseudo-data-path', type=str, default='pseudo-data')
    # augmentation
    # parser.add_argument('--base-size', type=int, default=512, help='base image size')
    # parser.add_argument('--crop-size', type=int, default=512, help='crop image size')
    parser.add_argument('--flip-ratio', type=float, default=0.5)
    parser.add_argument('--resize-scale-range', type=str, default='0.5, 2.0')
    # model
    
    #parser.add_argument('--backbone', type=str, default='resnet152', help='backbone name')
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--n-blocks', type=str, default='3, 4, 23, 3', help='')
    parser.add_argument('--output-stride', type=int, default=16, help='') # len=16
    parser.add_argument('--multi-grids', type=str, default='1, 1, 1', help='')
    parser.add_argument('--deeplabv3-atrous-rates', type=str, default='6, 12, 18', help='')
    parser.add_argument('--deeplabv3-no-global-pooling', action='store_true', default=False)
    parser.add_argument('--deeplabv3-use-deformable-conv', action='store_true', default=False)
    parser.add_argument('--no-syncbn', action='store_true', default=False, help='using Synchronized Cross-GPU BatchNorm')
    # criterion
    parser.add_argument('--class-loss-weight', type=list, default=
    [0.0, 0.016682825992096393, 0.12286476797975535, 0.09874940237721894, 0.04047604729817842, 0.015269075073618998, 0.6013717852280317, 0.3362534066400197]) # 2022-06-07...
    parser.add_argument('--start-epoch', type=int, default=0, metavar='N', help='start epoch (default:0)')
    
    # loss
    parser.add_argument('--loss-name', type=str, default='combined_loss')
    parser.add_argument('--classes-weight', type=str, default=None)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default:0.9)')
    
    # optimizer
    parser.add_argument('--optimizer-name', type=str, default='Adam')
    
    # environment
    parser.add_argument('--use-cuda', action='store_true', default=True, help='using CUDA training')
    parser.add_argument('--num-GPUs', type=int, default=2, help='numbers of GPUs')
    
    parser.add_argument('--num_workers', type=int, default=8)
    # validation
    parser.add_argument('--eval', action='store_true', default=False, help='evaluation only')
    parser.add_argument('--no-val', action='store_true', default=False)

    parser.add_argument('--best-kappa', type=float, default=0)


    parser.add_argument('--resume-path', type=str, default=None)
    
    parser.add_argument('--resume_model', type=bool, default=False)
    parser.add_argument('--resume_model_path', type=str, default=
        '')
    parser.add_argument('--resume_start_epoch', type=int, default=0)
    parser.add_argument('--resume_total_epochs', type=int, default=500)

    args = parser.parse_args()
    directory = args.save_root + "/%s/%s/" % ( args.model, args.experiment_start_time)
    args.directory = directory
    if not os.path.exists(directory):
        os.makedirs(directory)
    config_file = os.path.join(directory, 'config.json')
    with open(config_file, 'w') as file:
        json.dump(vars(args), file, indent=4)

    make_config_csv(args, file_path=directory)

    if args.use_cuda:
        print('Numbers of GPUs:', len(args.gpu_ids))
    else:
        print("Using CPU")
    return args

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class Trainer(object):
    def __init__(self, args):
        self.args = args
        resize_scale_range = [float(scale) for scale in args.resize_scale_range.split(',')]

        sync_transform = sync_transforms.ComposeWHU([
            sync_transforms.RandomFlipWHU(args.flip_ratio),
        ])

        self.resore_transform = transforms.Compose([
            transforms.ToPILImage()
        ])
        self.visualize = transforms.Compose([transforms.ToTensor()]) # /255.
        
        dataset_name = args.dataset_name
        class_name = []
        if dataset_name == 'fifteen': 
            from class_names import fifteen_classes 
            class_name = fifteen_classes()
        if dataset_name == 'eight': 
            from class_names import eight_classes
            class_name = eight_classes()
        if dataset_name == 'five': 
            from class_names import five_classes
            class_name = five_classes()
        if dataset_name == 'seven': 
            from class_names import seven_classes
            class_name = seven_classes()
        self.train_dataset = WHUOPTSARDataset(class_name,
                                              root=args.train_data_root,
                                              mode='train',
                                              #sync_transforms=sync_transform,
                                              sync_transforms=None,
                                              open_with_PIL=args.open_with_PIL,
                                              normalize=args.normalize_images,
                                              augmentations=alb_transform) # random flip
        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=args.train_batch_size,
                                       num_workers=args.num_workers,
                                       shuffle=True,
                                       drop_last=True)
        print('class names {}.'.format(self.train_dataset.class_names))
        print('Number samples {}.'.format(len(self.train_dataset)))
        if not args.no_val:
            val_data_set = WHUOPTSARDataset(class_name,
                                            root=args.val_data_root,
                                            mode='val',
                                            sync_transforms=None,
                                            open_with_PIL=args.open_with_PIL,
                                            normalize=args.normalize_images,
                                            )
            self.val_loader = DataLoader(dataset=val_data_set,
                                         batch_size=args.val_batch_size,
                                         num_workers=args.num_workers,
                                         shuffle=False,
                                         drop_last=True)
        self.num_classes = len(self.train_dataset.class_names)
        print("Number of classes： ", self.num_classes)
        print(self.train_dataset.class_names)
        self.class_loss_weight = torch.Tensor(args.class_loss_weight)
         # -===================！！！！！！！  ignore 0

        if (args.loss_name == 'cross_entropy'):
            self.criterion = nn.CrossEntropyLoss(weight=self.class_loss_weight, reduction='mean', ignore_index=0).cuda()
        self.val_mean_IOU = 0.0

        n_blocks = args.n_blocks
        n_blocks = [int(b) for b in n_blocks.split(',')]
        atrous_rates = args.deeplabv3_atrous_rates
        atrous_rates = [int(s) for s in atrous_rates.split(',')]
        multi_grids = args.multi_grids
        multi_grids = [int(g) for g in multi_grids.split(',')]

        if args.model == 'deeplabv3_version_1':
            model = model1(num_classes=self.num_classes)# dilate_rate=[6,12,18]
            # resume
            if args.resume_path:
                state_dict = torch.load('')
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
        if args.model == 'deeplabv3_version_2':
            model = model2(num_classes=self.num_classes,
                           n_blocks=n_blocks,
                           atrous_rates=atrous_rates,
                           multi_grids=multi_grids,
                           output_stride=args.output_stride)
        if args.model == 'deeplabv3_version_3':
            model = deeplabv3(num_classes=self.num_classes,
                              n_blocks=n_blocks,
                              atrous_rates=atrous_rates,
                              multi_grids=multi_grids,
                              output_stride=args.output_stride,
                              resnet_backnone=args.model_backbone,
                              use_dropout=args.use_dropout)

            # model = DL3(num_classes=self.num_classes,
            #             n_blocks=n_blocks,
            #             atrous_rates=atrous_rates,
            #             multi_grids=multi_grids,
            #             output_stride=args.output_stride)
        # if args.model == 'hdc':
        #     from models.HDC.duc_hdc import ResNetDUC
        #     model = ResNetDUC(num_classes=self.num_classes)
        #
        # if args.model == 'solc':
        #     from models.SOLC.solc import SOLC
        #     model = SOLC(num_classes=self.num_classes)
        #     print('======> model SOLC ')
        #
        #
        # if args.model == 'solcv2':
        #     from models.SOLCV2.solcv2 import SOLCV2
        #     model = SOLCV2(num_classes=self.num_classes)
        #     print('======> model SOLC Version 2 ')
        #
        # if args.model == 'solcv3':
        #     from models.SOLCV3.solcv3 import SOLCV3_res50
        #     model = SOLCV3_res50(num_classes=self.num_classes)
        #     print('======> model SOLC Version 3 ')
        #
        #
        # if args.model == 'solcv5':
        #     from models.SOLCV5.solcv5 import SOLCV5
        #     print('n_blocks ', n_blocks, 'atrous_rates ', atrous_rates, 'multi_grids ', multi_grids, 'output_stride ', args.output_stride)
        #     model = SOLCV5(num_classes=self.num_classes, n_blocks=n_blocks,
        #                    atrous_rates=atrous_rates,
        #                    multi_grids=multi_grids,
        #                    output_stride=args.output_stride)
        #     print('======> model SOLC Version 5 ')
            
        if args.model == 'solcv7':
            from models.SOLCV7.solcv7 import SOLCV7
            
            model = SOLCV7(num_classes=self.num_classes)
            print('======> model SOLC Version seven =============== ')    
            # from models.SOLCV7.solcv7 import SOLCV7
            
        if args.model == 'mcanet':
            from models.MCANet.mcanet import MCANet
            
            model = MCANet(num_classes=self.num_classes)
            print('======> model MCANet (Paper) =============== ')    
            # from models.SOLCV7.solcv7 import SOLCV7
            
            
        # print(model)

        if args.resume_model:
            print('resume model', args.resume_model)
            state_dict = torch.load(args.resume_model_path)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            print('=========> resume model success', args.resume_model_path)

        if args.use_cuda:
            model = model.cuda()
            # self.model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
            # -===================！！！！！！！  
            self.model = nn.DataParallel(model, device_ids=args.gpu_ids)


        if args.optimizer_name == 'Adadelta':
            self.optimizer = torch.optim.Adadelta(model.parameters(),
                                                  lr=args.base_lr,
                                                  weight_decay=args.weight_decay)
        if args.optimizer_name == 'Adam':
        # -===================！！！！！！！  ignore 0
            self.optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), 
                                              lr=args.base_lr, weight_decay=args.weight_decay)
        if args.optimizer_name == 'SGD':
            self.optimizer = torch.optim.SGD(params=model.parameters(),
                                             lr=args.base_lr,
                                             momentum=args.momentum,
                                             weight_decay=args.weight_decay)

        self.max_iter = args.total_epochs * len(self.train_loader)
        self.save_pseudo_data_path = args.save_root + '/' + args.save_pseudo_data_path
        # self.mixup_transform = sync_transforms.Mixup()

        # with open(os.path.join(self.args.directory, "output_result.txt"), "w") as file:
        #     file.write("TRAINING RESULTS: \n\n")
        #
        #     file.write("RESNET 101 BACKBONE \n")
        #     file.write("NO DROPOUT \n")
        #     file.write("OPENCV WITH ALBUMENTATIONS \n\n")

        self.make_metrics_csv()

    def make_metrics_csv(self):
        # metrics_name_list = []
        # metrics_name_list.append(['epoch', 'learning_rate', 'train_loss', 'train_mIoU',' val_mIoU'])
        #
        # with open(os.path.join(self.args.directory, 'metrics.csv'), mode='w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(metrics_name_list)
        #
        #metrics_name_dict = {}
        metrics_data_headers = ['epoch', 'learning_rate', 'train_loss', 'train_acc_(OA)', 'train_kappa', 'train_mIoU', 'val_mIoU', 'val_acc_(OA)', 'val_kappa']
        df = pd.DataFrame(columns=metrics_data_headers)
        df.to_csv(os.path.join(self.args.directory, 'metrics.csv'), index=False)

        print(f"CSV file '{os.path.join(self.args.directory, 'metrics.csv')}' created successfully.")

    def fill_train_metrics_csv(self, metrics_train_dict):
        df = pd.DataFrame(metrics_train_dict)
        df.to_csv(os.path.join(self.args.directory, 'metrics.csv'), mode='a', index=False, header=False)

    def fill_val_metrics_csv(self, metrics_val_list, epoch):
        # df = pd.read_csv(os.path.join(self.args.directory, 'metrics.csv'))
        # df['val_mIoU'] = metrics_val_dict
        # df.to_csv(os.path.join(self.args.directory, 'metrics.csv'), index=False)

        df = pd.read_csv(os.path.join(self.args.directory, 'metrics.csv'))
        df.at[epoch, 'val_mIoU'] = metrics_val_list[0]
        df.at[epoch, 'val_acc_(OA)'] = metrics_val_list[1]
        df.at[epoch, 'val_kappa'] = metrics_val_list[2]
        df.to_csv(os.path.join(self.args.directory, 'metrics.csv'), index=False)

    def training(self, epoch):
        
        self.model.train()#
        
        train_loss = average_meter.AverageMeter()

        curr_iter = epoch * len(self.train_loader)
        #lr = self.args.base_lr * (1 - float(curr_iter) / self.max_iter) ** 0.9
        
        conf_mat = np.zeros((self.num_classes, self.num_classes)).astype(np.int64)
        tbar = tqdm(self.train_loader)
        for index, data in enumerate(tbar):
            # assert data[0].size()[2:] == data[1].size()[1:]
            # data = self.mixup_transform(data, epoch)
            imgs_sar = Variable(data[0])
            imgs_opt = Variable(data[1])
            masks = Variable(data[2])

            if self.args.use_cuda:
                imgs_sar = imgs_sar.cuda()
                imgs_opt = imgs_opt.cuda()
                masks = masks.cuda()
            
            self.optimizer.zero_grad()
            
            outputs = self.model(imgs_sar, imgs_opt)
            
            # torch.max(tensor, dim)：
            _, preds = torch.max(outputs, 1)
            preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)


            #loss = self.criterion(outputs, masks)
            #loss = combined_loss(outputs, masks.squeeze(1))

            if (args.loss_name == 'cross_entropy'):
                loss = self.criterion(outputs, masks)
            elif (args.loss_name == 'combined_loss'):
                loss = combined_loss(outputs, masks.squeeze(1))

            train_loss.update(loss, self.args.train_batch_size)
            writer.add_scalar('train_loss', train_loss.avg, curr_iter)
            
            loss.backward()
            self.optimizer.step()

            tbar.set_description('epoch {}/{}, training loss {}, with learning rate {}.'.format(epoch, args.total_epochs,train_loss.avg, self.optimizer.state_dict()['param_groups'][0]['lr']))
            
            masks = masks.data.cpu().numpy().squeeze().astype(np.uint8)
            
            conf_mat += metric.confusion_matrix(pred=preds.flatten(),
                                                label=masks.flatten(),
                                                num_classes=self.num_classes)
                                                
        train_acc, train_acc_per_class, train_acc_cls, train_IoU, train_mean_IoU, train_kappa = metric.evaluate(conf_mat)
        writer.add_scalar(tag='train_loss_per_epoch', scalar_value=train_loss.avg, global_step=epoch, walltime=None)
        writer.add_scalar(tag='train_oa', scalar_value=train_acc, global_step=epoch, walltime=None)
        writer.add_scalar(tag='train_kappa', scalar_value=train_kappa, global_step=epoch, walltime=None)

        for i in range(self.num_classes):
            # table.add_row([i, self.train_dataset.class_names[i], train_acc_per_class[i], train_IoU[i]])
            print('====> class id ', i, self.train_dataset.class_names[i], train_acc_per_class[i], train_IoU[i])
        # print(table)
        print("train_acc (OA):", train_acc)
        print("train_mean_IoU (Iou):", train_mean_IoU)
        print("kappa (Kappa):", train_kappa)

        metrics_dict = {
            'epoch' : [epoch],
            'learning_rate: ': [self.optimizer.state_dict()['param_groups'][0]['lr']],
            'train_loss':[train_loss.avg.item()],
            'train_acc_(OA)': [train_acc],
            'train_kappa': [train_kappa],
            'train_mean_IoU':[train_mean_IoU]
        }
        self.fill_train_metrics_csv(metrics_dict)

        # with open(os.path.join(self.args.directory, "output_result.txt"), "a") as file:
        #     file.write("EPOCH NUM: {} \n".format(epoch))
        #     file.write("----------------------------------------------- \n")
        #     file.write("learning_rate: {} \n".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
        #     file.write("loss: {} \n\n".format(train_loss.avg))
        #     file.write("train_acc (OA): {} \n".format(train_acc))
        #     file.write("train_mean_IoU (Iou): {} \n".format(train_mean_IoU))
        #     file.write("train_kappa (Kappa): {} \n\n".format(train_kappa))
        

    def validating(self, epoch):
        self.model.eval()
        conf_mat = np.zeros((self.num_classes, self.num_classes)).astype(np.int64)
        tbar = tqdm(self.val_loader)
        for index, data in enumerate(tbar):
            # assert data[0].size()[2:] == data[1].size()[1:]
            imgs_sar = Variable(data[0])
            imgs_opt = Variable(data[1])
            masks = Variable(data[2])

            if self.args.use_cuda:
                imgs_sar = imgs_sar.cuda()
                imgs_opt = imgs_opt.cuda()
                masks = masks.cuda()
                
            self.optimizer.zero_grad()
            outputs = self.model(imgs_sar, imgs_opt)
            _, preds = torch.max(outputs, 1)
            preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)
            masks = masks.data.cpu().numpy().squeeze().astype(np.uint8)
            score = _.data.cpu().numpy()
            val_visual = []
            # img_pil = self.resore_transform(data[1][0])
            # img_pil = Image.fromarray(np.uint8(np.array(img_pil)[:, :, :3]))
            # img_pil.convert('RGB')
            # print('convert success')
            for i in range(score.shape[0]):
                num_score = np.sum(score[i] > 0.9)
                if num_score > 0:
                    img_pil = self.resore_transform(data[1][i])
                    preds_pil = Image.fromarray(preds[i].astype(np.uint8)).convert('L')
                    pred_vis_pil = colorize_mask(preds[i])
                    gt_vis_pil = colorize_mask(data[2][i].numpy())
                    img_pil = Image.fromarray(np.uint8(np.array(img_pil)[:, :, :3]))
                    val_visual.extend([self.visualize(img_pil.convert('RGB')),
                                       self.visualize(gt_vis_pil.convert('RGB')),
                                       self.visualize(pred_vis_pil.convert('RGB'))])
            if val_visual:
                val_visual = torch.stack(val_visual, 0)
                val_visual = torchvision.utils.make_grid(tensor=val_visual,
                                                         nrow=3,
                                                         padding=5,
                                                         normalize=False,
                                                         #range=None,
                                                         scale_each=False,
                                                         pad_value=0)
                writer.add_image(tag='pres&GTs', img_tensor=val_visual, global_step=None, walltime=None)
                
            conf_mat += metric.confusion_matrix(pred=preds.flatten(),
                                                label=masks.flatten(),
                                                num_classes=self.num_classes)
                                                
        val_acc, val_acc_per_class, val_acc_cls, val_IoU, self.val_mean_IOU, val_kappa = metric.evaluate(conf_mat)
        writer.add_scalars(main_tag='val_single_oa',
                           tag_scalar_dict={self.train_dataset.class_names[i]: val_acc_per_class[i] for i in range(len(self.train_dataset.class_names))},
                           global_step=epoch, walltime=None)
        writer.add_scalars(main_tag='val_single_iou',
                           tag_scalar_dict={self.train_dataset.class_names[i]: val_IoU[i] for i in range(len(self.train_dataset.class_names))},
                           global_step=epoch, walltime=None)

        writer.add_scalar('val_oa', val_acc, epoch)
        writer.add_scalar('val_oa_per_cls', val_acc_cls, epoch)
        writer.add_scalar('val_mean_IoU', self.val_mean_IOU, epoch)
        writer.add_scalar('val_kappa', val_kappa, epoch)
        model_name = 'epoch_%d_oa_%.5f_kappa_%.5f' % (epoch, val_acc, val_kappa)
        if val_kappa > self.args.best_kappa:
            torch.save(self.model.state_dict(), os.path.join(self.args.directory, model_name+'.pth'))
            self.args.best_kappa = val_kappa
            
        torch.save(self.model.state_dict(), os.path.join(self.args.directory, model_name+'_latest.pth')) #  arg.directory changed 

        for i in range(self.num_classes):
            # table.add_row([i, self.train_dataset.class_names[i], val_acc_per_class[i], val_IoU[i]])
            print('====> class id ', i, self.train_dataset.class_names[i], val_acc_per_class[i], val_IoU[i])
        # print(table)
        print("val_acc (OA):", val_acc)
        print("val_mean_IoU (Iou):", self.val_mean_IOU)
        print("kappa (Kappa):", val_kappa)

        self.fill_val_metrics_csv([self.val_mean_IOU, val_acc, val_kappa], epoch)

        # with open(os.path.join(self.args.directory, "output_result.txt"), "a") as file:
        #
        #     file.write("val_acc (OA): {} \n".format(val_acc))
        #     file.write("val_mean_IoU (Iou): {} \n".format(self.val_mean_IOU))
        #     file.write("kappa (Kappa): {} \n\n".format(val_kappa))


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True
    args = parse_args()
    writer = SummaryWriter(args.directory)
    trainer = Trainer(args)

    if args.eval:
        # print("Evaluating model:", args.resume)
        trainer.validating(epoch=0)
    else:
        print("Starting Epoch:", args.start_epoch)

    if args.resume_model:
        print("=====> Continue Train:")
        args.start_epoch = args.resume_start_epoch
        args.total_epochs = args.resume_total_epochs
    #scheduler = StepLR(trainer.optimizer, step_size=args.step_size, gamma=args.gamma)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
    )

    for epoch in range(args.start_epoch, args.total_epochs):
        trainer.training(epoch)
        scheduler.step(trainer.val_mean_IOU)
        if not trainer.args.no_val:
            trainer.validating(epoch)

