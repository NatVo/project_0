
import os
import cv2
import numpy as np
from common.common import clean_directory, create_directory

def make_concatenated_results(root_path, output_dir):
    sub_path = os.listdir(root_path)
    output_path = os.path.join(root_path, output_dir)
    create_directory(output_path)
    print(sub_path)
    current_path = os.path.join(root_path, 'opt')
    counter = 0

    for f in os.listdir(current_path):
        current_file_path = os.path.join(current_path, f)
        if os.path.isfile(current_file_path):
            opt_img = cv2.imread(current_file_path)
            sar_img_path = current_file_path.replace('opt', 'sar')
            predict_img_path = current_file_path.replace('opt', 'predict')
            gt_img_path = current_file_path.replace('opt', 'gt')

            sar_img = cv2.imread(sar_img_path)
            predict_img = cv2.imread(predict_img_path)
            gt_img = cv2.imread(gt_img_path)

            #print(opt_img.shape, ' ', sar_img.shape, ' ', predict_img.shape, ' ', gt_img.shape)

            out_img = np.concatenate((opt_img, sar_img, predict_img, gt_img), axis=1)
            #print(os.path.join(output_path, (str(counter) + '.png')))
            cv2.imwrite(os.path.join(output_path, (str(counter) + '.png')), out_img)
            counter += 1



if __name__ == "__main__":

    make_concatenated_results('/home/natvo/Documents/Semantic_Segmentation_Projects/tmp_projects/SOLC/result',
                              'output')