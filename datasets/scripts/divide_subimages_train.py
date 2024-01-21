import sys
sys.path.append('../..')
import os.path as osp
import os
import numpy as np
import shutil
from tqdm import tqdm

#divide training data
LR_folder="/workspace/datasets/DIV2K_train_sub/LR/x4"
GT_folder="/workspace/datasets/DIV2K_train_sub/HR/x4"

save_list=["/workspace/datasets/DIV2K_train_sub_class3/LR/x4",
           "/workspace/datasets/DIV2K_train_sub_class2/LR/x4",
           "/workspace/datasets/DIV2K_train_sub_class1/LR/x4",
           "/workspace/datasets/DIV2K_train_sub_class3/HR/x4",
           "/workspace/datasets/DIV2K_train_sub_class2/HR/x4",
           "/workspace/datasets/DIV2K_train_sub_class1/HR/x4"]
for i in save_list:
    if os.path.exists(i):
        pass
    else:
        os.makedirs(i)
threshold=[27.16882,35.149761]

#f1 = open("/data0/xtkong/ClassSR-github/codes/data_scripts/divide_val.log")
f1 = open("/workspace/datasets/divide_train.log")
a1 = f1.readlines()
#index=0
for i in tqdm(a1):
    #index+=1
    #print(index)
    if ('- PSNR:' in i and 'INFO:' in i) and ('results' not in i):
        psnr=float(i.split('PSNR: ')[1].split(' dB')[0])
        filename=i.split('INFO: ')[1].split(' ')[0]
        filename=filename+".png"
        #print(filename,psnr)
        if psnr < threshold[0]:
            shutil.copy(osp.join(LR_folder, filename), osp.join(save_list[0], filename))
            shutil.copy(osp.join(GT_folder, filename), osp.join(save_list[3], filename))
        if psnr >= threshold[0] and psnr < threshold[1]:
            shutil.copy(osp.join(LR_folder, filename), osp.join(save_list[1], filename))
            shutil.copy(osp.join(GT_folder, filename), osp.join(save_list[4], filename))
        if psnr >= threshold[1]:
            shutil.copy(osp.join(LR_folder, filename), osp.join(save_list[2], filename))
            shutil.copy(osp.join(GT_folder, filename), osp.join(save_list[5], filename))

f1.close()