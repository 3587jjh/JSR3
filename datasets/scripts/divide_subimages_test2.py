import sys
sys.path.append('../..')
import os.path as osp
import os
import numpy as np
import shutil
from tqdm import tqdm

#divide testing data for single SR models
LR_folder="/workspace/datasets/DIV2K_valid_sub/LR/x4"
GT_folder="/workspace/datasets/DIV2K_valid_sub/HR/x4"

save_list=["/workspace/datasets/DIV2K_valid_sub_classB/LR/x4",
           "/workspace/datasets/DIV2K_valid_sub_classA/LR/x4",
           "/workspace/datasets/DIV2K_valid_sub_classB/HR/x4",
           "/workspace/datasets/DIV2K_valid_sub_classA/HR/x4"]

for i in save_list:
    if os.path.exists(i):
        pass
    else:
        os.makedirs(i)
threshold=30.58681

f1 = open("/workspace/datasets/divide_val.log")
a1 = f1.readlines()

for i in tqdm(a1):
    if ('- PSNR:' in i and 'INFO:' in i) and ('results' not in i):
        psnr=float(i.split('PSNR: ')[1].split(' dB')[0])
        filename=i.split('INFO: ')[1].split(' ')[0]
        filename=filename+".png"
        #print(filename,psnr)
        if psnr < threshold:
            shutil.copy(osp.join(LR_folder, filename), osp.join(save_list[0], filename))
            shutil.copy(osp.join(GT_folder, filename), osp.join(save_list[2], filename))
        else:
            shutil.copy(osp.join(LR_folder, filename), osp.join(save_list[1], filename))
            shutil.copy(osp.join(GT_folder, filename), osp.join(save_list[3], filename))
f1.close()