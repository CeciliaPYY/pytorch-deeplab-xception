import cv2
import easydict
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import time

import torch
import torch.nn as nn

from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd
from modeling.deeplab import *
from torch.utils.data import DataLoader

# load model structure and training parameters
model = DeepLab(num_classes=3,
                backbone='resnet',
                output_stride=16,
                sync_bn=None,
                freeze_bn=False)

model_path = "./run/coco/deeplab-resnet/experiment_10/checkpoint.pth.tar" # this is where your training result is
model.load_state_dict(torch.load(model_path)['state_dict'])

# let the model preparing for evaluating
model.eval()

# setting base args
args = easydict.EasyDict({"backbone": "resnet",
                         "out-stride": 16,
                         "dataset": "coco",
                         "use-sbd": True,
                         "workers": 1,
                         "base-size": 513,
                         "crop_size": 513,
                         "sync-bn": None,
                         "freeze-bn": False,
                         "loss-type": "ce",
                         "epochs": 1000,
                         "start_epoch": 0,
                         "batch_size": 1,
                         "test-batch-size": 1,
                         "use-balanced-weights": False,
                         "lr": 0.01,
                         "lr-scheduler": "poly",
                         "momentum": 0.9,
                         "weight-decay": 5e-4,
                         "nesterov": False,
                         "cuda": True,
                         "gpu-ids": 0,
                         "seed": 1,
                         "resume": None,
                         "checkname": None,
                         "ft": False,
                         "eval-interval": 1,
                         "no-val": False})

kwargs = {'num_workers': 4, 'pin_memory': True}

# load testing images
test_path = "/home/work/yuyanpeng/pytorch-deeplab-xception/coco/images/test2017"
test_imgs = os.listdir(test_path)
test_imgs = [os.path.join(test_path, im) for im in test_imgs if im.endswith("JPG")]

# test inference time
all_time = time.time()
for i in range(len(test_imgs)):
    print(test_imgs[i])
    a_time = time.time()
    img = Image.open(test_imgs[i])
    
    # --- image cropping ---
    crop_size = args.crop_size
    w, h = img.size
    if w > h:
        oh = args.crop_size
        ow = int(1.0 * w * oh / h)
    else:
        ow = args.crop_size
        oh = int(1.0 * h * ow / w)
    img = img.resize((ow, oh), Image.BILINEAR)
    # center crop
    w, h = img.size
    x1 = int(round((w - args.crop_size) / 2.))
    y1 = int(round((h - args.crop_size) / 2.))
    #     img = img.crop((x1, y1, x1 + args.crop_size, y1 + args.crop_size))

# --- normalizing ---
mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)
img = np.array(img).astype(np.float32)
img /= 255.0
    img -= mean
    img /= std
    
    
    # --- to tensor ---
    img = np.array(img).astype(np.float32).transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img).float()
    img = img.cuda()
    
    with torch.no_grad():
        model.cuda()
        output = model(img)
print("The very one test cost :{}".format(time.time() - a_time))
print("The total images test cost :{}".format(time.time() - all_time))
print("The average inference of images cost :{} sec".format((time.time() - all_time)/ len(test_imgs)))


# you can alse show result mask
org_imgs = []
resize_ims = []
result_mask_imgs = []
for i in range(len(test_imgs)):
    print(test_imgs[i])
    a_time = time.time()
    img = Image.open(test_imgs[i])
    org_imgs.append(img)
    
    # --- image cropping ---
    crop_size = args.crop_size
    w, h = img.size
    if w > h:
        oh = args.crop_size
        ow = int(1.0 * w * oh / h)
    else:
        ow = args.crop_size
        oh = int(1.0 * h * ow / w)
    img = img.resize((ow, oh), Image.BILINEAR)
    # center crop
    w, h = img.size
    x1 = int(round((w - args.crop_size) / 2.))
    y1 = int(round((h - args.crop_size) / 2.))
    #     img = img.crop((x1, y1, x1 + args.crop_size, y1 + args.crop_size))

# --- normalizing ---
mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)
img = np.array(img).astype(np.float32)
img /= 255.0
    img -= mean
    img /= std
    
    resize_ims.append(img)
    
    # --- to tensor ---
    img = np.array(img).astype(np.float32).transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img).float()
    img = img.cuda()
    
    with torch.no_grad():
        model.cuda()
        output = model(img)
        result_mask_imgs.append(output.cpu().numpy()[0].transpose([1,2,0]))

for i in range(len(result_mask_imgs)):
    plt.imshow(result_mask_imgs[i])
    plt.title("{}".format(test_imgs[i]))
    plt.axis("off")
    plt.show()






