import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import numpy as np 
import pickle
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time
import os
import argparse
import errno


sys.path.append('model/')
sys.path.append('utils/')
sys.path.append('datasets/')

from deeplabv3 import DeepLabV3
from cityscapes_dataset import DATASET_CITYSCAPES, DATASET_CITYSCAPES_FOGGY

parser = argparse.ArgumentParser(description='Compute Confusion Matrix')
print(os.getcwd())
parser.add_argument('--model_path', type=str, default= os.getcwd() + '/pretrained_models/model_13_2_2_2_epoch_580.pth')
parser.add_argument('--fog_scale', type=float, default=0.005)
parser.add_argument('--dataset_path', type=str, default='data/cityscapes')
parser.add_argument('--compute_unperturbed', action='store_true')
args = parser.parse_args()

device         = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name     = '0'
batch_size     = 8
model          = DeepLabV3(model_name, project_dir = os.getcwd()).to(device)
model.load_state_dict(torch.load(args.model_path))
print(model)
test_datasets1 = DATASET_CITYSCAPES(cityscapes_path = 'data/cityscapes', split = 'val')
test_datasets2 = DATASET_CITYSCAPES_FOGGY(cityscapes_path = 'data/cityscapes', split = 'val', fog_scale = args.fog_scale)
test_loader1   = torch.utils.data.DataLoader(dataset = test_datasets1, batch_size = batch_size, shuffle = False, num_workers = 4)
test_loader2   = torch.utils.data.DataLoader(dataset = test_datasets2, batch_size = batch_size, shuffle = False, num_workers = 4)

# model.to(device)
model.eval()

conf_mat_unperturbed = np.zeros((20,20)).astype(int)
conf_mat_foggy       = np.zeros((20,20)).astype(int)

if args.compute_unperturbed:
    with torch.no_grad():
        for num_batch, (img, lbl_img) in enumerate(test_loader1):
            img         = Variable(img).to(device)
            output      = model(img)
            output      = F.softmax(output, dim = 1)
            output      = torch.argmax(output, dim = 1)
            output      = output.detach().cpu().numpy()
            lbl_img     = lbl_img.cpu().numpy()
            # print(output.shape)
            for i in range(output.shape[0]):
                xx = lbl_img[i].flatten()
                yy = output[i].flatten()
                for x, y in zip(xx, yy):
                    conf_mat_unperturbed[x, y] += 1
            print("Confusion matrix, unperturbed data")
            print(conf_mat_unperturbed)


    print(conf_mat_unperturbed)

with torch.no_grad():
    for num_batch, (img, lbl_img) in enumerate(test_loader2):
        img         = Variable(img).to(device)
        output      = model(img)
        output      = F.softmax(output, dim = 1)
        output      = torch.argmax(output, dim = 1)
        output      = output.detach().cpu().numpy()
        lbl_img     = lbl_img.cpu().numpy()
        # print(output.shape)
        for i in range(output.shape[0]):
            xx = lbl_img[i].flatten()
            yy = output[i].flatten()
            for x, y in zip(xx, yy):
                conf_mat_foggy[x, y] += 1
        print("Confusion matrix, foggy data")
        print(conf_mat_foggy)

print(conf_mat_foggy)

try:
    os.mkdir('metrics/confusion_matrix/')
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

if args.compute_unperturbed:
    np.savetxt('metrics/confusion_matrix/conf_mat_unperturbed.txt', conf_mat_unperturbed, delimiter=',')
np.savetxt('metrics/confusion_matrix/conf_mat_foggy_' + str(args.fog_scale) +'.txt', conf_mat_foggy, delimiter=',')
