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
sys.path.append('model/')
sys.path.append('utils/')
from deeplabv3 import DeepLabV3
from datasets import DATASET_CITYSCAPES, DATASET_CITYSCAPES_FOGGY

device         = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name     = '0'
batch_size     = 8
model          = DeepLabV3(model_name, project_dir = os.getcwd()).to(device)
model.load_state_dict(torch.load('pretrained_models/model_13_2_2_2_epoch_580.pth'))
print(model)
test_datasets1 = DATASET_CITYSCAPES(cityscapes_path = os.getcwd() + '/data/cityscapes', split = 'val')
test_datasets2 = DATASET_CITYSCAPES_FOGGY(cityscapes_path = os.getcwd() + '/data/cityscapes', split = 'val', fog_scale = 0.005)
test_loader1   = torch.utils.data.DataLoader(dataset = test_datasets1, batch_size = batch_size, shuffle = False, num_workers = 2)
test_loader2   = torch.utils.data.DataLoader(dataset = test_datasets2, batch_size = batch_size, shuffle = False, num_workers = 2)

# model.to(device)
model.eval()

conf_mat_unperturbed = np.zeros((20,20)).astype(int)
conf_mat_foggy       = np.zeros((20,20)).astype(int)

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

np.savetxt('conf_mat_unperturbed.txt', conf_mat_unperturbed, delimiter=',')
np.savetxt('conf_mat_foggy.txt', conf_mat_foggy, delimiter=',')
