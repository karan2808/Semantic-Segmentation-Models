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
sys.path.append('datasets/')
from deeplabv3 import DeepLabV3
from cityscapes_dataset import DATASET_CITYSCAPES, DATASET_CITYSCAPES_FOGGY
from utils import convert_lbl2color

batch_size      = 4
device          = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name      = '0'
model           = DeepLabV3(model_name, project_dir = os.getcwd()).to(device)

val_dataset         = DATASET_CITYSCAPES(cityscapes_path = os.getcwd() + '/data/cityscapes', split = 'val')
val_loader          = torch.utils.data.DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = False, num_workers = 8)

val_dataset_05      = DATASET_CITYSCAPES_FOGGY(cityscapes_path = os.getcwd() + '/data/cityscapes', split = 'val', fog_scale = 0.005)
val_loader_05       = torch.utils.data.DataLoader(dataset = val_dataset_05, batch_size = batch_size, shuffle = False, num_workers = 8)

val_dataset_1       = DATASET_CITYSCAPES_FOGGY(cityscapes_path = os.getcwd() + '/data/cityscapes', split = 'val', fog_scale = 0.01)
val_loader_1        = torch.utils.data.DataLoader(dataset = val_dataset_1, batch_size = batch_size, shuffle = False, num_workers = 8)

val_dataset_2       = DATASET_CITYSCAPES_FOGGY(cityscapes_path = os.getcwd() + '/data/cityscapes', split = 'val', fog_scale = 0.02)
val_loader_2        = torch.utils.data.DataLoader(dataset = val_dataset_2, batch_size = batch_size, shuffle = False, num_workers = 8)

model.eval()


counter = 0
with torch.no_grad():
    for num_batch, (img, lbl_img) in enumerate(val_loader):
        # print(counter)
        if counter >= 20:
            break
        img         = Variable(img).to(device)
        output      = model(img)
        output      = F.softmax(output, dim = 1)
        output      = torch.argmax(output, dim = 1)
        output      = output.detach().cpu().numpy()
        lbl_img     = lbl_img.cpu().numpy()
        for i in range(batch_size):
            input_image   = img[i].permute(1, 2, 0).detach().cpu().numpy()
            input_image   = (input_image - np.amin(input_image)) / (np.amax(input_image) - np.amin(input_image))
            input_image   = (input_image * 255).astype(np.int)
            # print(input_image.shape)
            cv2.imwrite('demo/input_image_' + str(counter) + '.png', input_image)
            label_image = convert_lbl2color(lbl_img[i])
            cv2.imwrite('demo/label_image_' + str(counter) + '.png', label_image)
            current_image = convert_lbl2color(output[i])
            cv2.imwrite('demo/unperturbed_img_' + str(counter) + '.png', current_image)
            counter += 1


counter = 0
with torch.no_grad():
    for num_batch, (img, lbl_img) in enumerate(val_loader_05):
        if counter >= 20:
            break
        img         = Variable(img).to(device)
        output      = model(img)
        output      = F.softmax(output, dim = 1)
        output      = torch.argmax(output, dim = 1)
        output      = output.detach().cpu().numpy()
        # print(output)
        # print(output.shape)
        lbl_img     = lbl_img.cpu().numpy()
        # print(lbl_img)
        # print(lbl_img.shape)
        for i in range(batch_size):
            current_image = convert_lbl2color(output[i])
            cv2.imwrite('demo/foggy_005_image_' + str(counter) + '.png', current_image)
            counter += 1

counter = 0
with torch.no_grad():
    for num_batch, (img, lbl_img) in enumerate(val_loader_1):
        if counter >= 20:
            break
        img         = Variable(img).to(device)
        output      = model(img)
        output      = F.softmax(output, dim = 1)
        output      = torch.argmax(output, dim = 1)
        output      = output.detach().cpu().numpy()
        lbl_img     = lbl_img.cpu().numpy()
        for i in range(batch_size):
            current_image = convert_lbl2color(output[i])
            cv2.imwrite('demo/foggy_01_image_' + str(counter) + '.png', current_image)
            counter += 1

counter = 0
with torch.no_grad():
    for num_batch, (img, lbl_img) in enumerate(val_loader_2):
        if counter >= 20:
            break
        img         = Variable(img).to(device)
        output      = model(img)
        output      = F.softmax(output, dim = 1)
        output      = torch.argmax(output, dim = 1)
        output      = output.detach().cpu().numpy()
        lbl_img     = lbl_img.cpu().numpy()
        for i in range(batch_size):
            current_image = convert_lbl2color(output[i])
            cv2.imwrite('demo/foggy_02_image_' + str(counter) + '.png', current_image)
            counter += 1
