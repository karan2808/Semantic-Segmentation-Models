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


parser = argparse.ArgumentParser(description='Compute Confusion Matrix')
print(os.getcwd())
parser.add_argument('--model_path', type=str, default= os.getcwd() + '/pretrained_models/model_13_2_2_2_epoch_580.pth')
parser.add_argument('--fog_scale', type=float, default=0.005)
parser.add_argument('--dataset_path', type=str, default='data/cityscapes')
parser.add_argument('--compute_unperturbed', action='store_true')
args = parser.parse_args()

batch_size      = 4
device          = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name      = '0'
learning_rate   = 0.005
num_epochs      = 20
model           = DeepLabV3(model_name, project_dir = os.getcwd()).to(device)
train_datasets  = DATASET_CITYSCAPES_FOGGY(cityscapes_path = os.getcwd() + '/data/cityscapes', split = 'train', fog_scale = 0.01)
val_datasets    = DATASET_CITYSCAPES_FOGGY(cityscapes_path = os.getcwd() + '/data/cityscapes', split = 'val', fog_scale = 0.01)

train_loader    = torch.utils.data.DataLoader(dataset = train_datasets, batch_size = batch_size, shuffle = True, num_workers = 1)
val_loader      = torch.utils.data.DataLoader(dataset = val_datasets, batch_size = batch_size, shuffle = False, num_workers = 1)

optimizer       = torch.optim.Adam(model.parameters(), lr = learning_rate)
loss_function   = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    for num_iter, (img, lbl_img) in enumerate(train_loader):
        optimizer.zero_grad()
        
        img         = Variable(img).to(device)
        lbl_img     = Variable(lbl_img.type(torch.LongTensor)).to(device)
        output      = model(img)

        # compute the loss 
        loss        = loss_function(output, lbl_img)
        print(loss.item())
        loss.backward()
        optimizer.step()

