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
from datasets import DatasetTrain, DatasetVal

batch_size      = 4
device          = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name      = '0'
num_epochs      = 20
learning_rate   = 1e-3
model           = DeepLabV3(model_name, project_dir = os.getcwd()).to(device)
train_datasets  = DatasetTrain(cityscapes_path = os.getcwd() + '/data/cityscapes')
val_datasets    = DatasetVal(cityscapes_path = os.getcwd() + '/data/cityscapes')

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
        print(epoch, num_iter, loss.item())
        loss.backward()
        optimizer.step()
