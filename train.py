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
import matplotlib.pyplot as plt 

batch_size      = 8
device          = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name      = '0'
learning_rate   = 0.001
num_epochs      = 20
fog_scale       = 0.01
model           = DeepLabV3(model_name, project_dir = os.getcwd()).to(device)
train_datasets  = DATASET_CITYSCAPES_FOGGY(cityscapes_path = os.getcwd() + '/data/cityscapes', split = 'train', fog_scale = 0.01)
val_datasets    = DATASET_CITYSCAPES_FOGGY(cityscapes_path = os.getcwd() + '/data/cityscapes', split = 'val', fog_scale = 0.01)

train_loader    = torch.utils.data.DataLoader(dataset = train_datasets, batch_size = batch_size, shuffle = True, num_workers = 1)
val_loader      = torch.utils.data.DataLoader(dataset = val_datasets, batch_size = batch_size, shuffle = False, num_workers = 1)

optimizer       = torch.optim.Adam(model.parameters(), lr = learning_rate)
loss_function   = nn.CrossEntropyLoss()

train_log_frequency         = 50
validation_log_frequency    = 50


train_loss  = []
valid_loss  = []


def validate(epoch):
    model.eval()
    running_loss  = 0
    counter       = 0
    for num_iter, (img, lbl_img) in enumerate(val_loader):
        img             = Variable(img).to(device)
        lbl_img         = Variable(lbl_img.type(torch.LongTensor)).to(device)
        output          = model(img)
        loss            = loss_function(output, lbl_img)
        running_loss   += loss.item()
        counter        += batch_size
    
        if num_iter % validation_log_frequency == 0:
            print("Valid Epoch: " + str(epoch) + " Valid Loss: " + str(running_loss / counter)) 
    valid_loss.append(running_loss/counter)

def train(epoch):
    model.train()
    running_loss  = 0
    counter       = 0
    for num_iter, (img, lbl_img) in enumerate(train_loader):
        optimizer.zero_grad()

        img             = Variable(img).to(device)
        lbl_img         = Variable(lbl_img.type(torch.LongTensor)).to(device)
        output          = model(img)

        loss            = loss_function(output, lbl_img)
        loss.backward()
        optimizer.step()
        
        running_loss   += loss.item()
        counter        += batch_size

        if counter % train_log_frequency == 0:
            print("Train Epoch: " + str(epoch) + " Train Loss: " + str(running_loss / counter)) 

    validate(epoch)
    train_loss.append(running_loss/counter)
    
for epoch in range(num_epochs):
    train(epoch)
    torch.save(model.state_dict(), 'saved_models/deeplab_v3_foggy_' + str(epoch) + '.pth')
    plt.plot(train_loss, label = 'train_loss')
    plt.plot(valid_loss, label = 'valid_loss')
    plt.savefig("loss_foggy.png")
    plt.legend()
    plt.show()
