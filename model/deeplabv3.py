import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from resnet import ResNet18, ResNet34
from aspp import ASPP, ASPP_Bottleneck

class DeepLabV3(nn.Module):
    def __init__(self, model_name, project_dir):
        super(DeepLabV3, self).__init__()
        self.num_classes   = 20
        self.model_name    = model_name
        self.project_dir   = project_dir
        self.create_model_dirs()
        # specify the type of resnet model / or any other model for feature extraction
        self.resnet        = ResNet18(project_dir)
        self.aspp          = ASPP(num_classes=self.num_classes) 

    def forward(self, x):
        # shape of x : b, c, h, w
        h, w = x.size()[2], x.size()[3]
        # batch_size, 512, h/16, w/16  
        # ResNet18, ResNet_34, batch_size, 512, h/8, w/8 
        feature_map = self.resnet(x)
        # batch_size, num_classes, h/16, w/16
        output      = self.aspp(feature_map)
        output      = F.upsample(output, size=(h, w), mode="bilinear")
        return output

    def create_model_dirs(self):
        self.logs_dir        = self.project_dir + "/training_logs"
        self.model_dir       = self.logs_dir + "/model_%s" % self.model_name
        self.checkpoints_dir = self.model_dir + "/checkpoints"

        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)