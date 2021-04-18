import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.models as models

def make_layer(block, in_ch, out_ch, num_blocks, stride = 1, dilation = 1):
    # the stride for the rest of the blocks should be 1
    strides = [stride] + [1] * (num_blocks - 1)

    # get the blocks
    blocks = []
    for stride in strides:
        blocks.append(block(in_ch = in_ch, out_ch = out_ch, stride = stride, dilation = dilation))
        in_ch = block.expansion * out_ch

    layer = nn.Sequential(*blocks)
    return layer

class BasicBlock(nn.Module):
    # amount of expansion required
    expansion = 1

    def __init__(self, in_ch, out_ch, stride = 1, dilation = 1):

        super(BasicBlock, self).__init__()
        out_ch_final = self.expansion * out_ch

        self.conv1   = nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride = stride, padding = dilation, dilation = dilation, bias = False)
        self.bn1     = nn.BatchNorm2d(out_ch)

        self.conv2   = nn.Conv2d(out_ch, out_ch, kernel_size = 3, stride = stride, padding = dilation, dilation = dilation, bias = False)
        self.bn2     = nn.BatchNorm2d(out_ch)

        if (stride != 1) or (in_ch != out_ch_final):
            conv = nn.Conv2d(in_ch, out_ch_final, kernel_size = 1, stride = stride, bias = False)
            bn   = nn.BatchNorm2d(out_ch_final)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out)) + self.downsample(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, out_ch, stride = 1, dilation = 1):
        super(Bottleneck, self).__init__()

        out_ch_final = self.expansion * out_ch

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size = 1, bias = False)
        self.bn1   = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size = 3, stride = stride, padding = dilation, dilation = dilation)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.conv3 = nn.Conv2d(out_ch, out_ch_final, kernel_size = 1, bias = False)
        self.bn3   = nn.BatchNorm2d(out_ch_final)

        if (in_ch != out_ch_final) or (stride != 1):
            conv = nn.Conv2d(in_ch, out_ch_final, kernel_size = 1, stride = stride, bias = False)
            bn   = nn.BatchNorm2d(out_ch_final)
            self.downsample = nn.Sequential(conv, bn)
        
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out)) + self.downsample(x)
        out = F.relu(out)
        return out

class ResNet_BasicBlock(nn.Module):
    
    def __init__(self, num_layers, directory):
        super(ResNet_BasicBlock, self).__init__()

        if num_layers == 18:
            resnet = models.resnet18()
            # load pretrained model:
            resnet.load_state_dict(torch.load(directory + "/pretrained_models/resnet/resnet18-5c106cde.pth"))
            # remove the last 4 layers, fc, avg pool, conv4, conv5:
            self.resnet = nn.Sequential(*list(resnet.children())[:-4])
            num_blocks_layer_4 = 2
            num_blocks_layer_5 = 2
            print ("pretrained resnet, 18")
        
        elif num_layers == 34:
            resnet = models.resnet34()
            # load pretrained model:
            resnet.load_state_dict(torch.load(directory + "/pretrained_models/resnet/resnet34-333f7ec4.pth"))
            # remove the last 4 layers, fc, avg pool, conv4, conv5:
            self.resnet = nn.Sequential(*list(resnet.children())[:-4])
            num_blocks_layer_4 = 6
            num_blocks_layer_5 = 3
            print ("pretrained resnet, 34")
        else:
            raise Exception("num_layers must be in {18, 34}!")

        self.layer4 = make_layer(BasicBlock, in_ch=128, out_ch=256, num_blocks=num_blocks_layer_4, stride=1, dilation=2)

        self.layer5 = make_layer(BasicBlock, in_ch=256, out_ch=512, num_blocks=num_blocks_layer_5, stride=1, dilation=4)

    def forward(self, x):
        # print(x.shape)
        # x : (batch_size, 3, h, w)
        output = self.resnet(x) # batch_size, 128, h/8, w/8)
        output = self.layer4(output) # batch_size, 256, h/8, w/8
        output = self.layer5(output) # batch_size, 512, h/8, w/8
        # print(output.shape)
        return output

def ResNet18(project_dir = None):
    return ResNet_BasicBlock(num_layers=18, directory = project_dir)

def ResNet34(project_dir = None):
    return ResNet_BasicBlock(num_layers=34, directory = project_dir)
