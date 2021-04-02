import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, num_classes):
        super(ASPP, self).__init__()

        self.conv_1x1_1      = nn.Conv2d(512, 256, kernel_size = 1)
        self.bn_conv_1x1_1   = nn.BatchNorm2d(256)

        self.conv_3x3_1      = nn.Conv2d(512, 256, kernel_size = 3, stride = 1, padding = 6, dilation = 6)
        self.bn_conv_3x3_1   = nn.BatchNorm2d(256)

        self.conv_3x3_2      = nn.Conv2d(512, 256, kernel_size = 3, stride = 1, padding = 12, dilation = 12)
        self.bn_conv_3x3_2   = nn.BatchNorm2d(256)

        self.conv_3x3_3      = nn.Conv2d(512, 256, kernel_size = 3, stride = 1, padding = 18, dilation = 18)
        self.bn_conv_3x3_3   = nn.BatchNorm2d(256)

        self.avg_pool        = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2      = nn.Conv2d(512, 256, kernel_size = 1)
        self.bn_conv_1x1_2   = nn.BatchNorm2d(256)

        self.conv_1x1_3      = nn.Conv2d(1280, 256, kernel_size = 1)
        self.bn_conv_1x1_3   = nn.BatchNorm2d(256)

        self.conv_1x1_4      = nn.Conv2d(256, num_classes, kernel_size = 1)

    def forward(self, feature_map):
        # feature map shape: batch_size, 512, h/16, w/16
        feature_map_h = feature_map.size()[2]
        feature_map_w = feature_map.size()[3]

        out_1x1       = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))) # batch_size, 256, h/16, w/16
        out_3x3_1     = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map))) # batch_size, 256, h/16, w/16
        out_3x3_2     = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map))) # batch_size, 256, h/16, w/16
        out_3x3_3     = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map))) # batch_size, 256, h/16, w/16

        output        = self.avg_pool(feature_map) # shape b, 512, 1, 1
        output        = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(output))) # shape 256, 1, 1
        output        = F.upsample(output, size = (feature_map_h, feature_map_w), mode = 'bilinear')

        output        = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, output], 1)
        output        = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(output)))
        output        = self.conv_1x1_4(output)

        return output

class ASPP_Bottleneck(nn.Module):
    def __init__(self, num_classes):
        super(ASPP_Bottleneck, self).__init__()

        self.conv_1x1_1    = nn.Conv2d(4*512, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_3x3_1    = nn.Conv2d(4*512, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)

        self.conv_3x3_2    = nn.Conv2d(4*512, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)

        self.conv_3x3_3    = nn.Conv2d(4*512, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)

        self.avg_pool      = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2    = nn.Conv2d(4*512, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        self.conv_1x1_3    = nn.Conv2d(1280, 256, kernel_size=1) # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)

        self.conv_1x1_4    = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, feature_map):
        # feature_map : batch_size, 4*512, h/16, w/16

        feature_map_h = feature_map.size()[2]
        feature_map_w = feature_map.size()[3]

        out_1x1   = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))) # b, 256, h/16, w/16
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map))) # b, 256, h/16, w/16
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map))) # b, 256, h/16, w/16
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map))) # b, 256, h/16, w/16

        out_img = self.avg_pool(feature_map) # b, 512, 1, 1
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) # b, 256, 1, 1
        out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") # b, 256, h/16, w/16

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) # b, 1280, h/16, w/16
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) # b, 256, h/16, w/16
        out = self.conv_1x1_4(out) # b, num_classes, h/16, w/16

        return out