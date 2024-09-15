from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision

# unetpp原始论文：
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


# class VGGBlock(nn.Module):
#     def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True)):
#         super(VGGBlock, self).__init__()
#         self.act_func = act_func
#         self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
#         self.bn1 = nn.BatchNorm2d(middle_channels)
#         self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.act_func(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.act_func(out)
#         return out

class UNetPP(nn.Module):
    def __init__(self, in_channel, out_channel, deepsupervision):
        """
                初始化UNetPP模型。
                :param in_channel: 输入图像的通道数。
                :param out_channel: 输出图像的通道数。
                :param deepsupervision: 是否使用深度监督。如果为True，则在不同层级上产生输出。
        """
        super().__init__()

        # self.args = args
        # 深度监督的标志
        self.deepsupervision = deepsupervision
        # 定义每个卷积层中的滤波器数量
        nb_filter = [32, 64, 128, 256, 512]
        # 定义最大池化层，用于下采样
        self.pool = nn.MaxPool2d(2, 2)
        # 定义上采样层，用于上采样
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 定义编码器部分的卷积层
        self.conv0_0 = DoubleConv(in_channel, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])
        # 定义解码器部分的卷积层，同时融合编码器的特征图
        self.conv0_1 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = DoubleConv(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = DoubleConv(nb_filter[3] + nb_filter[4], nb_filter[3])
        # 定义更多的解码器卷积层，用于更深层次的特征融合
        self.conv0_2 = DoubleConv(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = DoubleConv(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = DoubleConv(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])

        self.conv0_3 = DoubleConv(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = DoubleConv(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])

        self.conv0_4 = DoubleConv(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])
        # 定义Sigmoid激活函数，用于将输出映射到[0, 1]范围
        self.sigmoid = nn.Sigmoid()
        # 根据是否使用深度监督来定义输出层
        if self.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
        else:
            # 只在最后一个层级上产生输出
            self.final = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)

    # 前向传播函数，根据上面初始化好的组件进行前向传播的操作
    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deepsupervision:
            output1 = self.final1(x0_1)
            output1 = self.sigmoid(output1)
            output2 = self.final2(x0_2)
            output2 = self.sigmoid(output2)
            output3 = self.final3(x0_3)
            output3 = self.sigmoid(output3)
            output4 = self.final4(x0_4)
            output4 = self.sigmoid(output4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            output = self.sigmoid(output)
            return output
