# -*-coding:utf-8 -*-

from model.unet_model import UNet
from utils.dataset import Dateset_Loader, Dateset_Loader_RGB
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from model.unetPP import UNetPP


def train_net(net, device, data_path, epochs=40, batch_size=2, lr=0.00001):
    '''

    :param net: 语义分割网络
    :param device: 网络训练所使用的设备
    :param data_path: 数据集的路径
    :param epochs: 训练的轮数
    :param batch_size: 批次大小
    :param lr: 学习率
    :return:
    '''
    # 加载数据集
    dataset = Dateset_Loader(data_path)
    per_epoch_num = len(dataset) / batch_size
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    # 定义RMSprop算法，RMSProp算法（Root Mean Square Propagation）是一种自适应学习率的优化算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-08,
    #                        amsgrad=False)  # 此处的优化算法可以修改为adam算法
    # 定义Loss算法，此处使用的损失函数为二进制交叉熵损失函数
    criterion = nn.BCEWithLogitsLoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 开始训练
    loss_record = []
    with tqdm(total=epochs * per_epoch_num) as pbar:
        for epoch in range(epochs):
            # 训练模式
            net.train()
            # 按照batch_size开始训练
            for image, label in train_loader:
                optimizer.zero_grad()
                # 将数据拷贝到device中
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                preds = net(image)
                # if args.deepsupervision:
                #                 outputs = model(inputs)
                # 注意 unet加加此处应该执行多尺度的损失计算
                loss = 0
                for output in preds:
                    loss += criterion(output, label)
                loss /= len(preds)
                # loss = criterion(pred[0], label) + criterion(pred[1], label)  + criterion(pred[2], label)  + criterion(pred[3], label)
                # 预测过程中的输出需要转化为四个维度的输出，预测的过程中只取最后一个环节的
                pbar.set_description("Processing Epoch: {} Loss: {}".format(epoch + 1, loss))
                # 如果当前的损失比最好的损失小，则保存当前论次的模型
                if loss < best_loss:
                    best_loss = loss
                    torch.save(net.state_dict(), 'best_model_unetpp_my.pth')
                loss.backward()
                optimizer.step()
                pbar.update(1)
            # print(loss.item())
            loss_record.append(loss.item())

    # 绘制loss折线图
    plt.figure()
    # 绘制折线图
    plt.plot([i + 1 for i in range(0, len(loss_record))], loss_record)
    # 添加标题和轴标签
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('results/unetpp/training_loss.png')


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda')
    # 设置输出的通道和输出的类别数目，这里的1表示执行的是二分类的任务
    net = UNetPP(1, 1, True).train()
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path = "../TongeImageDataset"  # todo 或者使用相对路径也是可以的
    print("进度条出现卡着不动不是程序问题，是他正在计算，请耐心等待")
    time.sleep(1)
    train_net(net, device, data_path, epochs=80, batch_size=1)  # 开始训练，如果你GPU的显存小于4G，这里只能使用CPU来进行训练。
