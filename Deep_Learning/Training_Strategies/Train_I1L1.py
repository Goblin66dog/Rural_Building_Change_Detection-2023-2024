import random

import cv2
import numpy as np
import torch.utils.data
from sklearn.metrics import recall_score, precision_score
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from Deep_Learning.Data_Readers.I1L1 import DataLoader
from Deep_Learning.Models.AG_UNet.model import AGUNet
# from Deep_Learning.Models.Segformer_UNet_Simplifier.model import SegFormer
# from Deep_Learning.Models.UNet.model import UNet
from Deep_Learning.Models.SegFormer_OutConv.model import SegFormer
from Deep_Learning.Models.DeepLab_V3_Plus.model import DeepLab
from Deep_Learning.Models.ASPP_U2Net.model import ASPPU2Net

import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

# backbone_lr = 0.0001*0.01
def train(device, epochs=20, batch_size=1, lr=1e-3,step_size_up=2):
    #todo:
    net = SegFormer(num_classes=1,pretrained=False,in_channel=3,phi="b3")
    # net = ASPPU2Net(image_channels=4, texture_channels=1,classes=1)
########################################################################################################################
    # pretrained_dict = torch.load(r"D:\Github_Repo\logs\SegFormer\model.pth",
    #                              map_location=device)
    # model_dict = net.state_dict()
    # a = {}
    # for k, v in pretrained_dict.items():
    #     try:
    #         if np.shape(model_dict[k]) == np.shape(v):
    #             a[k] = v
    #     except:
    #         pass
    # model_dict.update(a)
    # net.load_state_dict(model_dict)
    # print('Finished!')

########################################################################################################################
    net.to(device=local_device)
    writer = {
        "loss": SummaryWriter(r"logs\loss"),
        'recall': SummaryWriter(r"logs\recall"),
        'precision': SummaryWriter(r"logs\precision"),
        "label": SummaryWriter(r"logs\label"),
         "pred": SummaryWriter(r"logs\pred"),
        "lr": SummaryWriter(r"logs" + r"\lr")

    }
    TrainLoader = DataLoader(r"D:\CUMT Datasets\Train_Final")
    ValidLoader = DataLoader(r"D:\CUMT Datasets\Valid_Final")

    #优化器
    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.AdamW(net.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                  mode='triangular2',
                                                  max_lr=lr,
                                                  base_lr=1e-7,
                                                  step_size_up=step_size_up,
                                                  step_size_down=1,
                                                  cycle_momentum=False)
    # scheduler =  torch.optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=0.5)
    #损失函数
    # todo:change the loss function
    best_loss = float("inf")
    net.train()
    #训练epochs
    step = 0
    for epoch in range(epochs):
########################################################################################################################
        # if epoch > epochs // 2:
        #     batch_size = 1
        train_data = torch.utils.data.DataLoader(
            dataset=TrainLoader,
            batch_size=batch_size,
            shuffle=True
        )
        # todo:
        if epoch == epochs // 2:
            for param in net.backbone.parameters():
                param.requires_grad = False
            for param in net.decode_head.parameters():
                param.requires_grad = False
        if epoch < step_size_up:
            optimizer.param_groups[0]['lr'] = lr
        if epoch > epochs * 8 //9 and optimizer.param_groups[0]['lr'] > 1e-6:
            optimizer.param_groups[0]['lr'] = 1e-6
    ########################################################################################################################
        for image, label in train_data:
            optimizer.zero_grad()

            # 将数据拷贝到device中
            image   = image.to(device=device, dtype=torch.float32)
            label   = label.to(device=device, dtype=torch.float32)

            #预测图像
            pred = net(image)
            # writer["pred"].add_images("pred", texture, step)
            # writer["label"].add_images("label", label, step)

            # todo:change the loss function
            L1 = nn.BCEWithLogitsLoss()
            loss = L1(pred, label)
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), r'model.pth')
            # 更新参数
            loss.backward()
            optimizer.step()
            step += 1
            if step % 500 == 0:
                net.eval()
                with torch.no_grad():
                    num_choose = random.randint(0, len(ValidLoader) - 1)
                    val_image, val_label = ValidLoader[num_choose]
                    val_image = val_image.to(device=device, dtype=torch.float32)
                    val_label   = val_label.to(device=device, dtype=torch.float32)

                    val_image   = val_image.reshape(1, 3, val_image.shape[1], val_image.shape[2])
                    val_label   = val_label.reshape(1, 1, val_label.shape[1], val_label.shape[2])

                    val_pred = net(val_image)


                    val_pred = torch.sigmoid(val_pred)

                    val_pred[val_pred >  0.5] = 1
                    val_pred[val_pred <= 0.5] = 0

                    val_pred = np.array(val_pred.data.cpu())
                    val_label = np.array(val_label.data.cpu())

                    val_pred[np.isnan(val_pred)] = 0


                    writer["pred"].add_images("pred", val_pred, step)
                    writer["label"].add_images("label", val_label, step)

                    val_label = val_label.reshape(-1)
                    val_pred = val_pred.reshape(-1)

                    recall = recall_score(val_label, val_pred)
                    precision = precision_score(val_label, val_pred)
                net.train()
                writer['recall'].add_scalar("data", recall, step)
                writer['precision'].add_scalar("data", precision, step)
            writer['loss'].add_scalar("data", loss, step)
            writer['lr'].add_scalar("data", optimizer.state_dict()['param_groups'][0]['lr'], step)
        scheduler.step()

    writer["loss"].close()
    writer["recall"].close()
    writer["precision"].close()
    writer["label"].close()
    writer["pred"].close()
    writer["lr"].close()


if __name__=="__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    local_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 选择使用的网络
    # use_net = ("SegFormer_U")
    # if use_net == "U2Net":
    #     local_net = U2Net(image_channels=4, texture_channels=1, classes=1)
    # elif use_net == "ASPPU2Net":
    #     local_net = ASPPU2Net(image_channels=4, texture_channels=1, classes=1)
    #
    # elif use_net == "ESFNet":
    # elif use_net == "UNet":
    #     local_net = UNet(channels=3, classes=1)
    # else:
    #     local_net = SegFormer(num_classes=1, phi="b3",in_channel=5)
    # 指定训练集地址，开始训练
    train(local_device)