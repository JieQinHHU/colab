import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.PraNet_Res2Net import PraNet
from utils.dataloader import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)  # 5*（mask均值-mask本身）+1
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none') # 二值交叉熵，reduce='none' 表示不对每个样本的损失求和，保留每个像素点的损失。
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3)) #(2,3)宽度和高度 二值交叉熵x权重

    pred = torch.sigmoid(pred)   # 非线性激活的预测结果
    inter = ((pred * mask)*weit).sum(dim=(2, 3))  #预测结果和真实值的交集
    union = ((pred + mask)*weit).sum(dim=(2, 3))  #预测结果和真实标注的并集
    wiou = 1 - (inter + 1)/(union - inter+1)      #像素级联合加权交并比
    return (wbce + wiou).mean() #两种损失值的均值


def train(train_loader, model, optimizer, epoch):
    model.train() #指示处于训练模式
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25] # 三种图像缩放比例
    loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter() # utils的函数
    for i, pack in enumerate(train_loader, start=1):#pack应该是train_loader中的元素，enumerate是python自带遍历函数
        for rate in size_rates:
            optimizer.zero_grad()     # 开始优化
            # ---- data prepare ----
            images, gts = pack #读取图片和mask
            images = Variable(images).cuda()  #把图像读入cuda，pytoch函数
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True) # F是python模型，提高大量函数，上采样 双线性插值
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(images)
            # ---- loss function ----
            loss5 = structure_loss(lateral_map_5, gts) #计算损失函数
            loss4 = structure_loss(lateral_map_4, gts)
            loss3 = structure_loss(lateral_map_3, gts)
            loss2 = structure_loss(lateral_map_2, gts)
            loss = loss2 + loss3 + loss4 + loss5    # TODO: try different weights for loss
            # ---- backward ----
            loss.backward() #反向传播
            clip_gradient(optimizer, opt.clip)  # 修剪防止梯度消失或爆炸
            optimizer.step() #调用优化器的 step（） 方法来更新参数。优化器实现的优化算法（例如，Adam，SGD等）使用这些梯度来更新模型的权重，将它们沿最小化损失函数的方向移动。
            # ---- recording loss ----
            if rate == 1:
                loss_record2.update(loss2.data, opt.batchsize)
                loss_record3.update(loss3.data, opt.batchsize)
                loss_record4.update(loss4.data, opt.batchsize)
                loss_record5.update(loss5.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step: # 20取模
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()))
    save_path = 'snapshots/{}/'.format(opt.train_save) #大括号{}为待填参数，就是后面的format字符串。例如，如果opt.train_save为“PraNet_Res2Net”，则save_path将为“snapshots/PraNet_Res2Net/” 
    os.makedirs(save_path, exist_ok=True)
    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), save_path + 'PraNet-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'PraNet-%d.pth'% epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=20, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,
                        default='./data/TrainDataset', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='PraNet_Res2Net')
    opt = parser.parse_args()

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = PraNet().cuda()

    # ---- flops and params ----
    # from utils.utils import CalParams
    # x = torch.randn(1, 3, 352, 352).cuda()
    # CalParams(lib, x)

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = '{}/image/'.format(opt.train_path)
    gt_root = '{}/mask/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch)
