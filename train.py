import logging
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import argparse
from datetime import datetime
from COD import COD
from utils.data_val import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter


def hybrid_e_loss(pred, mask):
    """ Hybrid Eloss """
    # adaptive weighting masks
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    # weighted binary cross entropy loss function
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = ((weit * wbce).sum(dim=(2, 3)) + 1e-8) / (weit.sum(dim=(2, 3)) + 1e-8)

    # weighted e loss function
    pred = torch.sigmoid(pred)
    mpred = pred.mean(dim=(2, 3)).view(pred.shape[0], pred.shape[1], 1, 1).repeat(1, 1, pred.shape[2], pred.shape[3])
    phiFM = pred - mpred

    mmask = mask.mean(dim=(2, 3)).view(mask.shape[0], mask.shape[1], 1, 1).repeat(1, 1, mask.shape[2], mask.shape[3])
    phiGT = mask - mmask

    EFM = (2.0 * phiFM * phiGT + 1e-8) / (phiFM * phiFM + phiGT * phiGT + 1e-8)
    QFM = (1 + EFM) * (1 + EFM) / 4.0
    eloss = 1.0 - QFM.mean(dim=(2, 3))

    # weighted iou loss function
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1.0 - (inter + 1 + 1e-8) / (union - inter + 1 + 1e-8)

    return (wbce + eloss + wiou).mean()


def bce(pred, mask):
    # BCE损失
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    return wbce.mean()


def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    size_rates = [0.75, 1, 1.25]
    loss_e_record, loss_g_record, loss_c_record = AvgMeter(), AvgMeter(), AvgMeter()
    try:
        for i, (images, gts) in enumerate(train_loader, start=1):
            for rate in size_rates:
                optimizer.zero_grad()
                # ---- data prepare ----
                images = images.cuda()
                gts = gts.cuda()
                # ---- rescale ----
                trainsize = int(round(opt.trainsize * rate / 32) * 32)
                if rate != 1:
                    images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                # ---- forward ----
                edge, glo, cam_out_2, cam_out_3 = model(images)
                # ---- loss function ----
                loss_e = bce(edge, gts)
                loss_g = hybrid_e_loss(glo, gts)
                loss_c = hybrid_e_loss(cam_out_2, gts) + hybrid_e_loss(cam_out_3, gts)
                loss = loss_e + loss_g + loss_c
                # ---- backward ----
                loss.backward()
                clip_gradient(optimizer, opt.clip)
                optimizer.step()
                step += 1
                epoch_step += 1
                loss_all += loss.data
                # ---- recording loss ----
                if rate == 1:
                    loss_e_record.update(loss_e.data, opt.batchsize)
                    loss_g_record.update(loss_g.data, opt.batchsize)
                    loss_c_record.update(loss_c.data, opt.batchsize)
            # ---- train visualization ----
            if i % 20 == 0 or i == total_step:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                      '[loss_e: {:0.4f}, loss_g: {:0.4f}, loss_c: {:0.4f}]'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step,
                             loss_e_record.show(), loss_g_record.show(), loss_c_record.show()))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss_e: {:0.4f}, loss_g: {:0.4f}, '
                    'loss_c: {:0.4f}]'.
                    format(epoch, opt.epoch, i, total_step, loss_e_record.show(), loss_g_record.show(),
                           loss_c_record.show()))
        loss_all /= epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        if epoch % 10 == 0:
            torch.save(model.state_dict(), save_path + 'COD_{}.pth'.format(epoch))
            print('[Saving Snapshot:]', save_path + 'COD_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'COD_{}.pth'.format(epoch))
        print('Save checkpoints successfully!')
        raise


def test(test_loader1, model, epoch, save_path):
    """
        validation function
        """
    global best_mae1, best_epoch1
    model.eval()

    with torch.no_grad():
        mae_sum1 = 0
        for i in range(test_loader1.size):
            image, gt, name, img_for_post = test_loader1.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            e_g_out, s_g_out, cam_out_2, cam_out_3 = model(image)
            # 最终预测cam_out_3
            res = F.interpolate(cam_out_3, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum1 += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae1 = mae_sum1 / test_loader1.size
        print('Test: Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae1, best_mae1, best_epoch1))
        if epoch == 1:
            best_mae1 = mae1
        else:
            if mae1 < best_mae1:
                best_mae1 = mae1
                best_epoch1 = epoch
                torch.save(model.state_dict(), save_path + 'Test_best_{}.pth'.format(best_epoch1))
                print('Test: Save state_dict successfully! Best epoch:{}.'.format(best_epoch1))
        logging.info(
            'Test: [Val Info]:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae1, best_epoch1, best_mae1))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=80, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='choosing optimizer AdamW or SGD')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=416, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')

    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')

    parser.add_argument('--train_root', type=str,
                        default='./Dataset/TrainDataset/',
                        help='path to train dataset')
    parser.add_argument('--test_root', type=str,
                        default='./Dataset/TestDataset/CAMO/',
                        help='the test rgb images root')
    parser.add_argument('--save_root', type=str,
                        default='./model_pth/')

    opt = parser.parse_args()

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    cudnn.benchmark = True

    # build the model
    model = COD(channel=64).cuda()

    ## optimizer
    params = model.parameters()
    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)
    print(optimizer)

    save_path = opt.save_root
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader = get_loader(
        image_root=opt.train_root + 'Imgs/',
        gt_root=opt.train_root + 'GT/',
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        shuffle=True,
        num_workers=16
    )
    val_loader = test_dataset(image_root=opt.test_root + 'Imgs/',
                               gt_root=opt.test_root + 'GT/',
                               testsize=opt.trainsize)
    total_step = len(train_loader)

    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    logging.info(
        'Config: epoch: {}; lr: {}; optimizer: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; decay_epoch: {};  save_root: {} '.
        format(opt.epoch, opt.lr, opt.optimizer, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate,
               opt.decay_epoch, save_path))

    step = 0
    best_mae1 = 1
    best_epoch1 = 0

    print("Start Training...")

    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)

        train(train_loader, model, optimizer, epoch, save_path)
        test(val_loader, model, epoch, save_path)

    os.system("shutdown")
