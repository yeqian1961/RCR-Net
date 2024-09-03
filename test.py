import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from COD import COD
from utils.data_val import test_dataset
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=416, help='testing size')
    parser.add_argument('--pth_path', type=str, default='./model_pth/best.pth')
    opt = parser.parse_args()

    model = COD()
    model.load_state_dict(torch.load(opt.pth_path))

    model.cuda()
    model.eval()
    for _data_name in ['CAMO', 'COD10K', 'CHAMELEON', 'NC4K']:
        # 测试图
        data_path = './Dataset/TestDataset/{}'.format(_data_name)
        # 保存图
        save_path = './result/{}/'.format(_data_name)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 测试图路径
        image_root = '{}/Imgs/'.format(data_path)
        gt_root = '{}/GT/'.format(data_path)

        test_loader = test_dataset(image_root, gt_root, opt.testsize)

        for i in range(test_loader.size):
            image, gt, name, _ = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)

            image = image.cuda()

            e_g_out, s_g_out, cam_out_2, cam_out_3 = model(image)


            res = F.interpolate(cam_out_3, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            print('> {} - {}'.format(_data_name, name))

            cv2.imwrite(save_path + name, res*255)
        print(_data_name, 'Finish!')
