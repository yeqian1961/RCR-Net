import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b5


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class DFE(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DFE, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, kernel_size=1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, kernel_size=3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, kernel_size=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x0 + self.conv_res(x))
        x2 = self.branch2(x1 + self.conv_res(x))
        x3 = self.branch3(x2 + self.conv_res(x))

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        # 最后relu激活（之前不激活，conv+bn）
        x = self.relu(x_cat + self.conv_res(x))
        return x


class CA(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CA, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.relu1 = nn.ReLU()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu1(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class GAB(nn.Module):
    def __init__(self, channel, subchannel):
        super(GAB, self).__init__()
        self.group = channel // subchannel
        self.conv = nn.Sequential(
            nn.Conv2d(channel + self.group, channel, 3, padding=1), nn.ReLU(True),
        )

    # 通过组引导操作将候选特征pki和rk1（对应x和y）组合起来，然后使用残差阶段来生成细化特征pki+1
    def forward(self, x, y):
        xs = torch.chunk(x, self.group, dim=1)
        x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
                               xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y,
                               xs[16], y, xs[17], y, xs[18], y, xs[19], y, xs[20], y, xs[21], y, xs[22], y, xs[23], y,
                               xs[24], y, xs[25], y, xs[26], y, xs[27], y, xs[28], y, xs[29], y, xs[30], y, xs[31], y),
                              1)
        x = x + self.conv(x_cat)
        return x


class MSFA(nn.Module):
    def __init__(self, feature_channel, intern_channel):
        super(MSFA, self).__init__()
        self.feature_channel = feature_channel
        self.conv1 = BasicConv2d(feature_channel + intern_channel, intern_channel, kernel_size=3, padding=1)
        self.out = nn.Conv2d(intern_channel, 1, kernel_size=3, padding=1)
        self.gab = GAB(feature_channel, feature_channel // 32)

    # features: Semantic Prior # 320,H/16,W/16
    # cam_guidance:  Global/Neighbour Prior     # 1,H/32,W/32
    # edge_guidance: Edge Prior        # 64,H/4,W/4
    def forward(self, features, cam_guidance, edge_guidance):
        # 将 Global/Neighbour Prior 和 Edge Prior 大小调整到 Semantic Prior大小，通道不变
        crop_sal = F.interpolate(cam_guidance, size=features.size()[2:], mode='bilinear',
                                 align_corners=True)  # Global/Neighbour Prior 1,H/16,W/16
        crop_edge = F.interpolate(edge_guidance, size=features.size()[2:], mode='bilinear',
                                  align_corners=True)  # Edge Prior 96,H/16,W/16
        sal_r = -1 * (torch.sigmoid(crop_sal)) + 1
        x_r = self.gab(features, sal_r)  # 384,H/16,W/16
        x = self.conv1(torch.cat((x_r, crop_edge), dim=1))  # 64,H/16,W/16
        x = self.out(F.relu(x))
        x = x + crop_sal  # 论文里没提
        return x


class COD(nn.Module):
    def __init__(self, channel=64):
        super(COD, self).__init__()
        # pvt
        self.backbone = pvt_v2_b5()  # [64, 128, 320, 512]
        path = './pretrained/pvt_v2_b5.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.DFE1 = DFE(512, channel)

        self.CA = CA(channel, channel)

        self.out_DFE = nn.Conv2d(channel, 1, 1)
        self.out_GAM = nn.Conv2d(channel, 1, 1)

        self.msfa_2 = MSFA(feature_channel=320, intern_channel=channel)
        self.msfa_3 = MSFA(feature_channel=128, intern_channel=channel)

    def forward(self, x):
        # backbone PVT
        pvt = self.backbone(x)  # [64, 128, 320, 512]
        x4 = pvt[0]  # 64,H/4,W/4
        x3 = pvt[1]  # 128,H/8,W/8
        x2 = pvt[2]  # 320,H/16,W/16
        x1 = pvt[3]  # 512,H/32,W/32

        x1 = self.DFE1(x1)  # 64,H/32,W/32

        # CA edge information e_g 64,H/4,W/4
        e_g = self.CA(x4)  # 64,H/4,W/4
        e_g_out = self.out_GAM(e_g)  # 1,H/4,W/4
        edge = F.interpolate(e_g_out, scale_factor=4, mode='bilinear', align_corners=True)  # Sup-1 (H/4,W/4 -> H,W)

        # TEM global information: s_g_out 1,H/32,W/32
        s_g_out = self.out_DFE(x1)  # 1,H/32,W/32
        glo = F.interpolate(s_g_out, scale_factor=32, mode='bilinear', align_corners=True)  # Sup-4 (H/32,W/32 -> H,W)

        # neighbour prior map: s_2, bs 1,H/16,W/16
        s_2 = self.msfa_2(x2,  # 320,H/16,W/16
                         s_g_out,  # 1,H/32,W/32
                         e_g)  # 64,H/4,W/4
        cam_out_2 = F.interpolate(s_2, scale_factor=16, mode='bilinear', align_corners=True)  # Sup-3 (H/16,W/16 -> H,W)

        # neighbour prior map: s_2, bs 1,H/8,W/8
        # final map: cam_out_2
        s_3 = self.msfa_3(x3,  # 128,H/8,W/8
                         s_2 + F.interpolate(s_g_out, scale_factor=2, mode='bilinear', align_corners=True),
                         # 1,H/16,W/16
                         e_g)  # 64,H/4,W/4
        cam_out_3 = F.interpolate(s_3, scale_factor=8, mode='bilinear', align_corners=True)  # Sup-2 (H/8,W/8 -> H,W)

        return edge, glo, cam_out_2, cam_out_3


if __name__ == '__main__':
    pytorch_model = COD(channel=64).cuda()
    input_tensor = torch.randn(1, 3, 416, 416).cuda()
    edge, glo, cam_out_2, cam_out_3 = pytorch_model(input_tensor)
    print(edge.size(), glo.size(), cam_out_2.size(), cam_out_3.size())



