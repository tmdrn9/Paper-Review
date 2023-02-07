import torch
import torch.nn as nn
import timm


class feature_extractor(nn.Module):
    def __init__(self):
        super(feature_extractor, self).__init__()
        self.backbone = timm.create_model('vgg16', pretrained = True)

        self.block123 = self.backbone.features[:17]
        self.block4 = self.backbone.features[17:23]
        self.block5 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=2),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=2),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=2),
                                    nn.ReLU(inplace=True))

    def forward(self, x):
        f3 = self.block123(x)
        f4 = self.block4(f3)
        f5 = self.block5(f4)

        return f3, f4, f5


class SCSA(nn.Module):  # Self-Correlation with Spatial Attention
    def __init__(self, in_ch=512, T=48):
        super(SCSA, self).__init__()
        self.in_ch = in_ch
        self.transform_ch = self.in_ch // 8
        self.T = T
        self.f_transform = nn.Conv2d(self.in_ch, self.transform_ch, kernel_size=(1, 1), stride=(1, 1))
        self.g_transform = nn.Conv2d(self.in_ch, self.transform_ch, kernel_size=(1, 1), stride=(1, 1))
        self.h_transform = nn.Conv2d(self.in_ch, self.in_ch, kernel_size=(1, 1), stride=(1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.scale = nn.Parameter(torch.tensor(0.))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, f_):
        shape_b, shape_c, shape_h, shape_w = f_.shape

        # 논문상에서 shape이 h,w,c이라서 바꿔줌
        f_ = (f_ / torch.linalg.matrix_norm(f_,dim=(0,1)).unsqueeze(dim=0)) # 512,64,64 -> 64,64,512
        f = self.f_transform(f_)
        f = f.permute(0, 2, 3, 1)  # 32,64,64 -> 64,64,32
        g = self.g_transform(f_)
        g = g.permute(0, 2, 3, 1)  # 32,64,64 -> 64,64,32
        h = self.h_transform(f_)
        h = h.permute(0, 2, 3, 1)  # 256,64,64 -> 64,64,256
        f_ = f_.permute(0, 2, 3, 1) # 64,64,256

        # attention score distribution
        s = torch.matmul(f.reshape(shape_b,-1,self.transform_ch), g.permute(0,3,2,1).reshape(shape_b,self.transform_ch,-1)) # 64*64,64*64
        beta = self.softmax(s)  # 64*64,64*64

        # Attention Values
        o = torch.matmul(beta, h.reshape(shape_b,-1,self.in_ch))  # 64*64,256
        o = o.reshape(shape_b, shape_h, shape_w, shape_c) # 64,64,256

        fdotdot = self.scale * o + f_  # 64,64,256
        # fdotdot = fdotdot.reshape(shape_b,self.in_ch, -1)  # 512,64^2

        c_loop = torch.empty((shape_b, shape_h, shape_w, shape_h*shape_w)).to('cuda')
        for c in range(shape_b):
            c_loop[c] = torch.matmul(fdotdot[c], fdotdot[c].reshape(self.in_ch, -1))  # 64,64,512 * 512,64^2 = 64,64,64^2
        c_loop = c_loop.sort(dim=3,descending=True).values
        c_loop = c_loop[:, :, :, :self.T]
        c_loop = self.relu(c_loop)
        c_loop = c_loop / torch.linalg.matrix_norm(c_loop,dim=(0,1)).unsqueeze(dim=0)

        return c_loop.permute(0,3,1,2)


class ASSP(nn.Module):
    def __init__(self, h=512//8, w=512//8, T=48, out_ch=48):  # in_ch확인
        super(ASSP, self).__init__()
        self.T = T
        self.h = h
        self.w = w
        self.in_ch = self.T*3
        self.out_ch = out_ch
        self.conv1_48 = nn.Sequential(nn.Conv2d(self.in_ch, self.out_ch, kernel_size=(1, 1), stride=(1, 1)),  # 확인
                                      nn.ReLU(inplace=True))

        self.atrousconv3_48_6 = nn.Sequential(
            nn.Conv2d(self.in_ch, self.out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(6,6), dilation=6),
            nn.ReLU(inplace=True))

        self.atrousconv3_48_12 = nn.Sequential(
            nn.Conv2d(self.in_ch, self.out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(12,12), dilation=12),
            nn.ReLU(inplace=True))

        self.atrousconv3_48_18 = nn.Sequential(
            nn.Conv2d(self.in_ch, self.out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(18,18), dilation=18),
            nn.ReLU(inplace=True))

        self.avgpool_conv1_48 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                              nn.Conv2d(self.in_ch, self.out_ch, kernel_size=(3, 3), stride=(1, 1),
                                                        padding=(1, 1)),
                                              nn.ReLU(inplace=True))
        self.upsample = nn.Upsample(size=[self.h, self.w], mode="bilinear")

    def forward(self, x):

        out_1x1 = self.conv1_48(x)
        out_3x3_1 = self.atrousconv3_48_6(x)
        out_3x3_2 = self.atrousconv3_48_12(x)
        out_3x3_3 = self.atrousconv3_48_18(x)
        out_avgpool = self.avgpool_conv1_48(x)
        out_avgpool = self.upsample(out_avgpool)

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_avgpool], 1)

        return out


class feature_upsample(nn.Module):
    def __init__(self, in_ch=240, out_ch=1, hidden_ch=96):
        super(feature_upsample, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(in_ch, hidden_ch, kernel_size=(1, 1), stride=(1, 1)),
                                    nn.ReLU(inplace=True),
                                    nn.Upsample(scale_factor=2))

        self.block2 = nn.Sequential(nn.Conv2d(hidden_ch, in_ch, kernel_size=(1, 1), stride=(1, 1)),
                                    nn.ReLU(inplace=True),
                                    nn.Upsample(scale_factor=2))

        self.block3 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_ch, in_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), stride=(1, 1)),
                                    nn.ReLU(inplace=True),
                                    nn.Upsample(scale_factor=2))

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)

        return out


class self_deep_matching(nn.Module):
    def __init__(self):
        super(self_deep_matching, self).__init__()
        self.feature_extractor = feature_extractor()
        self.SC_SA1 = SCSA(in_ch=256)
        self.SC_SA2 = SCSA()
        self.SC_SA3 = SCSA()
        self.ASSP = ASSP()
        self.feature_upsample = feature_upsample()

    def forward(self, x):
        f3, f4, f5 = self.feature_extractor(x)
        f3 = self.SC_SA1(f3)
        f4 = self.SC_SA2(f4)
        f5 = self.SC_SA3(f5)
        out = torch.cat([f3, f4, f5], 1)
        out = self.ASSP(out)
        out = self.feature_upsample(out)

        return out
