import torch
from torch import nn,optim
import torch.nn.functional as F


def ConvBNRelu(in_channel, out_channel, kernel_size,padding):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel,
                  kernel_size=kernel_size,
                  stride=1,
                  padding=padding),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
        nn.Dropout2d(p=0.3)
    )

class BaseInception2(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size1,kernel_size2,kernel_size3,kernel_size4,
                 padding1,padding2,padding3,padding4):
        super(BaseInception2, self).__init__()

        self.branch1_conv = ConvBNRelu(in_channel,
                                       out_channel,
                                       kernel_size1,
                                       padding1)

        self.branch2_conv = ConvBNRelu(in_channel,
                                       out_channel,
                                       kernel_size2,
                                       padding2)

        self.branch3_conv = ConvBNRelu(in_channel,
                                       out_channel,
                                       kernel_size3,
                                       padding3)

        self.branch4_conv = ConvBNRelu(in_channel,
                                       out_channel,
                                       kernel_size4,
                                       padding4)


    def forward(self, x):
        out1 = self.branch1_conv(x)
        out2 = self.branch2_conv(x)
        out3 = self.branch3_conv(x)
        out4 = self.branch4_conv(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

class CNN4(nn.Module):
    def __init__(self, num_channels=1,out_channel_list=64):
        super(CNN4, self).__init__()
        self.block0 = nn.Sequential(
            nn.Conv2d(num_channels, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block1 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1,3),kernel_size2=(3,1),kernel_size3=(1,5),kernel_size4=(5,1),
                           padding1=(0,1),padding2=(1,0),padding3=(0,2),padding4=(2,0))
        )
        self.block1_2 = nn.Sequential(
            nn.Conv2d(out_channel_list*4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                           padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
        )
        self.block2_2 = nn.Sequential(
            nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                           padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
        )
        self.block3_2 = nn.Sequential(
            nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block4 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                           padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
        )
        self.block4_2 = nn.Sequential(
            nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(out_channel_list, num_channels, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.block0(x)
        out = self.block1(out)
        out = self.block1_2(out)
        out = self.block2(out)
        out = self.block2_2(out)
        out = self.block3(out)
        out = self.block3_2(out)
        out = self.block4(out)
        out = self.block4_2(out)
        out = self.block5(out)
        return out


class CNN5(nn.Module):
    def __init__(self, num_channels=1,out_channel_list=32):
        super(CNN5, self).__init__()
        self.block0 = nn.Sequential(
            nn.Conv2d(num_channels, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block1 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1,3),kernel_size2=(3,1),kernel_size3=(1,5),kernel_size4=(5,1),
                           padding1=(0,1),padding2=(1,0),padding3=(0,2),padding4=(2,0))
        )
        self.block1_2 = nn.Sequential(
            nn.Conv2d(out_channel_list*4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                           padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
        )
        self.block2_2 = nn.Sequential(
            nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                           padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
        )
        self.block3_2 = nn.Sequential(
            nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block4 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                           padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
        )
        self.block4_2 = nn.Sequential(
            nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block5 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                           padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
        )
        self.block5_2 = nn.Sequential(
            nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block6 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                           padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
        )
        self.block6_2 = nn.Sequential(
            nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block7 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                           padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
        )
        self.block7_2 = nn.Sequential(
            nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block8 = nn.Sequential(
            nn.Conv2d(out_channel_list, num_channels, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.block0(x)
        out = self.block1(out)
        out = self.block1_2(out)
        out = self.block2(out)
        out = self.block2_2(out)
        out = self.block3(out)
        out = self.block3_2(out)
        out = self.block4(out)
        out = self.block4_2(out)
        out = self.block5(out)
        out = self.block5_2(out)
        out = self.block6(out)
        out = self.block6_2(out)
        out = self.block7(out)
        out = self.block7_2(out)
        out = self.block8(out)
        return out


class CNN7(nn.Module):
    def __init__(self, num_channels=1,out_channel_list=32):
        super(CNN7, self).__init__()
        self.block0 = nn.Sequential(
            nn.Conv2d(num_channels, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block1 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1,3),kernel_size2=(3,1),kernel_size3=(1,5),kernel_size4=(5,1),
                           padding1=(0,1),padding2=(1,0),padding3=(0,2),padding4=(2,0))
        )
        self.block1_2 = nn.Sequential(
            nn.Conv2d(out_channel_list*4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                           padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
        )
        self.block2_2 = nn.Sequential(
            nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                           padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
        )
        self.block3_2 = nn.Sequential(
            nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block4 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                           padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
        )
        self.block4_2 = nn.Sequential(
            nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block5 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                           padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
        )
        self.block5_2 = nn.Sequential(
            nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block6 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                           padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
        )
        self.block6_2 = nn.Sequential(
            nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block7 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                           padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
        )
        self.block7_2 = nn.Sequential(
            nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block8 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                           padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
        )
        self.block8_2 = nn.Sequential(
            nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block9 = nn.Sequential(
            nn.Conv2d(out_channel_list, num_channels, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.block0(x)
        out = self.block1(out)
        out = self.block1_2(out)
        out = self.block2(out)
        out = self.block2_2(out)
        out = self.block3(out)
        out = self.block3_2(out)
        out = self.block4(out)
        out = self.block4_2(out)
        out = self.block5(out)
        out = self.block5_2(out)
        out = self.block6(out)
        out = self.block6_2(out)
        out = self.block7(out)
        out = self.block7_2(out)
        out = self.block8(out)
        out = self.block8_2(out)
        out = self.block9(out)
        return out


class CNN10(nn.Module):
    def __init__(self, num_channels=1,out_channel_list=64):
        super(CNN10, self).__init__()
        self.block0 = nn.Sequential(
            nn.Conv2d(num_channels, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block1 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1,3),kernel_size2=(3,1),kernel_size3=(1,5),kernel_size4=(5,1),
                           padding1=(0,1),padding2=(1,0),padding3=(0,2),padding4=(2,0))
        )
        self.block1_2 = nn.Sequential(
            nn.Conv2d(out_channel_list*4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                           padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
        )
        self.block2_2 = nn.Sequential(
            nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                           padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
        )
        self.block3_2 = nn.Sequential(
            nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block4 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                           padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
        )
        self.block4_2 = nn.Sequential(
            nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block5 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                           padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
        )
        self.block5_2 = nn.Sequential(
            nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block6 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                           padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
        )
        self.block6_2 = nn.Sequential(
            nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block7 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                           padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
        )
        self.block7_2 = nn.Sequential(
            nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block8 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                           padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
        )
        self.block8_2 = nn.Sequential(
            nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block9 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                           padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
        )
        self.block9_2 = nn.Sequential(
            nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block10 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                           padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
        )
        self.block10_2 = nn.Sequential(
            nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block11 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                           padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
        )
        self.block11_2 = nn.Sequential(
            nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block12 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                           padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
        )
        self.block12_2 = nn.Sequential(
            nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block13 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                           padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
        )
        self.block13_2 = nn.Sequential(
            nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block14 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                           padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
        )
        self.block14_2 = nn.Sequential(
            nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block15 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                           padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
        )
        self.block15_2 = nn.Sequential(
            nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block16 = nn.Sequential(
            BaseInception2(in_channel=out_channel_list,
                           out_channel=out_channel_list,
                           kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                           padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
        )
        self.block16_2 = nn.Sequential(
            nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block17 = nn.Sequential(
            nn.Conv2d(out_channel_list, num_channels, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.block0(x)
        out = self.block1(out)
        out = self.block1_2(out)
        out = self.block2(out)
        out = self.block2_2(out)
        out = self.block3(out)
        out = self.block3_2(out)
        out = self.block4(out)
        out = self.block4_2(out)
        out = self.block5(out)
        out = self.block5_2(out)
        out = self.block6(out)
        out = self.block6_2(out)
        out = self.block7(out)
        out = self.block7_2(out)
        out = self.block8(out)
        out = self.block8_2(out)
        out = self.block9(out)
        out = self.block9_2(out)
        out = self.block10(out)
        out = self.block10_2(out)
        out = self.block11(out)
        out = self.block11_2(out)
        out = self.block12(out)
        out = self.block12_2(out)
        out = self.block13(out)
        out = self.block13_2(out)
        out = self.block14(out)
        out = self.block14_2(out)
        out = self.block15(out)
        out = self.block15_2(out)
        out = self.block16(out)
        out = self.block16_2(out)
        out = self.block17(out)
        return out

    class CNN11(nn.Module):
        def __init__(self, num_channels=1, out_channel_list=128):
            super(CNN11, self).__init__()
            self.block0 = nn.Sequential(
                nn.Conv2d(num_channels, out_channel_list, kernel_size=1, padding=0),
                nn.ReLU(inplace=True)
            )
            self.block1 = nn.Sequential(
                BaseInception2(in_channel=out_channel_list,
                               out_channel=out_channel_list,
                               kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                               padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
            )
            self.block1_2 = nn.Sequential(
                nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
                nn.ReLU(inplace=True)
            )
            self.block2 = nn.Sequential(
                BaseInception2(in_channel=out_channel_list,
                               out_channel=out_channel_list,
                               kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                               padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
            )
            self.block2_2 = nn.Sequential(
                nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
                nn.ReLU(inplace=True)
            )
            self.block3 = nn.Sequential(
                BaseInception2(in_channel=out_channel_list,
                               out_channel=out_channel_list,
                               kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                               padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
            )
            self.block3_2 = nn.Sequential(
                nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
                nn.ReLU(inplace=True)
            )
            self.block4 = nn.Sequential(
                BaseInception2(in_channel=out_channel_list,
                               out_channel=out_channel_list,
                               kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                               padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
            )
            self.block4_2 = nn.Sequential(
                nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
                nn.ReLU(inplace=True)
            )
            self.block5 = nn.Sequential(
                BaseInception2(in_channel=out_channel_list,
                               out_channel=out_channel_list,
                               kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                               padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
            )
            self.block5_2 = nn.Sequential(
                nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
                nn.ReLU(inplace=True)
            )
            self.block6 = nn.Sequential(
                BaseInception2(in_channel=out_channel_list,
                               out_channel=out_channel_list,
                               kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                               padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
            )
            self.block6_2 = nn.Sequential(
                nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
                nn.ReLU(inplace=True)
            )
            self.block7 = nn.Sequential(
                BaseInception2(in_channel=out_channel_list,
                               out_channel=out_channel_list,
                               kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                               padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
            )
            self.block7_2 = nn.Sequential(
                nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
                nn.ReLU(inplace=True)
            )
            self.block8 = nn.Sequential(
                BaseInception2(in_channel=out_channel_list,
                               out_channel=out_channel_list,
                               kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                               padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
            )
            self.block8_2 = nn.Sequential(
                nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
                nn.ReLU(inplace=True)
            )
            self.block9 = nn.Sequential(
                BaseInception2(in_channel=out_channel_list,
                               out_channel=out_channel_list,
                               kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                               padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
            )
            self.block9_2 = nn.Sequential(
                nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
                nn.ReLU(inplace=True)
            )
            self.block10 = nn.Sequential(
                BaseInception2(in_channel=out_channel_list,
                               out_channel=out_channel_list,
                               kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                               padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
            )
            self.block10_2 = nn.Sequential(
                nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
                nn.ReLU(inplace=True)
            )
            self.block11 = nn.Sequential(
                BaseInception2(in_channel=out_channel_list,
                               out_channel=out_channel_list,
                               kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                               padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
            )
            self.block11_2 = nn.Sequential(
                nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
                nn.ReLU(inplace=True)
            )
            self.block12 = nn.Sequential(
                BaseInception2(in_channel=out_channel_list,
                               out_channel=out_channel_list,
                               kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                               padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
            )
            self.block12_2 = nn.Sequential(
                nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
                nn.ReLU(inplace=True)
            )
            self.block13 = nn.Sequential(
                BaseInception2(in_channel=out_channel_list,
                               out_channel=out_channel_list,
                               kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                               padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
            )
            self.block13_2 = nn.Sequential(
                nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
                nn.ReLU(inplace=True)
            )
            self.block14 = nn.Sequential(
                BaseInception2(in_channel=out_channel_list,
                               out_channel=out_channel_list,
                               kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                               padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
            )
            self.block14_2 = nn.Sequential(
                nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
                nn.ReLU(inplace=True)
            )
            self.block15 = nn.Sequential(
                BaseInception2(in_channel=out_channel_list,
                               out_channel=out_channel_list,
                               kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                               padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
            )
            self.block15_2 = nn.Sequential(
                nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
                nn.ReLU(inplace=True)
            )
            self.block16 = nn.Sequential(
                BaseInception2(in_channel=out_channel_list,
                               out_channel=out_channel_list,
                               kernel_size1=(1, 3), kernel_size2=(3, 1), kernel_size3=(1, 5), kernel_size4=(5, 1),
                               padding1=(0, 1), padding2=(1, 0), padding3=(0, 2), padding4=(2, 0))
            )
            self.block16_2 = nn.Sequential(
                nn.Conv2d(out_channel_list * 4, out_channel_list, kernel_size=1, padding=0),
                nn.ReLU(inplace=True)
            )
            self.block17 = nn.Sequential(
                nn.Conv2d(out_channel_list, num_channels, kernel_size=1, padding=0),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            out = self.block0(x)
            out = self.block1(out)
            out = self.block1_2(out)
            out = self.block2(out)
            out = self.block2_2(out)
            out = self.block3(out)
            out = self.block3_2(out)
            out = self.block4(out)
            out = self.block4_2(out)
            out = self.block5(out)
            out = self.block5_2(out)
            out = self.block6(out)
            out = self.block6_2(out)
            out = self.block7(out)
            out = self.block7_2(out)
            out = self.block8(out)
            out = self.block8_2(out)
            out = self.block9(out)
            out = self.block9_2(out)
            out = self.block10(out)
            out = self.block10_2(out)
            out = self.block11(out)
            out = self.block11_2(out)
            out = self.block12(out)
            out = self.block12_2(out)
            out = self.block13(out)
            out = self.block13_2(out)
            out = self.block14(out)
            out = self.block14_2(out)
            out = self.block15(out)
            out = self.block15_2(out)
            out = self.block16(out)
            out = self.block16_2(out)
            out = self.block17(out)
            return out