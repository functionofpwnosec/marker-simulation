import torch
import torch.nn as nn


########################################################################################################################
# Generator
########################################################################################################################

class LSTMUnet3dGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, norm='batch'):
        super().__init__()

        channels = 64
        self.e1 = CBR_3d(in_channels, channels, norm=None, relu=0.2)
        self.e2 = CBR_3d(channels, 2 * channels, norm=norm, relu=0.2)
        self.e3 = CBR_3d(2 * channels, 4 * channels, norm=norm, relu=0.2)
        self.e4 = CBR_3d(4 * channels, 8 * channels, norm=norm, relu=0.2)
        self.e5 = CBR_3d(8 * channels, 8 * channels, norm=norm, relu=0.2)
        self.e6 = CBR_3d(8 * channels, 8 * channels, norm=norm, relu=0.2)
        self.e7 = CBR_3d(8 * channels, 8 * channels, norm=norm, relu=0.2)
        self.e8 = CBR_3d(8 * channels, 8 * channels, norm=None, relu=0.2)

        self.lstm = nn.LSTM(input_size=8 * channels, hidden_size=8 * channels, num_layers=2, batch_first=True)

        self.d1 = CBDR_3d(8 * channels, 8 * channels, norm=norm, relu=0.0)
        self.d2 = CBDR_3d(2 * 8 * channels, 8 * channels, norm=norm, relu=0.0)
        self.d3 = CBDR_3d(2 * 8 * channels, 8 * channels, norm=norm, relu=0.0)
        self.d4 = CBDR_3d(2 * 8 * channels, 8 * channels, norm=norm, dropout=False, relu=0.0)
        self.d5 = CBDR_3d(2 * 8 * channels, 4 * channels, norm=norm, dropout=False, relu=0.0)
        self.d6 = CBDR_3d(2 * 4 * channels, 2 * channels, norm=norm, dropout=False, relu=0.0)
        self.d7 = CBDR_3d(2 * 2 * channels, channels, norm=norm, dropout=False, relu=0.0)
        self.d8 = nn.ConvTranspose3d(2 * channels, out_channels, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)

        e8 = e8.permute((0, 2, 1, 3, 4))

        lstm, _ = self.lstm(e8.view(x.shape[0], x.shape[2], -1))

        d0 = lstm.view(e8.shape).permute((0, 2, 1, 3, 4))
        d1 = self.d1(d0)

        cat2 = torch.cat((d1, e7), dim=1)
        d2 = self.d2(cat2)

        cat3 = torch.cat((d2, e6), dim=1)
        d3 = self.d3(cat3)

        cat4 = torch.cat((d3, e5), dim=1)
        d4 = self.d4(cat4)

        cat5 = torch.cat((d4, e4), dim=1)
        d5 = self.d5(cat5)

        cat6 = torch.cat((d5, e3), dim=1)
        d6 = self.d6(cat6)

        cat7 = torch.cat((d6, e2), dim=1)
        d7 = self.d7(cat7)

        cat8 = torch.cat((d7, e1), dim=1)
        d8 = self.d8(cat8)

        x = torch.tanh(d8)

        return x


class UnetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, norm='batch'):
        super().__init__()

        channels = 64
        self.e1 = CBR(in_channels, channels, norm=None, relu=0.2)
        self.e2 = CBR(channels, 2 * channels, norm=norm, relu=0.2)
        self.e3 = CBR(2 * channels, 4 * channels, norm=norm, relu=0.2)
        self.e4 = CBR(4 * channels, 8 * channels, norm=norm, relu=0.2)
        self.e5 = CBR(8 * channels, 8 * channels, norm=norm, relu=0.2)
        self.e6 = CBR(8 * channels, 8 * channels, norm=norm, relu=0.2)
        self.e7 = CBR(8 * channels, 8 * channels, norm=norm, relu=0.2)
        self.e8 = CBR(8 * channels, 8 * channels, norm=None, relu=0.2)

        self.d1 = CBDR(8 * channels, 8 * channels, norm=norm, relu=0.0)
        self.d2 = CBDR(2 * 8 * channels, 8 * channels, norm=norm, relu=0.0)
        self.d3 = CBDR(2 * 8 * channels, 8 * channels, norm=norm, relu=0.0)
        self.d4 = CBDR(2 * 8 * channels, 8 * channels, norm=norm, dropout=False, relu=0.0)
        self.d5 = CBDR(2 * 8 * channels, 4 * channels, norm=norm, dropout=False, relu=0.0)
        self.d6 = CBDR(2 * 4 * channels, 2 * channels, norm=norm, dropout=False, relu=0.0)
        self.d7 = CBDR(2 * 2 * channels, channels, norm=norm, dropout=False, relu=0.0)
        self.d8 = nn.ConvTranspose2d(2 * channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)

        d1 = self.d1(e8)

        cat2 = torch.cat((d1, e7), dim=1)
        d2 = self.d2(cat2)

        cat3 = torch.cat((d2, e6), dim=1)
        d3 = self.d3(cat3)

        cat4 = torch.cat((d3, e5), dim=1)
        d4 = self.d4(cat4)

        cat5 = torch.cat((d4, e4), dim=1)
        d5 = self.d5(cat5)

        cat6 = torch.cat((d5, e3), dim=1)
        d6 = self.d6(cat6)

        cat7 = torch.cat((d6, e2), dim=1)
        d7 = self.d7(cat7)

        cat8 = torch.cat((d7, e1), dim=1)
        d8 = self.d8(cat8)

        x = torch.tanh(d8)

        return x


class Unet3dGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, norm='batch'):
        super().__init__()

        channels = 64
        self.e1 = CBR_3d(in_channels, channels, norm=None, relu=0.2)
        self.e2 = CBR_3d(channels, 2 * channels, norm=norm, relu=0.2)
        self.e3 = CBR_3d(2 * channels, 4 * channels, norm=norm, relu=0.2)
        self.e4 = CBR_3d(4 * channels, 8 * channels, norm=norm, relu=0.2)
        self.e5 = CBR_3d(8 * channels, 8 * channels, norm=norm, relu=0.2)
        self.e6 = CBR_3d(8 * channels, 8 * channels, norm=norm, relu=0.2)
        self.e7 = CBR_3d(8 * channels, 8 * channels, norm=norm, relu=0.2)
        self.e8 = CBR_3d(8 * channels, 8 * channels, norm=None, relu=0.2)

        self.d1 = CBDR_3d(8 * channels, 8 * channels, norm=norm, relu=0.0)
        self.d2 = CBDR_3d(2 * 8 * channels, 8 * channels, norm=norm, relu=0.0)
        self.d3 = CBDR_3d(2 * 8 * channels, 8 * channels, norm=norm, relu=0.0)
        self.d4 = CBDR_3d(2 * 8 * channels, 8 * channels, norm=norm, dropout=False, relu=0.0)
        self.d5 = CBDR_3d(2 * 8 * channels, 4 * channels, norm=norm, dropout=False, relu=0.0)
        self.d6 = CBDR_3d(2 * 4 * channels, 2 * channels, norm=norm, dropout=False, relu=0.0)
        self.d7 = CBDR_3d(2 * 2 * channels, channels, norm=norm, dropout=False, relu=0.0)
        self.d8 = nn.ConvTranspose3d(2 * channels, out_channels, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)

        d1 = self.d1(e8)

        cat2 = torch.cat((d1, e7), dim=1)
        d2 = self.d2(cat2)

        cat3 = torch.cat((d2, e6), dim=1)
        d3 = self.d3(cat3)

        cat4 = torch.cat((d3, e5), dim=1)
        d4 = self.d4(cat4)

        cat5 = torch.cat((d4, e4), dim=1)
        d5 = self.d5(cat5)

        cat6 = torch.cat((d5, e3), dim=1)
        d6 = self.d6(cat6)

        cat7 = torch.cat((d6, e2), dim=1)
        d7 = self.d7(cat7)

        cat8 = torch.cat((d7, e1), dim=1)
        d8 = self.d8(cat8)

        x = torch.tanh(d8)

        return x


########################################################################################################################
# Discriminator
########################################################################################################################

class PatchGAN(nn.Module):
    def __init__(self, in_channels=6, n_layers=4, norm='batch'):
        super().__init__()

        out_channels = 64
        layers = [CBR(in_channels, out_channels, norm=None, relu=0.2)]

        for i in range(1, n_layers-1):
            in_channels = out_channels
            out_channels = min(2 * out_channels, 512)
            layers += [CBR(in_channels, out_channels, norm=norm, relu=0.2)]

        in_channels = out_channels
        out_channels = min(2 * out_channels, 512)

        layers += [
            CBR(in_channels, out_channels, norm=norm, stride=1, relu=0.2),
            nn.Conv2d(out_channels, 1, 4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PatchGAN3d(nn.Module):
    def __init__(self, in_channels=6, n_layers=4, norm='batch'):
        super().__init__()

        out_channels = 64
        layers = [CBR_3d(in_channels, out_channels, norm=None, relu=0.2)]

        for i in range(1, n_layers-1):
            in_channels = out_channels
            out_channels = min(2 * out_channels, 512)
            layers += [CBR_3d(in_channels, out_channels, norm=norm, relu=0.2)]

        in_channels = out_channels
        out_channels = min(2 * out_channels, 512)

        layers += [
            CBR_3d(in_channels, out_channels, norm=norm, stride=1, relu=0.2),
            nn.Conv3d(out_channels, 1, (1, 4, 4), stride=1, padding=(0, 1, 1))
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


########################################################################################################################
# Submodules
########################################################################################################################

class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, norm='batch', relu=0.0):
        super().__init__()

        layers = []

        if norm == 'batch':
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels)
            ]
        elif norm == 'instance':
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.InstanceNorm2d(out_channels)
            ]
        elif norm is None:
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)]

        if relu is not None:
            if relu == 0.0:
                layers += [nn.ReLU()]
            elif relu > 0.0:
                layers += [nn.LeakyReLU(relu)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)


class CBDR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, norm='batch', dropout=False, relu=0.0):
        super().__init__()

        layers = []

        if norm == 'batch':
            layers += [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels)
            ]
        elif norm == 'instance':
            layers += [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.InstanceNorm2d(out_channels)
            ]
        elif norm is None:
            layers += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)]

        if dropout:
            layers += [nn.Dropout(0.5)]

        if relu is not None:
            if relu == 0.0:
                layers += [nn.ReLU()]
            elif relu > 0.0:
                layers += [nn.LeakyReLU(relu)]

        self.cbdr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbdr(x)


class CBR_3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, norm='batch', relu=0.0):
        super().__init__()

        layers = []

        if norm == 'batch':
            layers += [
                nn.Conv3d(in_channels, out_channels, (1, kernel_size, kernel_size), stride=(1, stride, stride),
                          padding=(0, padding, padding), bias=False),
                nn.BatchNorm3d(out_channels)
            ]
        elif norm == 'instance':
            layers += [
                nn.Conv3d(in_channels, out_channels, (1, kernel_size, kernel_size), stride=(1, stride, stride),
                          padding=(0, padding, padding)),
                nn.InstanceNorm3d(out_channels)
            ]
        elif norm is None:
            layers += [nn.Conv3d(in_channels, out_channels, (1, kernel_size, kernel_size), stride=(1, stride, stride),
                                 padding=(0, padding, padding))]

        if relu is not None:
            if relu == 0.0:
                layers += [nn.ReLU()]
            elif relu > 0.0:
                layers += [nn.LeakyReLU(relu)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)


class CBDR_3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, norm='batch', dropout=False, relu=0.0):
        super().__init__()

        layers = []

        if norm == 'batch':
            layers += [
                nn.ConvTranspose3d(in_channels, out_channels, (1, kernel_size, kernel_size), stride=(1, stride, stride),
                                   padding=(0, padding, padding), bias=False),
                nn.BatchNorm3d(out_channels)
            ]
        elif norm == 'instance':
            layers += [
                nn.ConvTranspose3d(in_channels, out_channels, (1, kernel_size, kernel_size), stride=(1, stride, stride),
                                   padding=(0, padding, padding)),
                nn.InstanceNorm3d(out_channels)
            ]
        elif norm is None:
            layers += [nn.ConvTranspose3d(in_channels, out_channels, (1, kernel_size, kernel_size),
                                          stride=(1, stride, stride), padding=(0, padding, padding))]

        if dropout:
            layers += [nn.Dropout(0.5)]

        if relu is not None:
            if relu == 0.0:
                layers += [nn.ReLU()]
            elif relu > 0.0:
                layers += [nn.LeakyReLU(relu)]

        self.cbdr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbdr(x)