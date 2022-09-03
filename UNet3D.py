from operator import add
import torch
import torch.nn as nn


def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 5, 5), padding=1),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_c, out_c, kernel_size=(3, 5, 5), padding=1),
        nn.ReLU(inplace=True)
    )
    return conv

def crop_tensor(original, target):
    delta = (torch.tensor(original.shape) - torch.tensor(target.shape))[-2:]
    odd_ind = torch.where(delta % 2 != 0)
    add_minus = torch.tensor([0, 0])
    add_minus[odd_ind] += 1
    delta = delta // 2
    return original[:, :, :, delta[0]:original.shape[3]-delta[0]-add_minus[0], delta[1]:original.shape[4]-delta[1]-add_minus[1]]


class UNet3D(nn.Module):
    
    def __init__(self, n_out):
        # TODO: bump those up to a multiple
        n_chan_1 = 1
        n_chan_2 = 2
        n_chan_3 = 4
        n_chan_4 = 8
        
        super(UNet3D, self).__init__()

        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(1, n_chan_1) 
        self.down_conv_2 = double_conv(n_chan_1, n_chan_2)
        self.down_conv_3 = double_conv(n_chan_2, n_chan_3)
        self.down_conv_4 = double_conv(n_chan_3, n_chan_4)
        
        self.up_trans_1 = nn.ConvTranspose3d(n_chan_4, n_chan_3, kernel_size=(2, 5, 5), stride=2)
        self.up_conv_1 = double_conv(n_chan_4, n_chan_3)
        self.up_trans_2 = nn.ConvTranspose3d(n_chan_3, n_chan_2, kernel_size=(3, 5, 5), stride=2)
        self.up_conv_2 = double_conv(n_chan_3, n_chan_2)
        self.up_trans_3 = nn.ConvTranspose3d(n_chan_2, n_chan_1, kernel_size=(2, 5, 5), stride=2)
        self.up_conv_3 = double_conv(n_chan_2, n_chan_1)

        # TODO: do not use kernel size of 10 but kernel size=1
        self.out = nn.Conv3d(n_chan_1, 1, kernel_size=(10, 1, 1))

        self.fc = nn.Linear(96889, n_out)

    
    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        ## encoder
        x1 = self.down_conv_1(x) #
        x2 = self.max_pool(x1)
        x3 = self.down_conv_2(x2) #
        x4 = self.max_pool(x3)
        x5 = self.down_conv_3(x4) #
        x6 = self.max_pool(x5)        
        x7 = self.down_conv_4(x6) #
        ## x7.shape: 128, 1, 33, 48

        x = self.up_trans_1(x7)
        y = crop_tensor(x5, x)
        x = self.up_conv_1(torch.cat([x, y], 1))

        x = self.up_trans_2(x)
        y = crop_tensor(x3, x)
        x = self.up_conv_2(torch.cat([x, y], 1))

        x = self.up_trans_3(x)
        y = crop_tensor(x1, x)
        x = self.up_conv_3(torch.cat([x, y], 1))

        x = self.out(x)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        return self.fc(x)
        