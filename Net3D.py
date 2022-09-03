import torch
import torch.nn as nn

class Net3D(nn.Module):
 
    def __init__(self, n_out):
        super(Net3D, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = self.conv_layer_3d(1, 8)
        self.conv2 = self.conv_layer_3d(8, 15)
        self.conv3 = self.conv_layer_3d(15, 25)
        # self.conv4 = self.conv_layer_3d(120, 250)
        
        self.fc_1 = self.fc_layer(1750, 900)
        self.fc_2 = nn.Linear(900, n_out)

    def conv_layer_3d(self, in_c, out_c, kernel_size=(5, 5, 5), pool = 5):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size, padding=2),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, pool, pool)),
        )
        return conv_layer

    def conv_layer_3d_UP(self, in_c, out_c, kernel_size=(5, 5, 5), pool = 5):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size, padding=2),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, pool, pool)),
        )
        return conv_layer

    def fc_layer(self, in_d, out_d):
        fc = nn.Sequential(
        nn.Linear(in_d, out_d),
        nn.LeakyReLU(),
        nn.BatchNorm1d(out_d),
        nn.Dropout(p = 0.3),
        )
        return fc

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = self.fc_1(x)
        x = self.fc_2(x)
        return x

