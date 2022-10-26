import torch
import torch.nn as nn

class Net3D(nn.Module):
 
    def __init__(self, n_channels, n_out):
        super(Net3D, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = self.conv_layer_3d(1, n_channels)
        self.conv2 = self.conv_layer_3d(n_channels, n_channels * 2)
        self.conv3 = self.conv_layer_3d(n_channels * 2, n_channels * 4)
        self.conv4 = self.conv_layer_3d(n_channels * 4, n_channels * 8, z_pool=1)
        
        self.fc = self.fc_layer(n_channels * 8 * 20 * 28, 3200)
        self.final_fc = nn.Linear(3200, n_out)

    def conv_layer_3d(self, in_c, out_c, kernel_size=(3, 3, 3), z_pool = 2):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size, padding=1),
        nn.LeakyReLU(),
        nn.MaxPool3d((z_pool, 2, 2)),
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
        x = self.conv4(x)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = self.fc(x)
        x = self.final_fc(x)
        return x

    
