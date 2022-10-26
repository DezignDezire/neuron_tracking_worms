import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=1),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_c, out_c, kernel_size=(3, 3, 3), padding=1),
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

def fc_layer(in_d, out_d):
    fc = nn.Sequential(
    nn.Linear(in_d, out_d),
    nn.LeakyReLU(),
    nn.BatchNorm1d(out_d),
    nn.Dropout(p = 0.3),
    )
    return fc


class UNet3D(nn.Module):
    
    def __init__(self, n_channels, n_out, n_bottleneck_maps, run_name):
        super(UNet3D, self).__init__()
        
        self.run_name = run_name
        
        n_chan_1 = n_channels
        n_chan_2 = n_channels * 2
        n_chan_3 = n_channels * 4
        n_chan_4 = n_channels * 8

        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(1, n_chan_1) 
        self.down_conv_2 = double_conv(n_chan_1, n_chan_2)
        self.down_conv_3 = double_conv(n_chan_2, n_chan_3)
        self.down_conv_4 = double_conv(n_chan_3, n_chan_4)
        
        self.up_trans_1 = nn.ConvTranspose3d(n_chan_4, n_chan_3, kernel_size=(2, 3, 3), stride=2)
        self.up_conv_1 = double_conv(n_chan_4, n_chan_3)
        self.up_trans_2 = nn.ConvTranspose3d(n_chan_3, n_chan_2, kernel_size=(3, 3, 3), stride=2)
        self.up_conv_2 = double_conv(n_chan_3, n_chan_2)
        self.up_trans_3 = nn.ConvTranspose3d(n_chan_2, n_chan_1, kernel_size=(2, 3, 3), stride=2)
        self.up_conv_3 = double_conv(n_chan_2, n_chan_1)

        self.bottleneck_down = nn.Conv3d(n_chan_1, n_bottleneck_maps, kernel_size=(10, 1, 1))
        self.fc = nn.Linear(n_bottleneck_maps * 325 * 450, n_out)
        self.bottleneck_up = nn.ConvTranspose3d(n_bottleneck_maps, 1, kernel_size=(10, 1, 1))


    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        input_image = torch.mean(torch.squeeze(x.cpu(), dim=1), dim=1)
        ## encoder
        x1 = self.down_conv_1(x) #
        x2 = self.max_pool(x1)
        x3 = self.down_conv_2(x2) #
        x4 = self.max_pool(x3)
        x5 = self.down_conv_3(x4) #
        x6 = self.max_pool(x5)        
        x7 = self.down_conv_4(x6) #
        
        x = self.up_trans_1(x7)
        x = x[:, :, :, :, :-1]
        x = self.up_conv_1(torch.cat([x, x5], 1))

        x = self.up_trans_2(x)
        x = x[:, :, :, :-1, :]
        x = self.up_conv_2(torch.cat([x, x3], 1))

        x = self.up_trans_3(x)
        x = x[:, :, :, :, :-1]
        x = self.up_conv_3(torch.cat([x, x1], 1))

        x = self.bottleneck_down(x)
        # plot_bottleneck(x, input_image, self.run_name)

        # Regression output
        x_flat = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x1 = self.fc(x_flat)
        x2 = self.bottleneck_up(x)
        return x1, x2
    

def plot_bottleneck(image, input_image, run_name):
    for ind in range(2): # first two samples in batch
        img = torch.squeeze(image[ind], dim=1).cpu()
        # f, axs = plt.subplots(2, 3, dpi=300)

        # for ax, img in zip(axs.flat, img[:5]):
        #     im = ax.imshow(img)
        #     ax.axis('off')

        # axs[1,2].imshow(input_image[ind], cmap = 'plasma')
        # axs[1,2].axis('off')

        # f.subplots_adjust(hspace=-0.6)
        # cbar = f.colorbar(im, ax=axs[:,:], shrink=0.4)
        # cbar.ax.tick_params(labelsize=6)
        # plt.savefig('runs/{}/bottleneck_images/{}'.format(run_name, str(ind)))

        f, axs = plt.subplots(2, 2, dpi=300)

        for ax, img in zip(axs.flat, img[:3]):
            im = ax.imshow(img)
            ax.axis('off')

        axs[1,1].imshow(input_image[ind], cmap = 'plasma')
        axs[1,1].axis('off')

        f.subplots_adjust(hspace=-0.18)
        cbar = f.colorbar(im, ax=axs[:,:], shrink=0.7)
        cbar.ax.tick_params(labelsize=9)
        plt.savefig('runs/{}/bottleneck_images/{}'.format(run_name, str(ind)))
