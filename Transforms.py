import torch
import numpy as np
import torchvision.transforms.functional as TF
import math
import random
import matplotlib.pyplot as plt

from visualise import retransform_points


class Transforms:
    def __init__(self, rotation_prob = 0, max_rot_angle = 360, 
                flip_prob = 0) -> None:
        ## rotation
        self.rotation_prob = rotation_prob
        self.max_rot_angle = max_rot_angle
        self.applied_rot_angle = None

        ## flip
        self.flip_prob = flip_prob
        self.applied_mode = None

        ##
        # self.

    
    def __call__(self, image = None, positions = None):
        if random.random() < self.rotation_prob:
            angle = random.choice(list(range(self.max_rot_angle)))
            image =  self.rotate_image(image, angle)
            positions = self.rotate_positions(positions, angle)

        if random.random() < self.flip_prob:
            ## only second order flips make sense here (in 3D)
            modes = [[1, 1, 1], [1, -1, -1], [-1, -1, 1], [-1, 1, -1]]
            mode = torch.tensor(random.choice(modes))
            image = self.flip_image(image, mode)
            positions = self.flip_positions(positions, mode)

        return image, positions






    def rotate_image(self, image, angle):
        if image == None:
            return None

        return TF.rotate(image, angle)

    def rotate_positions(self, positions, angle):
        if positions == None:
            return None

        radians = torch.tensor(angle * math.pi / 180)
        points = torch.tensor(positions[:,[2,1]].T, dtype=float)
        c, s = torch.cos(radians), torch.sin(radians)
        R = torch.tensor([[c, s], [-s, c]], dtype=float)
        res = R @ points
        res = res.T
        stacked = np.stack([positions[:,0], res[:,1], res[:,0]])
        res = torch.from_numpy(stacked).T
        return res


    def flip_image(self, image, mode):
        if image == None:
            return None
        
        dims = tuple(torch.where(mode == torch.tensor(-1))[0])
        return torch.flip(image, dims)

    def flip_positions(self, positions, mode):
        if positions == None:
            return None
        
        positions = positions * mode
        return positions


    # def random_crop(image, positions):
    # positions = torch.tensor(positions)
    # img_shape = image.shape
    # crop_sizes_xy = [300, 300]
    # center_of_mass_positions = torch.mean(positions, dim=0)
    # print(center_of_mass_positions)
    # positions = positions - center_of_mass_positions
    # center_of_mass_positions = torch.unsqueeze(torch.tensor(center_of_mass_positions), 0)
    # center_of_mass = retransform_points(center_of_mass_positions)

    # print('ceeeenter', center_of_mass)
    # image = TF.crop(image, int(center_of_mass[0,1] - crop_sizes_xy[1]/2), int(center_of_mass[0,2] - crop_sizes_xy[0]/2), crop_sizes_xy[0], crop_sizes_xy[1])
    # # TT.Pad starts at left side
    # print(image.shape)
    # assert(image.shape[2] == crop_sizes_xy[1])
    # assert(image.shape[1] == crop_sizes_xy[0])
    # padd_left_and_right = int((img_shape[2] - image.shape[2]) / 2)
    # padd_top_and_bottom = int((img_shape[1] - image.shape[1]) / 2)
    # print(padd_left_and_right, padd_top_and_bottom)
    # image = TF.pad(image, (padd_left_and_right, padd_top_and_bottom))
    # print(image.shape)
    # return image, positions



## translations - shifting / moving

## crops