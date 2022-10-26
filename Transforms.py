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
        if len(positions) == 0:
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
        if len(positions) == 0:
            return None
        positions = torch.tensor(positions) * mode
        return positions

