import numpy as np
import pandas as pd
import h5py
import torch
import zarr

from Transforms import Transforms
import torch.nn as nn


DATA_DIRECTORY = '/scratch/neurobiology/zimmer/Philipp/bright_worm5/'

## takes 4D array and returns biggest value for every 3D image
biggest_img_value = lambda x: np.max(x, axis = (0,1,2))
def normalize_4D(data):
  biggest_vals = np.expand_dims(biggest_img_value(data),axis = (0,1))
  data = np.divide(data, biggest_vals)
  ## downsample to quater size
  data = torch.unsqueeze(torch.from_numpy(data), dim = 0)
  down_sample = nn.MaxPool3d((2, 2, 2))
  data = down_sample(data)
  data = torch.squeeze(data, dim = 0)

  return data

# only consider finished neurons
def get_finished_neurons(path):
  finished_neurons_df = pd.read_csv(path)
  finished_neurons = list(finished_neurons_df['Neuron ID'][finished_neurons_df['Finished?'] == True])
  finished_neurons = sorted([int(''.join(filter(str.isdigit, finish))) for finish in finished_neurons])
  return finished_neurons

## traces shape: 3060 x 918 -> at most 153 neurons per frame
## for each frame: dict of neuron  label and center x, y, z coordiantes
def dict_from_row(row, sub_list_size = 6):
    positions = {} 
    sublists = [row[n:n+sub_list_size] for n in range(0, len(row), sub_list_size)]
    for sl in sublists:
        sl = list(sl)
        # one dict entriy looks like this: 'label': [z, x, y] - [21, 900, 650]
        positions[int(sl[5])] = [sl[1], sl[2], sl[3]]
    return positions

## center position of neurons 
def transform_row_to_labels(row, finished_neurons, n_neurons, image_shape):
    res = np.zeros(n_neurons * 3)
    row_dict = dict_from_row(row)
    ## key: neuron_id, val: [x, y, z]
    for key, val in row_dict.items():
        try:
          ## center neurons
            ind = finished_neurons.index(key) * 3 # assert
            scaled_z = val[0] - image_shape[0]/2
            scaled_y = val[1] - image_shape[1]/2
            scaled_x = val[2] - image_shape[2]/2
            res[ind], res[ind+1], res[ind+2] = scaled_z, scaled_y, scaled_x
        except:
            pass
    return res

# ground truth: each finished neuron has 3 coordinates, neurons are places subsequntly in the columns 
def get_ground_truth_coordinates(traces, finished_neurons, n_neurons, image_shape):
  y = torch.empty((0, n_neurons * 3))
  for i, row in traces.iterrows():
      labels = transform_row_to_labels(row, finished_neurons, n_neurons, image_shape)
      y = torch.cat((y, torch.unsqueeze(torch.tensor(labels), 0)), 0)
  return y


def load_images_and_positions():
    ## image size: n_samples, 21, 650, 900 - (z, y, x)
  img_data = zarr.open(DATA_DIRECTORY + 'dat/2021-12-17_16-28-19_worm5-channel-0-pco_camera1bigtiff_preprocessed.zarr.zip')
  ## load labels - neuron center positions
  traces_file = h5py.File(DATA_DIRECTORY + '4-traces/red_traces.h5')
  # column_names = ['area', 'z', 'x', 'y', 'intensity_image', 'label']
  traces = pd.DataFrame(traces_file['df_with_missing']['block0_values'])
  finished_neurons = get_finished_neurons(DATA_DIRECTORY + '3-tracking/manual_annotation/manual_tracking.csv')
  n_samples = len(traces)
  n_neurons = len(finished_neurons)
  ## y_positions shape: neuron1_x, neuron1_y, neuron1_z, neuron2_x, neuron2_y,...
  try:
    y_pos = torch.load('y_pos_worm5')
    y_positions = y_pos
    print("loaded stored positions")
  except:
    y_positions = get_ground_truth_coordinates(traces, finished_neurons, n_neurons, img_data[0].shape)
    torch.save(y_positions, 'y_pos_worm5')
    print("created positions from traces file")

  return img_data, y_positions, n_samples, n_neurons 


def scale_positions(positions, n_neurons, image_shape):
  scale = 10
  output = np.zeros((n_neurons, 3))
  zimmer_fluroscence_um_per_pixel_xy: float = 0.325
  ## scale along x dimension
  output[:,0] = positions[:,0] / image_shape[2] * (1 / zimmer_fluroscence_um_per_pixel_xy) * scale
  output[:,1] = positions[:,1] / image_shape[2] * scale
  output[:,2] = positions[:,2] / image_shape[2] * scale
  return output


class SupervisedDataset(object):
  # mode :: 0: train, 1: val, 2:test
  def __init__(self, mode, batch_size):
    self.mode = mode
    self.img_data, self.y_positions, n_samples, self.n_neurons = load_images_and_positions()
    self.transforms = Transforms(rotation_prob = 0.8, max_rot_angle = 180, flip_prob = 0.6)

    np.random.seed(1220)
    indices = np.random.permutation(n_samples)
    if self.mode == 0:   # TRAIN
      self.indxs = indices[:int(n_samples*0.7)]
    elif self.mode == 1: # VAL
      self.indxs = indices[int(n_samples*0.7):int(n_samples*0.85)]
    elif self.mode == 2: # TEST
      self.indxs = indices[int(n_samples*0.85):]
      # self.indxs = indices[int(n_samples*0.85):int(n_samples*0.85)+100]
    else:
      exit("mode must be between 0-2")

    round_to_batch_size = int(len(self.indxs) / batch_size)
    self.n_samples = round_to_batch_size * batch_size


  def __getitem__(self, idx):
    idx = self.indxs[idx]
    image = normalize_4D(self.img_data[idx])  # dual index for dataset mode
    image = torch.tensor(image.clone().detach(), dtype=torch.float32)    # turn into float
    positions = torch.tensor(self.y_positions[idx], dtype=torch.float32)
    positions = positions / 2
    positions = positions.reshape(self.n_neurons, 3)
    positions = scale_positions(positions, self.n_neurons, image.shape)
    if self.mode == 0:
      image, positions = self.transforms(image, positions)
    
    positions = torch.tensor(positions.flatten(), dtype=torch.float32)

    return image, positions


  def __len__(self):
    return self.n_samples

  def label_output_dims(self):
    return len(self.y_positions[0])

