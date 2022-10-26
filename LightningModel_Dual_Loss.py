import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only

from visualise import visualise_neurons, plot_confusion_matrix, retransform_points


class SupervisedModel_Dual(pl.LightningModule):

    "dataset is an array of [train, val, test] dataset"
    def __init__(self, net: nn.Module, dataset, batch_size, pixel_loss_ratio, run_name, lr = 0.001):
      super(SupervisedModel_Dual, self).__init__()
      self.net = net
      self.dataset = dataset
      self.loss_fn = nn.MSELoss(reduction='none')
      self.batch_size = batch_size
      self.pixel_loss_ratio = pixel_loss_ratio
      self.run_name = run_name
      self.lr = lr
      self.conf_mat = None

    def forward(self, x):
      return self.net(x)

    def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
      scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
      return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def prepare_data(self):
      self.train_dataset = self.dataset[0]
      self.val_dataset = self.dataset[1]
      self.test_dataset = self.dataset[2]

    def train_dataloader(self):
      return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
      return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
      return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

 
    ## masked loss
    def regr_loss(self, targets, predictions, vis = False):
      mask = torch.ones(targets.shape).to(self.device).masked_fill(targets == 0, False)
      loss = self.loss_fn(targets, predictions)
      n_neurons_shape = (loss.shape[0], -1, 3)
      loss = (loss * mask)
      point_loss = torch.reshape(loss, n_neurons_shape)
      loss = loss.sum()
      non_zero_elements = mask.sum()
      loss = loss / non_zero_elements

      unusable_pos = None
      assoc = None

      if vis:
        # distance over threshold
        point_loss = torch.sum(point_loss, dim=2)
        threshold = 0.004
        unusable_pos = [torch.where(loss_vec > threshold) for loss_vec in point_loss]

        # mean euclidean dists over all points
        targets = torch.reshape(targets, n_neurons_shape)
        predictions = torch.reshape(predictions, n_neurons_shape)
        mean_euclidean_dist = torch.mean(torch.sum(torch.square(targets - predictions), dim=2), dim=1)

        # correct associations (neares predictions)
        targets = torch.transpose(targets, 0, 1)
        predictions = torch.transpose(predictions, 0, 1)
        # get neuron-id with smallest distance 
        smallest_diff = [torch.argmin(torch.sum(torch.square(targets - p), dim=2), dim=0) for p in predictions]
        assoc = torch.vstack(smallest_diff).T
        
        # get smallest neighboring distance
        smallest_diff = [torch.topk(torch.sum(torch.square(targets - p), dim=2), dim=0, k = 2, largest = False)[0][1] for p in targets]
        flip = torch.vstack(smallest_diff).T
        print(flip.shape)
        min_neighbor_dist = [torch.min(arr[arr>0], dim = 0).values.item() if torch.numel(arr[arr>0]) > 0 else [] for arr in flip]
        print(min_neighbor_dist)

        return loss, mean_euclidean_dist, unusable_pos, assoc, min_neighbor_dist
      
      return loss

    def pixelwise_loss(self, images, recon_images):
      sq_pixel_diffs = torch.square(torch.absolute(images - recon_images))
      return torch.sum(sq_pixel_diffs)


    def training_step(self, batch, batch_idx):
      inputs, targets = batch
      predictions, recon_images = self(inputs)
      regr_loss = self.regr_loss(targets, predictions)
      # pixel_loss = self.pixelwise_loss(inputs, recon_images) / self.pixel_loss_ratio
      # loss = regr_loss + pixel_loss
      loss = regr_loss
      self.log('train_loss', loss.item())

      if batch_idx < 4:
        visualise_neurons(inputs, targets, predictions.clone().detach(), batch_idx, 'train', self.run_name)
      
      # print('regr_loss:', regr_loss.item(), ', pixel_loss:', pixel_loss.item(), ', total loss:', loss.item())
      # print('total loss:', loss.item())
      return {'loss': loss}

    def validation_step(self, batch, batch_idx):
      inputs, targets = batch
      predictions, recon_images = self(inputs)
      regr_loss = self.regr_loss(targets, predictions)
      pixel_loss = self.pixelwise_loss(inputs, recon_images)  / self.pixel_loss_ratio
      loss = regr_loss + pixel_loss
      self.log('val_loss', loss.item())

      if batch_idx < 4:
        visualise_neurons(inputs, targets, predictions, batch_idx, 'val', self.run_name)

      return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
      n_neurons = 109
      inputs, targets = batch
      predictions, recon_images = self(inputs)

      regr_loss, mean_euclidean_dist, unusable_pos, assoc, min_neighbor_dist = self.regr_loss(targets, predictions, True)
      pixel_loss = self.pixelwise_loss(inputs, recon_images)  / self.pixel_loss_ratio
      loss = regr_loss + pixel_loss

      if batch_idx < 20:
        visualise_neurons(inputs, targets, predictions, batch_idx, 'test', self.run_name, unusable_pos, assoc)
        self.conf_mat = plot_confusion_matrix(assoc, self.conf_mat, batch_idx, self.batch_size, self.run_name)

      return {'regr_loss': regr_loss,
              'euclidean_dist': mean_euclidean_dist.cpu(), 
              'n_usable_pos': torch.tensor([n_neurons - len(up[0]) for up in unusable_pos], dtype=torch.float),
              'all_unusable_pos': torch.tensor([i for u in unusable_pos for i in u[0].cpu().tolist()]),
              'corr_assoc': torch.tensor([len(torch.where(ass == torch.tensor(list(range(n_neurons))))[0]) for ass in assoc.cpu()], dtype=torch.float),
              'min_neighbor_dist': min_neighbor_dist}


    def validation_epoch_end(self, outputs):
      avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
      print("mean val loss:", str(avg_loss.cpu()))
      logs = {'val_loss': avg_loss}
      return {'progress_bar': logs}

    def test_epoch_end(self, outputs):
      losses = torch.stack([x['regr_loss'] for x in outputs]).cpu()
      euclidean_dist = torch.hstack([x['euclidean_dist'] for x in outputs]).cpu()
      usable_pos = torch.hstack([x['n_usable_pos'] for x in outputs]).cpu()
      all_unusable_pos = torch.hstack([x['all_unusable_pos'] for x in outputs])
      corr_assoc = torch.hstack([x['corr_assoc'] for x in outputs]).cpu()
      # min_neighbor_dist = [j for x in outputs for j in x['min_neighbor_dist']]
      min_neighbor_dist = [j for x in outputs for j in x['min_neighbor_dist']]

      avg_loss = losses.mean()
      print("mean test loss:", str(avg_loss.cpu()))
      print('euclidean_dist:', torch.mean(euclidean_dist))
      print('n_usable_pos:', torch.mean(usable_pos))
      print('corr_assoc:', torch.mean(corr_assoc))
      logs = {'tst_loss': avg_loss}
      return {'progress_bar': logs}

    def load_model(self, path):
      self.net.load_state_dict(torch.load(path, map_location='cuda:0'), strict=False)
      print('Model Created!')
