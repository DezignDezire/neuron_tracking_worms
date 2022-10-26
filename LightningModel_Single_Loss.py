import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only

from visualise import visualise_neurons


class SupervisedModel_Single(pl.LightningModule):

    "dataset is an array of [train, val, test] dataset"
    def __init__(self, net: nn.Module, dataset, batch_size, run_name, lr = 0.001):
      super(SupervisedModel_Single, self).__init__()
      self.net = net
      self.dataset = dataset
      self.loss_fn = nn.MSELoss(reduction='none')
      self.batch_size = batch_size
      self.run_name = run_name
      self.lr = lr

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
    def regr_loss(self, targets, predictions):
      mask = torch.ones(targets.shape).to(self.device).masked_fill(targets == 0, False)
      loss = self.loss_fn(targets, predictions)
      loss = (loss * mask).sum()

      non_zero_elements = mask.sum()
      return loss / non_zero_elements

    def training_step(self, batch, batch_idx):
      inputs, targets = batch
      predictions = self(inputs)
      loss = self.regr_loss(targets, predictions)
      self.log('train_loss', loss.item())

      if batch_idx < 4:
        visualise_neurons(inputs, targets, predictions.clone().detach(), batch_idx, 'train', self.run_name)
      
      return {'loss': loss}

    def validation_step(self, batch, batch_idx):
      inputs, targets = batch
      predictions = self(inputs)
      loss = self.regr_loss(targets, predictions)
      self.log('val_loss', loss.item())

      if batch_idx < 4:
        visualise_neurons(inputs, targets, predictions, batch_idx, 'val', self.run_name)

      return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
      inputs, targets = batch
      predictions = self(inputs)
      loss = self.regr_loss(targets, predictions)
                  
      if batch_idx < 20:
        visualise_neurons(inputs, targets, predictions, batch_idx, 'test', self.run_name)

      return {'tst_loss': loss}


    def validation_epoch_end(self, outputs):
      avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
      print("mean val loss:", str(avg_loss.cpu()))
      logs = {'val_loss': avg_loss}
      return {'progress_bar': logs}

    def test_epoch_end(self, outputs):
      losses = torch.stack([x['tst_loss'] for x in outputs]).cpu()
      avg_loss = losses.mean()
      print("mean test loss:", str(avg_loss.cpu()))
      logs = {'tst_loss': avg_loss}
      return {'progress_bar': logs}

    def load_model(self, path):
      self.net.load_state_dict(torch.load(path, map_location='cuda:0'), strict=False)
      print('Model Created!')
