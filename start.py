import os
import time
import torch
import pytorch_lightning as pl 
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor


from Dataset import SupervisedDataset
from Net3D import SupervisedModel, Net3D


start = time.time()


TRAIN = False
CONTINUE_TRAINING = True
MODEL_CHECKPOINT = "models/supervised-Net3D-epoch=33-val_loss=0.16.ckpt"

batch_size = 2


# TRAIN / TEST / EVAL SPLIT : 70 / 15 / 15
dataset = [SupervisedDataset(0, batch_size), SupervisedDataset(1, batch_size), SupervisedDataset(2, batch_size)]
label_output_dims = dataset[0].label_output_dims()

# model size: 275.200MB
net = Net3D(n_out=label_output_dims)
model = SupervisedModel(net, dataset, batch_size)
print(model)

logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name="supervised_logs")

if TRAIN:
    ## Trainer
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='C:/Users/ZimAdmin/Documents/src/models',
        filename='supervised-Net3D-{epoch:02d}-{val_loss:.2f}'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    if CONTINUE_TRAINING:
        trainer = pl.Trainer(default_root_dir=MODEL_CHECKPOINT, 
            max_epochs=100, callbacks=[checkpoint_callback, lr_monitor],logger=logger, accelerator='gpu', devices=1)
    else:
        trainer = pl.Trainer(max_epochs=100, callbacks=[checkpoint_callback, lr_monitor], logger=logger, accelerator='gpu', devices=1)
    
    trainer.fit(model)

else:
    trainer = pl.Trainer(logger=logger, accelerator='gpu', devices=1)
    model = SupervisedModel.load_from_checkpoint(MODEL_CHECKPOINT, net=net, dataset=dataset, batch_size=batch_size)
    trainer.test(model, verbose=True)




end = time.time()
print(end - start)

