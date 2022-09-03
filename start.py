import os
import time
import pytorch_lightning as pl 
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from Dataset import SupervisedDataset
from Net3D import Net3D
from SupervisedLightningModel import SupervisedModel


start = time.time()


TRAIN = False
LOAD_PRETRAINED_MODEL = True
MODEL_CHECKPOINT = "supervised-Net3D-epoch=06-val_loss=574.99.ckpt"

batch_size = 2


# TRAIN / TEST / EVAL SPLIT : 70 / 15 / 15
dataset = [SupervisedDataset(0, batch_size), SupervisedDataset(1, batch_size), SupervisedDataset(2, batch_size)]
label_output_dims = dataset[0].label_output_dims()

net = Net3D(n_out=label_output_dims)
model = SupervisedModel(net, dataset, batch_size)

if LOAD_PRETRAINED_MODEL:
    model = SupervisedModel.load_from_checkpoint('models/' + MODEL_CHECKPOINT, net=net, dataset=dataset, batch_size=batch_size)

print(model)

logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name="supervised_logs")

if TRAIN:
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='C:/Users/ZimAdmin/Documents/src/models',
        filename='supervised-Net3D-{epoch:02d}-{val_loss:.2f}'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    stochastic_weight_averager = StochasticWeightAveraging(swa_lrs=1e-2)

    trainer = pl.Trainer(max_epochs=150, callbacks=[checkpoint_callback, stochastic_weight_averager, lr_monitor], 
        logger=logger, accelerator='gpu', accumulate_grad_batches={0: 5, 5: 3, 10:2}, devices=1,
        auto_scale_batch_size= "power")    
    trainer.fit(model)

else:
    trainer = pl.Trainer(logger=logger, accelerator='gpu', devices=1)
    trainer.test(model, verbose=True)


end = time.time()
print(end - start)

