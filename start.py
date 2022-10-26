import os
import time
import argparse
import pytorch_lightning as pl 
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from Dataset import SupervisedDataset
from Net3D import Net3D
from UNet3D import UNet3D
from LightningModel_Dual_Loss import SupervisedModel_Dual
from LightningModel_Single_Loss import SupervisedModel_Single


import torch

start = time.time()


TRAIN = False
LOAD_PRETRAINED_MODEL = True


def parse_arguments_and_create_dirs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', type=str, choices=['Net3D', 'UNet3D'])
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--n_channels', type=int, help='number of channels of the first layer')
    parser.add_argument('--n_bottleneck_feature_maps', type=int, help='number of featuer maps in the final convolution')
    parser.add_argument('--pixel_loss_ratio', type=int, help='scaling ratio for pixel wise loss')
    args = parser.parse_args()

    run_name = '{}-{}-{}-{}'.format(args.model_type, args.n_channels, args.n_bottleneck_feature_maps, args.pixel_loss_ratio)
    
    if TRAIN and not LOAD_PRETRAINED_MODEL:
        os.mkdir('runs/{}'.format(run_name))
        os.mkdir('runs/{}/progress_images'.format(run_name))
    
    return args, run_name

args, run_name = parse_arguments_and_create_dirs()
args.pixel_loss_ratio = 10**args.pixel_loss_ratio
print(run_name)



# conf_mat = torch.load(f'runs/{run_name}/conf_mat.pt')
# print(conf_mat)
# # max_cm = torch.max(torch.diag(conf_mat))
# dia = torch.ones((109,109))
# dia.fill_diagonal_(0)
# conf_mat *= dia
# print(conf_mat)

# K = 15
# greatest = []
# for k in range(K):
#     biggest = 0
#     temp = None
#     for i in range(109):
#         for j in range(109):
#             curr = conf_mat[i, j]
#             if curr > biggest:
#                 biggest = curr
#                 temp = (i, j)
#     print(biggest)
#     greatest.append(temp)
#     conf_mat[list(temp)] = 0
# print(greatest)
                

# exit()




# TRAIN / TEST / EVAL SPLIT : 70 / 15 / 15
dataset = [SupervisedDataset(0, args.batch_size), SupervisedDataset(1, args.batch_size), SupervisedDataset(2, args.batch_size)]
label_output_dims = dataset[0].label_output_dims()

if args.model_type == 'Net3D':
    net = Net3D(args.n_channels, n_out=label_output_dims)
    model = SupervisedModel_Single(net, dataset, args.batch_size, run_name)

elif args.model_type == 'UNet3D':
    net = UNet3D(args.n_channels, label_output_dims, args.n_bottleneck_feature_maps, run_name)
    model = SupervisedModel_Dual(net, dataset, args.batch_size, args.pixel_loss_ratio, run_name)

else:
    exit('invalid model type')


if LOAD_PRETRAINED_MODEL:
    for file in os.listdir("runs/{}".format(run_name)):
        if file.endswith(".ckpt"):
            checkpoint = (os.path.join("runs/", run_name, file))
            break

    model = SupervisedModel_Dual.load_from_checkpoint(checkpoint, run_name=run_name, net=net, dataset=dataset, batch_size=args.batch_size, pixel_loss_ratio=args.pixel_loss_ratio)

#########################################################################################
# dataloader = torch.utils.data.DataLoader(dataset[0], shuffle=True)
# X, y = next(iter(dataloader))

# model.to_onnx("model_lightnining_export.onnx", 
#                 X.to("cuda"),
#                 export_params=True,
#                 input_names = ['input'],  
#                 output_names = ['output'], 
#                 dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}})
#########################################################################################

print(model)

logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name="supervised_logs")
logger = WandbLogger(project="neuron-tracking", name=run_name)

if TRAIN:
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='runs/{}/'.format(run_name),
        filename='{args.model_type}-{epoch:02d}-{val_loss:.2f}'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    stochastic_weight_averager = StochasticWeightAveraging(swa_lrs=1e-2)

    trainer = pl.Trainer(max_epochs=150, 
        callbacks=[checkpoint_callback, stochastic_weight_averager, lr_monitor], 
        logger=logger, log_every_n_steps=10,
        accelerator='gpu', devices=2, strategy='dp',
        # accumulate_grad_batches={0: 5, 5: 3, 10:2, 20:1}, 
        auto_scale_batch_size= "power")    
    trainer.fit(model)
else:
    trainer = pl.Trainer(logger=logger, accelerator='gpu', devices=1)
    trainer.test(model, verbose=True)


end = time.time()
print(end - start)
