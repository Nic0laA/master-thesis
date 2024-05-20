
import os
import swyft.lightning as sl
import torch
torch.set_float32_matmul_precision('high')
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import swyft.lightning as sl

from model import ViTInferenceNetwork

# Data loader function

def load_data(data_dir, batch_size=64, train_test_split=0.9):
    
    zarr_store = sl.ZarrStore(f"{data_dir}")
    
    train_data = zarr_store.get_dataloader(
        num_workers=8,
        batch_size=batch_size,
        idx_range=[0, int(train_test_split * len(zarr_store.data.z_int))],
        on_after_load_sample=False
    )

    val_data = zarr_store.get_dataloader(
        num_workers=8,
        batch_size=batch_size,
        idx_range=[
            int(train_test_split * len(zarr_store.data.z_int)),
            len(zarr_store.data.z_int) - 1,
        ],
        on_after_load_sample=None
    )
    
    return train_data, val_data


# Set up the pytorch trainer settings

data_dir = '/scratch-shared/scur2012/training_data/default_limits_2e6/training_data'
training_store_path = '/scratch-shared/scur2012/trained_models/vanilla_vit_1d'

lr_monitor = LearningRateMonitor(logging_interval="epoch")

early_stopping_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.0,
    patience=7,
    verbose=False,
    mode="min",
)
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=f"{training_store_path}",
    filename="{epoch}-{step}_{val_loss:.2f}_{train_loss:.2f}",
    mode="min",
    every_n_train_steps=5000,
)

# Make directory for logger
logger_tbl = pl_loggers.TensorBoardLogger(
    save_dir=f"{training_store_path}",
    name=f"tb_logs",
    version=None,
    default_hp_metric=False,
)

swyft_trainer = sl.SwyftTrainer(
    accelerator='gpu',
    devices=1,
    min_epochs=10,
    max_epochs=200,
    logger=logger_tbl,
    callbacks=[lr_monitor, early_stopping_callback, checkpoint_callback],
    enable_progress_bar = True,
    val_check_interval=5000,
)

# Load network model

network = ViTInferenceNetwork(batch_size=64, learning_rate=1.6e-4)

# Fit data to model

train_data, val_data = load_data(data_dir, batch_size=64, train_test_split=0.95)

swyft_trainer.fit(
    network, 
    train_data, 
    val_data, 
    #ckpt_path='/scratch-shared/scur2012/trained_models/vanilla_vit_1d/epoch=0_val_loss=-4.16_train_loss=-4.53.ckpt'
)
