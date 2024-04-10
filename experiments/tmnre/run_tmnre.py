
import numpy as np
import zarr
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import swyft.lightning as sl

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import constants
import peregrine_simulator
from peregrine_simulator import Simulator
import peregrine_network
from peregrine_network import InferenceNetwork

import importlib
importlib.reload(constants)
importlib.reload(peregrine_simulator)
importlib.reload(peregrine_network)

# Initialise configuration settings

conf = constants.default_conf
bounds = constants.limits

# Load swyft-based simulator. 
# This builds the framework for the computational DAG

swyft_simulator = Simulator(conf, bounds)

# Generate ground-truth observation

obs = swyft_simulator.generate_observation()

# Convert observation to swyft type sample

obs_sample = sl.Sample(
    {key: obs[key] for key in ["d_t", "d_f", "d_f_w", "n_t", "n_f", "n_f_w"]}
)

# Set up the zarr store

name_of_run = 'test'
number_of_simulations = 1000
shapes, dtypes = swyft_simulator.get_shapes_and_dtypes()
shapes.pop("n")
dtypes.pop("n")

zarr_store_dirs = '/scratch-shared/scur2012/peregrine_data/tmnre_experiments'
simulation_store_path = f"{zarr_store_dirs}/{name_of_run}/simulations"

chunk_size = 1000
zarr_store = sl.ZarrStore(f"{simulation_store_path}")
zarr_store.init(
    N=number_of_simulations, 
    chunk_size=chunk_size, 
    shapes=shapes,
    dtypes=dtypes)

# Generate the random observations and save the data to the zarr store

zarr_store.simulate(
    sampler=swyft_simulator,
    batch_size=chunk_size,
    max_sims=number_of_simulations,
)

# Initialise data loader for training network

network_settings = dict(
    min_epochs = 1,
    max_epochs = 10,
    early_stopping = 7,
    learning_rate = 5e-4,
    num_workers = 8,
    training_batch_size = 1,
    validation_batch_size = 1,
    train_split = 0.9,
    val_split = 0.1,
    shuffling = True,
    priors = dict(
        int_priors = conf['priors']['int_priors'],
        ext_priors = conf['priors']['ext_priors'],
    ),
    marginals = (0,1),
    one_d_only = True,
    ifo_list = conf["waveform_params"]["ifo_list"],
)


train_data = zarr_store.get_dataloader(
    num_workers=network_settings['num_workers'],
    batch_size=network_settings['training_batch_size'],
    idx_range=[0, int(network_settings['train_split'] * len(zarr_store.data.z_int))],
    on_after_load_sample=False,
)

val_data = zarr_store.get_dataloader(
    num_workers=network_settings['num_workers'],
    batch_size=network_settings['validation_batch_size'],
    idx_range=[
        int(network_settings['train_split'] * len(zarr_store.data.z_int)),
        len(zarr_store.data.z_int) - 1,
    ],
    on_after_load_sample=None,
)

training_store_path = f"{zarr_store_dirs}/{name_of_run}/training"
round_id = 1

# Set up the pytorch trainer settings

lr_monitor = LearningRateMonitor(logging_interval="step")
early_stopping_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.0,
    patience=network_settings["early_stopping"],
    verbose=False,
    mode="min",
)
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=f"{training_store_path}",
    filename="{epoch}_{val_loss:.2f}_{train_loss:.2f}" + f"_R{round_id}",
    mode="min",
)
logger_tbl = pl_loggers.TensorBoardLogger(
    save_dir=f"{training_store_path}",
    name=f"test_R{round_id}",
    version=None,
    default_hp_metric=False,
)

swyft_trainer = sl.SwyftTrainer(
    accelerator='gpu',
    gpus=1,
    min_epochs=network_settings["min_epochs"],
    max_epochs=network_settings["max_epochs"],
    logger=logger_tbl,
    callbacks=[lr_monitor, early_stopping_callback, checkpoint_callback],
)

# Load network model
network = InferenceNetwork(network_settings)

# Fit data to model
swyft_trainer.fit(network, train_data, val_data)

# Generate prior samples for testing the trained model
prior_simulator = Simulator(conf, bounds)
prior_samples = prior_simulator.sample(100_000, targets=["z_total"])

# Test the network
swyft_trainer.test(network, val_data, "/scratch-shared/scur2012/peregrine_data/tmnre_experiments/test/training/test_R1/version_2/checkpoints/epoch=7-step=8000.ckpt")

# logratios = swyft_trainer.infer( 
#     network, 
#     sl.Samples({k: [v] for k, v in obs.items()}).get_dataloader(batch_size=1), 
#     prior_samples.get_dataloader(batch_size=2048) )

logratios = swyft_trainer.infer( 
    network, 
    obs_sample, 
    prior_samples)
