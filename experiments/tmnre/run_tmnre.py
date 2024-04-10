
import numpy as np
import pandas as pd
import glob
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import multiprocessing
import pickle
from loguru import logger
import swyft.lightning as sl

import torch
torch.set_float32_matmul_precision('high')
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import gw_parameters
import peregrine_simulator
from peregrine_simulator import Simulator
import peregrine_network
from peregrine_network import InferenceNetwork

import importlib
importlib.reload(gw_parameters)
importlib.reload(peregrine_simulator)
importlib.reload(peregrine_network)

############################################################################################
#
# STEP ONE: GENERATE GROUND-TRUTH OBSERVATION
#
############################################################################################

# Set up log
zarr_store_dirs = '/scratch-shared/scur2012/peregrine_data/tmnre_experiments'
name_of_run = 'test2'
logger.add(f"{zarr_store_dirs}/{name_of_run}/run_log.txt")

# Initialise configuration settings

logger.info(f'Loading configuration settings for run {name_of_run}')

conf = gw_parameters.default_conf
bounds = gw_parameters.limits

logger.info(f'Starting bounds for priors')
logger.info(f'\n{pd.DataFrame.from_dict(bounds, orient='index', columns=['min','max'])}')

# Load swyft-based simulator. 
# This builds the framework for the computational DAG

swyft_simulator = Simulator(conf, bounds)

# Generate ground-truth observation

logger.info(f'Generating ground-truth observation')

obs = swyft_simulator.generate_observation()

# Convert observation to swyft type sample

obs_sample = sl.Sample(
    {key: obs[key] for key in ["d_t", "d_f", "d_f_w", "n_t", "n_f", "n_f_w"]}
)

############################################################################################
#
# STEP TWO: BEGIN ROUNDS OF TRAINING NETWORK
#
############################################################################################

simulations_per_round = [30_000, 60_000, 90_000, 120_000, 120_000, 150_000, 150_000, 150_000]

for rnd_id, number_of_simulations in enumerate(simulations_per_round):

    logger.info(f'Inititalising simulation for round {rnd_id+1}')

    # Initilise simulator with the updated limits
    swyft_simulator = Simulator(conf, bounds)

    logger.info(f'Setting up zarr store for round {rnd_id+1}')

    # Set up the zarr store
    shapes, dtypes = swyft_simulator.get_shapes_and_dtypes()
    shapes.pop("n")
    dtypes.pop("n")
    
    simulation_store_path = f"{zarr_store_dirs}/{name_of_run}/simulations/round_{rnd_id+1}"

    chunk_size = 1000
    zarr_store = sl.ZarrStore(f"{simulation_store_path}")
    zarr_store.init(
        N=number_of_simulations, 
        chunk_size=chunk_size, 
        shapes=shapes,
        dtypes=dtypes)

    # Generate the random observations and save the data to the zarr store (multiprocessing)

    njobs = 16
    batches = [chunk_size] * ( number_of_simulations // chunk_size ) + [number_of_simulations % chunk_size]

    def populate_zarr_simulation(n_sims):
        zarr_store.simulate(
            sampler=swyft_simulator,
            batch_size=n_sims,
            max_sims=n_sims,
        )

    logger.info(f'Starting {number_of_simulations} simulations in round {rnd_id+1}')

    # Creating a pool of worker processes
    with multiprocessing.Pool(njobs) as pool:
        results = pool.map(populate_zarr_simulation, batches)

    # Initialise data loader for training network

    logger.info(f'Initialising data loader for round {rnd_id+1}')

    network_settings = dict(
        min_epochs = 30,
        max_epochs = 200,
        early_stopping = 7,
        learning_rate = 5e-4,
        num_workers = 8,
        training_batch_size = 256,
        validation_batch_size = 256,
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

    # Set up the pytorch trainer settings

    logger.info(f'Setting up the pytorch trainer for round {rnd_id+1}')

    training_store_path = f"{zarr_store_dirs}/{name_of_run}/training/round_{rnd_id+1}"

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
        filename="{epoch}_{val_loss:.2f}_{train_loss:.2f}" + f"_round_{rnd_id+1}",
        mode="min",
    )
    logger_tbl = pl_loggers.TensorBoardLogger(
        save_dir=f"{training_store_path}",
        name=f"test_round_{rnd_id+1}",
        version=None,
        default_hp_metric=False,
    )

    swyft_trainer = sl.SwyftTrainer(
        accelerator='gpu',
        devices=1,
        min_epochs=network_settings["min_epochs"],
        max_epochs=network_settings["max_epochs"],
        logger=logger_tbl,
        callbacks=[lr_monitor, early_stopping_callback, checkpoint_callback],
    )

    # Load network model
    
    logger.info(f'Initialising network for round {rnd_id+1}')
    
    network = InferenceNetwork(network_settings)

    # Fit data to model
    
    logger.info(f'Fitting network for round {rnd_id+1}')
    
    swyft_trainer.fit(network, train_data, val_data)

    # Generate prior samples for testing the trained model
    
    logger.info(f'Generating prior samples for round {rnd_id+1}')
    
    prior_simulator = Simulator(conf, bounds)
    prior_samples = prior_simulator.sample(100_000, targets=["z_total"])

    # Test the network
    
    logger.info(f'Running network test for round {rnd_id+1}')
    
    test = swyft_trainer.test(network, val_data, glob.glob(f"{training_store_path}/epoch*_round_{rnd_id+1}.ckpt")[0])

    logger.info(f'Network loss in test set for round {rnd_id+1}: {test[0]['test_loss']}')

    logger.info(f'Running inference for round {rnd_id+1}')

    # Run inference model
    logratios = swyft_trainer.infer( 
        network, 
        obs_sample, 
        prior_samples)

    # Save results
    with open(f"{zarr_store_dirs}/{name_of_run}/logratios/round_{rnd_id+1}.pickle", 'wb') as p:
        pickle.dump(logratios, p)
    
    logger.info(f'Updating bounds for round {rnd_id+1}')
    
    # Update the bounds
    par_names = gw_parameters.intrinsic_variables + gw_parameters.extrinsic_variables
    
    new_bounds = sl.bounds.get_rect_bounds(logratios, 1e-5).bounds.squeeze(1).numpy()
    for i,pname in enumerate(par_names):
        bounds[pname] = list(new_bounds[i])
        
    logger.info(f'New bounds for round {rnd_id+1}')
    logger.info(f'\n{pd.DataFrame(bounds, columns=['min','max'], index=par_names)}')
     
    # Save the bounds
    np.savetxt(f"{zarr_store_dirs}/{name_of_run}/bounds/round_{rnd_id+1}.txt", new_bounds)