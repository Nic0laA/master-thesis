
import os
import numpy as np
import pandas as pd
import glob
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import pickle
import subprocess
from loguru import logger

import torch
torch.set_float32_matmul_precision('high')
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from lightning_fabric.utilities.seed import seed_everything
import swyft.lightning as sl

import gw_parameters
from peregrine_simulator import Simulator
from enet_network import InferenceNetwork

seed_everything(0)

############################################################################################
#
# STEP ONE: GENERATE GROUND-TRUTH OBSERVATION
#
############################################################################################

# Set up log
zarr_store_dirs = '/scratch-shared/scur2012/peregrine_data/tmnre_experiments'
name_of_run = 'enet_test_1'
logger.add(f"{zarr_store_dirs}/{name_of_run}/run_log.txt")

# Initialise configuration settings

logger.info(f'Loading configuration settings for run {name_of_run}')

conf = gw_parameters.default_conf
bounds = gw_parameters.limits
multi_bounds = None
list_of_cutoffs = [1e-5,1e-3,1e-2]

bounds_df = pd.DataFrame.from_dict(bounds, orient='index', columns=['min','max'])
logger.info(f'Starting bounds for priors\n{bounds_df}')

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
# STEP TWO: RUN SIMULATIONS
#
############################################################################################

simulations_per_round = [30_000, 60_000, 90_000, 120_000, 120_000, 150_000, 150_000, 150_000]

#simulations_per_round = [
#    0, 0, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 90_000, 100_000, 110_000, 120_000,
#]

# simulations_per_round = [
#     0, 0, 50_000, 100_000, 150_000, 200_000, 250_000,
# ]

trainer_settings = dict(
    min_epochs = 20,
    max_epochs = 200,
    early_stopping = 7,
    num_workers = 8,
    training_batch_size = 64,
    validation_batch_size = 64,
    train_split = 0.8,
)

network_settings = dict(
    # Peregrine
    shuffling = True,
    include_noise = True,
    priors = dict(
        int_priors = conf['priors']['int_priors'],
        ext_priors = conf['priors']['ext_priors'],
    ),
    marginals = ((0, 1),),
    one_d_only = True,
    ifo_list = conf["waveform_params"]["ifo_list"],
    learning_rate = 10e-4,
    training_batch_size = trainer_settings['training_batch_size'],
)

checkpoint_path = "/home/scur2012/Thesis/master-thesis/models/unet_peregrine/epoch=41-step=305000_val_loss=-5.51_train_loss=-5.75.ckpt"
checkpoint = torch.load(checkpoint_path)

network = InferenceNetwork(**network_settings)
# network.load_state_dict(checkpoint['state_dict'])

# Freeze unet layers
# for name, param in network.named_parameters():
#     if 'unet' in name:
#         param.requires_grad = False

def divide_into_njobs(number, njobs):
    quotient, remainder = divmod(number, njobs)
    parts = [quotient] * 18
    for i in range(remainder):
        parts[i] += 1
    return parts

for rnd_id, number_of_simulations in enumerate(simulations_per_round):

    logger.info(f'Inititalising simulation for round {rnd_id+1}')

    # Initilise simulator with the updated limits
    swyft_simulator = Simulator(conf, bounds)

    if number_of_simulations > 0:

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

        def populate_zarr_simulation(n_sims):
            
            zarr_store.simulate(
                sampler=swyft_simulator,
                batch_size=chunk_size,
                max_sims=n_sims,
            )
            
        njobs = 18
        if multi_bounds is None:
            logger.info(f'Starting {number_of_simulations} simulations in round {rnd_id+1}')
            simulations_per_job = divide_into_njobs(number_of_simulations, njobs)
            
            # Launch additional processes for parrallel simulations
            args = [f'--{par} {bounds[par][0]} {bounds[par][1]}'.split(' ') for par in bounds.keys()]
            flat_args = [item for row in args for item in row]
            processes = []
            for job in range(njobs):
                logger.info(f'Subprocess {job}. {simulations_per_job[job]} simulations')
                p = subprocess.Popen([
                    "/home/scur2012/Thesis/Peregrine/.venv/bin/python3.12",
                    "launch_simulations.py",
                    simulation_store_path,
                    "--n_simulations",
                    str(simulations_per_job[job]),
                ] + flat_args)
                processes.append(p) 
            for p in processes:
                p.wait()
                
        else:
            for split in [0.8, 0.2]:
                proportions = [1/5., 2/5., 2/5.]
                for i, mbound in enumerate(multi_bounds):
                
                    simulations_for_cutoff = int(number_of_simulations * proportions[i] * split)
                    simulations_per_job = divide_into_njobs(simulations_for_cutoff, njobs)
                    logger.info(f'Starting {simulations_for_cutoff} simulations in round {rnd_id+1} for boundaries with cutoff {list_of_cutoffs[i]}')

                    args = [f'--{par} {mbound[par][0]} {mbound[par][1]}'.split(' ') for par in mbound.keys()]
                    flat_args = [item for row in args for item in row]
                    print (flat_args)
                    processes = []
                    for job in range(njobs):
                        logger.info(f'Subprocess {job}. {simulations_per_job[job]} simulations')
                        p = subprocess.Popen([
                            "/home/scur2012/Thesis/Peregrine/.venv/bin/python3.12",
                            "launch_simulations.py",
                            simulation_store_path,
                            "--n_simulations",
                            str(simulations_per_job[job]),
                        ] + flat_args)
                        processes.append(p) 
                    for p in processes:
                        p.wait()
            
        ############################################################################################
        #
        # STEP THREE: TRAIN NETWORK FROM SIMULATIONS
        #
        ############################################################################################
                    
        # Initialise data loader for training network

        logger.info(f'Initialising data loader for round {rnd_id+1}')

        zarr_store_size = len(zarr_store.data.z_int)

        train_data = zarr_store.get_dataloader(
            num_workers=trainer_settings['num_workers'],
            batch_size=trainer_settings['training_batch_size'],
            idx_range=[0, int(trainer_settings['train_split'] * len(zarr_store.data.z_int))],
            on_after_load_sample=False,
        )

        val_data = zarr_store.get_dataloader(
            num_workers=trainer_settings['num_workers'],
            batch_size=trainer_settings['validation_batch_size'],
            idx_range=[
                int(trainer_settings['train_split'] * len(zarr_store.data.z_int)),
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
        patience=trainer_settings["early_stopping"],
        verbose=False,
        mode="min",
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"{training_store_path}",
        filename="{epoch}_{val_loss:.2f}_{train_loss:.2f}" + f"_round_{rnd_id+1}",
        mode="min",
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
        min_epochs=trainer_settings["min_epochs"],
        max_epochs=trainer_settings["max_epochs"],
        logger=logger_tbl,
        callbacks=[lr_monitor, early_stopping_callback, checkpoint_callback],
        enable_progress_bar = True
    )

    if number_of_simulations > 0:

        # Load network model
        
        # logger.info(f'Initialising network for round {rnd_id+1}')
        
        network = InferenceNetwork(**network_settings)
        #network.load_state_dict(checkpoint['state_dict'])

        # Freeze unet layers
        # for name, param in network.named_parameters():
        #     if 'unet' in name:
        #         param.requires_grad = False

        # Fit data to model
        
        logger.info(f'Fitting network for round {rnd_id+1}')
        
        swyft_trainer.fit(network, train_data, val_data)
        
        # Test the network
    
        logger.info(f'Running network test for round {rnd_id+1}')
        
        test = swyft_trainer.test(network, val_data)

        logger.info(f'Network loss in test set for round {rnd_id+1}: {test[0]['test_loss']}')
        
    else:
        logger.info(f'No additional training for {rnd_id+1}')

    # Generate prior samples for testing the trained model
    
    logger.info(f'Generating prior samples for round {rnd_id+1}')
    
    prior_simulator = Simulator(conf, bounds)
    prior_samples = prior_simulator.sample(100_000, targets=["z_total"])

    logger.info(f'Running inference for round {rnd_id+1}')

    # Run inference model
    logratios = swyft_trainer.infer( 
        network, 
        obs_sample, 
        prior_samples)

    # Save results
    os.makedirs(f"{zarr_store_dirs}/{name_of_run}/logratios", exist_ok = True)
    with open(f"{zarr_store_dirs}/{name_of_run}/logratios/round_{rnd_id+1}.pickle", 'wb') as p:
        pickle.dump(logratios, p)
    
    logger.info(f'Updating bounds for round {rnd_id+2}')
    
    # Update the bounds
    par_names = gw_parameters.intrinsic_variables + gw_parameters.extrinsic_variables
    
    new_bounds = sl.bounds.get_rect_bounds(logratios, 1e-5).bounds.squeeze(1).numpy()
    for i,pname in enumerate(par_names):
        bounds[pname] = list(new_bounds[i])
    
    bounds_df = pd.DataFrame.from_dict(bounds, orient='index', columns=['min','max'])
    logger.info(f'New bounds for round {rnd_id+2}\n{bounds_df}')

    # Save the bounds
    os.makedirs(f"{zarr_store_dirs}/{name_of_run}/bounds", exist_ok = True)
    bounds_df.to_csv(f"{zarr_store_dirs}/{name_of_run}/bounds/round_{rnd_id+1}.csv")

    # Finer detailed bounds for more precise sampling
    continue

    multi_bounds = []
    for i,cutoff in enumerate(list_of_cutoffs):
        tmp_bounds = sl.bounds.get_rect_bounds(logratios, cutoff).bounds.squeeze(1).numpy()
        tmp_bound_dict = bounds.copy()
        for j,pname in enumerate(par_names):
            tmp_bound_dict[pname] = list(tmp_bounds[j])
        multi_bounds.append(tmp_bound_dict)
