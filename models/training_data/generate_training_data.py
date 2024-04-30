
import pandas as pd
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import subprocess
from loguru import logger


import torch
torch.set_float32_matmul_precision('high')
from lightning_fabric.utilities.seed import seed_everything
import swyft.lightning as sl

import gw_parameters
import peregrine_simulator
from peregrine_simulator import Simulator

import importlib
importlib.reload(gw_parameters)
importlib.reload(peregrine_simulator)

seed_everything(0)

def generate_wave_data(data_dir, number_of_simulations=1000):
    
    # Set up log
    logger.add(f"{data_dir}/run_log.txt")

    # Initialise configuration settings

    logger.info(f'Loading configuration settings')

    conf = gw_parameters.default_conf
    bounds = gw_parameters.limits

    bounds_df = pd.DataFrame.from_dict(bounds, orient='index', columns=['min','max'])
    logger.info(f'Bounds for simulations\n{bounds_df}')

    # Initilise simulator with the bounds
    swyft_simulator = Simulator(conf, bounds)

    logger.info(f'Setting up zarr store for simulation data')

    # Set up the zarr store
    shapes, dtypes = swyft_simulator.get_shapes_and_dtypes()
    shapes.pop("n")
    dtypes.pop("n")
    
    simulation_store_path = f"{data_dir}/training_data"

    chunk_size = 1000
    
    zarr_store = sl.ZarrStore(f"{simulation_store_path}")

    zarr_store.init(
        N=number_of_simulations,
        chunk_size=chunk_size,
        shapes=shapes,
        dtypes=dtypes)
        

    logger.info(f'Starting {number_of_simulations} simulations')

    njobs = 18

    # Launch additional processes for parallel simulations
    args = [f'--{par} {bounds[par][0]} {bounds[par][1]}'.split(' ') for par in bounds.keys()]
    flat_args = [item for row in args for item in row]
    processes = []
    for job in range(njobs):
        p = subprocess.Popen([
            "/home/scur2012/Thesis/Peregrine/.venv/bin/python3.12",
            "launch_simulations.py",
            simulation_store_path,
        ] + flat_args)
        processes.append(p) 
    for p in processes:
        p.wait()
    
if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('-n', type=int, required=True)
    
    args = parser.parse_args()
    
    generate_wave_data(data_dir=args.datadir, number_of_simulations=args.n)
    