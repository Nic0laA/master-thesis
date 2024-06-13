import os
import argparse
import numpy as np
import swyft.lightning as sl

from simulator_utils import init_simulator, simulate
from inference_utils import setup_zarr_store, load_bounds
from config_utils import read_config, init_config

############################

if __name__ == "__main__":

    bounds = np.zeros((15,2))

    parser = argparse.ArgumentParser()
    parser.add_argument('zarr_store', type=str, help='Location of zarr store')
    parser.add_argument('--n_simulations', type=int, help='Number of simulations requested')
    parser.add_argument('--config', type=int, help='Path to config file')

    par_names = ['mass_ratio', 'chirp_mass', 'theta_jn', 'phase', 'tilt_1', 'tilt_2', 'a_1', 'a_2', 'phi_12', 'phi_jl', 'luminosity_distance', 'dec', 'ra', 'psi', 'geocent_time']
    for param in par_names:
        parser.add_argument(f'--{param}', type=float, nargs=2, default=bounds[0])
    
    args = parser.parse_args()
    
    simulation_store_path = args.zarr_store

    bounds[0] = args.mass_ratio
    bounds[1] = args.chirp_mass
    bounds[2] = args.theta_jn
    bounds[3] = args.phase
    bounds[4] = args.tilt_1
    bounds[5] = args.tilt_2
    bounds[6] = args.a_1
    bounds[7] = args.a_2
    bounds[8] = args.phi_12
    bounds[9] = args.phi_jl
    bounds[10] = args.luminosity_distance
    bounds[11] = args.dec
    bounds[12] = args.ra
    bounds[13] = args.psi
    bounds[14] = args.geocent_time

    tmnre_parser = read_config([args.config])
    conf = init_config(tmnre_parser, args, sim=True)

    simulator = init_simulator(conf, bounds)
    
    chunk_size = 1000

    zarr_store = sl.ZarrStore(f"{simulation_store_path}")

    zarr_store.simulate(
        sampler=simulator,
        batch_size=chunk_size,
        max_sims=args.n_simulations,
    )
