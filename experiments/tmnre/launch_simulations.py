import os
os.environ["DISABLE_TQDM"] = "True"
import argparse

import swyft.lightning as sl

import gw_parameters
from peregrine_simulator import Simulator

############################

if __name__ == "__main__":

    conf = gw_parameters.default_conf
    bounds = gw_parameters.limits

    parser = argparse.ArgumentParser()
    parser.add_argument('zarr_store', type=str, help='Location of zarr store')
    parser.add_argument('--n_simulations', type=int, help='Number of simulations requested')

    par_names = gw_parameters.intrinsic_variables + gw_parameters.extrinsic_variables
    for param in par_names:
        parser.add_argument(f'--{param}', type=float, nargs=2, default=bounds[param])
    
    args = parser.parse_args()
    
    simulation_store_path = args.zarr_store

    bounds['mass_ratio'] = args.mass_ratio
    bounds['chirp_mass'] = args.chirp_mass
    bounds['theta_jn'] = args.theta_jn
    bounds['phase'] = args.phase
    bounds['tilt_1'] = args.tilt_1
    bounds['tilt_2'] = args.tilt_2
    bounds['a_1'] = args.a_1
    bounds['a_2'] = args.a_2
    bounds['phi_12'] = args.phi_12
    bounds['phi_jl'] = args.phi_jl
    bounds['luminosity_distance'] = args.luminosity_distance
    bounds['dec'] = args.dec
    bounds['ra'] = args.ra
    bounds['psi'] = args.psi
    bounds['geocent_time'] = args.geocent_time

    swyft_simulator = Simulator(conf, bounds)
    
    chunk_size = 1000

    zarr_store = sl.ZarrStore(f"{simulation_store_path}")

    # if args.n_simulations is not None:
    #     simulations_remaining = max(0, zarr_store.sims_required - args.n_simulations)
    # else:
    #     simulations_remaining = 0

    # while zarr_store.sims_required > simulations_remaining:
    zarr_store.simulate(
        sampler=swyft_simulator,
        batch_size=chunk_size,
        max_sims=args.n_simulations,
    )
