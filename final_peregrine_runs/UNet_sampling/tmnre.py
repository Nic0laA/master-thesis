print(
    r"""
             /'{>           Initialising PEREGRINE
         ____) (____        ----------------------
       //'--;   ;--'\\      Type: TMNRE Inference
      ///////\_/\\\\\\\     Authors: U.Bhardwaj, J.Alvey
             m m            Version: v0.0.1 | April 2023
"""
)

import sys
from datetime import datetime
import glob
import pickle
import swyft.lightning as sl
from config_utils import read_config, init_config
from simulator_utils import init_simulator, simulate
from inference_utils import (
    setup_zarr_store,
    setup_dataloader,
    setup_trainer,
    init_network,
    save_logratios,
    save_bounds,
    load_bounds,
)

# For parallelisation
import subprocess
import psutil
import logging

def divide_into_njobs(number, njobs):
    quotient, remainder = divmod(number, njobs)
    parts = [quotient] * 18
    for i in range(remainder):
        parts[i] += 1
    return parts

if __name__ == "__main__":
    args = sys.argv[1:]
    print(
        f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [tmnre.py] | Reading config file"
    )
    print(f"Config: {args[0]}")
    tmnre_parser = read_config(args)
    conf = init_config(tmnre_parser, args)

    logging.basicConfig(
        filename=f"{conf['zarr_params']['store_path']}/log_{conf['zarr_params']['run_id']}.log",
        filemode="w",
        format="%(asctime)s | %(levelname)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO,
    )
    simulator = init_simulator(conf)
    bounds = None
    if conf["tmnre"]["generate_obs"]:
        obs = simulator.generate_observation()
        logging.warning(
            f"Overwriting observation file: {conf['zarr_params']['store_path']}/observation_{conf['zarr_params']['run_id']}"
        )
        with open(
            f"{conf['zarr_params']['store_path']}/observation_{conf['zarr_params']['run_id']}",
            "wb",
        ) as f:
            pickle.dump(obs, f)
    else:
        observation_path = conf["tmnre"]["obs_path"]
        with open(observation_path, "rb") as f:
            obs = pickle.load(f)
        subprocess.run(
            f"cp {observation_path} {conf['zarr_params']['store_path']}/observation_{conf['zarr_params']['run_id']}",
            shell=True,
        )
    logging.info(
        f"Observation loaded and saved in {conf['zarr_params']['store_path']}/observation_{conf['zarr_params']['run_id']}"
    )
    obs = sl.Sample(
        {key: obs[key] for key in ["d_t", "d_f", "d_f_w", "n_t", "n_f", "n_f_w"]}
    )

    for round_id in range(1, int(conf["tmnre"]["num_rounds"]) + 1):
        # Initialise the zarr store to save the simulations
        start_time = datetime.now()
        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [tmnre.py] | Initialising zarrstore for round {round_id}"
        )
        store = setup_zarr_store(conf, simulator, round_id=round_id)
        logging.info(f"Starting simulations for round {round_id}")
        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [tmnre.py] | Simulating data for round {round_id}"
        )
        if conf["zarr_params"]["run_parallel"]:
            print(
                f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [tmnre.py] | Running in parallel - spawning processes"
            )
            processes = []
            if conf["zarr_params"]["njobs"] == -1:
                njobs = psutil.cpu_count(logical=True)
            elif conf["zarr_params"]["njobs"] > psutil.cpu_count(logical=False):
                njobs = psutil.cpu_count(logical=True)
            else:
                njobs = conf["zarr_params"]["njobs"]
            
            if round_id > 1:
                # Specially adapted simulation launcher
                zarr_params = conf["zarr_params"]
                store_path = zarr_params["store_path"]
                simulation_store_path = f"{store_path}/simulations_{zarr_params['run_id']}_R{round_id}"
                number_of_simulations = conf["zarr_params"]["sim_schedule"][round_id - 1]
                list_of_cutoffs = [1e-5, 1e-3, 1e-2]
                for split in [0.9, 0.1]:
                    proportions = [2/5., 2/5., 1/5.]
                    for i, __ in enumerate(proportions):
                        
                        params = ['mass_ratio', 'chirp_mass', 'theta_jn', 'phase', 'tilt_1', 'tilt_2', 'a_1', 'a_2', 'phi_12', 'phi_jl', 'luminosity_distance', 'dec', 'ra', 'psi', 'geocent_time']
                        bounds = load_bounds(conf, round_id, str(list_of_cutoffs[i]))
                        mbounds = dict(zip(params, bounds))
                        simulations_for_cutoff = int(number_of_simulations * proportions[i] * split)
                        simulations_per_job = divide_into_njobs(simulations_for_cutoff, njobs)
                        logging.info(f'Starting {simulations_for_cutoff} simulations in round {round_id} for boundaries with cutoff {list_of_cutoffs[i]}')

                        args = [f'--{par} {mbounds[par][0]} {mbounds[par][1]}'.split(' ') for par in mbounds.keys()]
                        flat_args = [item for row in args for item in row]
                        print (flat_args)
                        processes = []
                        for job in range(njobs):
                            logging.info(f'Subprocess {job}. {simulations_per_job[job]} simulations')
                            p = subprocess.Popen([
                                sys.executable,
                                "launch_simulations.py",
                                simulation_store_path,
                                "--n_simulations",
                                str(simulations_per_job[job]),
                                "--config",
                                f"{conf['zarr_params']['store_path']}/config_{conf['zarr_params']['run_id']}.txt"
                            ] + flat_args)
                            processes.append(p) 
                        for p in processes:
                            return_code = p.wait()
                        if return_code != 0:
                            print(f"Subprocess failed with return code {return_code}")
                            sys.exit(return_code)
            else:
                for job in range(njobs):
                    p = subprocess.Popen(
                        [
                            sys.executable,
                            "run_parallel.py",
                            f"{conf['zarr_params']['store_path']}/config_{conf['zarr_params']['run_id']}.txt",
                            str(round_id),
                        ]
                    )
                    processes.append(p)
                for p in processes:
                    return_code = p.wait()
                if return_code != 0:
                    print(f"Subprocess failed with return code {return_code}")
                    sys.exit(return_code)
        else:
            bounds = load_bounds(conf, round_id)
            simulator = init_simulator(conf, bounds)
            simulate(simulator, store, conf)
        logging.info(f"Simulations for round {round_id} completed")
        # Initialise data loader for training
        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [tmnre.py] | Setting up dataloaders for round {round_id}"
        )
        train_data, val_data, trainer_dir = setup_dataloader(
            store, simulator, conf, round_id
        )

        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [tmnre.py] | Setting up trainer for round {round_id}"
        )
        trainer = setup_trainer(trainer_dir, conf, round_id)

        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [tmnre.py] | Initialising network for round {round_id}"
        )
        network = init_network(conf)

        if (
            not conf["tmnre"]["infer_only"]
            or len(glob.glob(f"{trainer_dir}/epoch*_R{round_id}.ckpt")) == 0
        ):
            print(
                f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [tmnre.py] | Training network for round {round_id}"
            )
            trainer.fit(network, train_data, val_data)
            logging.info(
                f"Training completed for round {round_id}, checkpoint available at {glob.glob(f'{trainer_dir}/epoch*_R{round_id}.ckpt')[0]}"
            )

        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [tmnre.py] | Generate prior samples"
        )
        prior_sim = init_simulator(conf, load_bounds(conf, round_id))
        prior_samples = prior_sim.sample(100_000, targets=["z_total"])

        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [tmnre.py] | Generate posterior samples"
        )
        trainer.test(
            network, val_data, glob.glob(f"{trainer_dir}/epoch*_R{round_id}.ckpt")[0]
        )
        logratios = trainer.infer(
            network, obs, prior_samples.get_dataloader(batch_size=2048)
        )
        logging.info(f"Logratios saved for round {round_id}")
        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [tmnre.py] | Saving logratios from round {round_id}"
        )
        save_logratios(logratios, conf, round_id)
        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [tmnre.py] | Update bounds from round {round_id}"
        )
        bounds1 = (
            sl.bounds.get_rect_bounds(logratios, threshold=1e-5)
            .bounds.squeeze(1)
            .numpy()
        )
        bounds2 = (
            sl.bounds.get_rect_bounds(logratios, threshold=1e-3)
            .bounds.squeeze(1)
            .numpy()
        )
        bounds3 = (
            sl.bounds.get_rect_bounds(logratios, threshold=1e-2)
            .bounds.squeeze(1)
            .numpy()
        )
        save_bounds(bounds1, conf, round_id, "1e-5")
        save_bounds(bounds2, conf, round_id, "1e-3")
        save_bounds(bounds3, conf, round_id, "1e-2")
        
        end_time = datetime.now()
        logging.info(f"Completed round {round_id}")
        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [tmnre.py] | Completed round {round_id} in {end_time - start_time}."
        )
