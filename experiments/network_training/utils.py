
import sys
import numpy as np
import pandas as pd
import h5py

from sympy import symbols, solve

from bilby.gw import WaveformGenerator, detector, source, conversion

path_to_peregrine = '/home/scur2012/Thesis/Peregrine/peregrine/peregrine'
if path_to_peregrine not in sys.path:
    sys.path.append(path_to_peregrine)
from simulator_utils import Simulator

from loguru import logger

# Generate random signals

def calc_bh_masses(row):

    mass_ratio = row['mass_ratio']
    chirp_mass = row['chirp_mass']

    m1_solve = (chirp_mass * (mass_ratio + 1)**(1/5)) / mass_ratio**(3/5)

    # m1 = symbols('m1')
    # m2 = mass_ratio * m1
    # m1_solve = float( solve(((m1*m2)**(3/5)) / ((m1 + m2)**(1/5)) - chirp_mass, m1)[0] )
    m2_solve = mass_ratio * m1_solve

    return pd.Series([m1_solve, m2_solve], index=['mass_1', 'mass_2'])


def sample_random_parameters(n_samples: int, limits: dict = None, distributions: dict = None):
    
    if limits is None:
        limits = dict(
            mass_ratio = [0.125,1.0],
            chirp_mass = [25,100],
            theta_jn = [0.0,3.14159],
            phase = [0.0,6.28318],
            tilt_1 = [0.0,3.14159],
            tilt_2 = [0.0,3.14159],
            a_1 = [0.05,1.0],
            a_2 = [0.05,1.0],
            phi_12 = [0.0,6.28318],
            phi_jl = [0.0,6.28318],
            luminosity_distance = [100,1500],
            dec = [-1.57079,1.57079],
            ra = [0.0,6.28318],
            psi = [0.0,3.14159],
            geocent_time = [-0.1,0.1],
        )
        
    if distributions is None:
        distributions = dict(
            mass_ratio = 'uniform',
            chirp_mass = 'uniform',
            theta_jn = 'uniform',
            phase = 'uniform',
            tilt_1 = 'uniform',
            tilt_2 = 'uniform',
            a_1 = 'uniform',
            a_2 = 'uniform',
            phi_12 = 'uniform',
            phi_jl = 'uniform',
            luminosity_distance = 'uniform',
            dec = 'uniform',
            ra = 'uniform',
            psi = 'uniform',
            geocent_time = 'uniform',
        )
    
    sample_params_df = pd.DataFrame(index = range(n_samples), columns = list(limits.keys()))
    
    for param_name, (lower, upper) in limits.items():
            
        if distributions[param_name] == 'cosine':
            uniform_samples = np.random.uniform(0, np.pi, n_samples)
            sample_params_df[param_name] = np.cos(uniform_samples)
        
        elif distributions[param_name] == 'normal':
            pass
        
        else:
            sample_params_df[param_name] = np.random.uniform(low=lower, high=upper, size=n_samples)
            
    #sample_params_df[['mass_1','mass_2']] = sample_params_df.apply(calc_bh_masses, axis=1)
    
    sample_params_df['mass_1'] = (sample_params_df['chirp_mass'] * (sample_params_df['mass_ratio'] + 1)**(1/5)) \
                                / sample_params_df['mass_ratio']**(3/5)
    sample_params_df['mass_2'] = sample_params_df['mass_ratio'] * sample_params_df['mass_1']
    
    return sample_params_df

def generate_random_observations(n_samples: int, limits: dict = None, distributions: dict = None, save_results: bool = False):
    
    logger.info(f'Generating {n_samples} random samples')
    sample_params = sample_random_parameters(n_samples, limits, distributions)
    
    intrinsic_variables = ['mass_ratio', 'chirp_mass', 'luminosity_distance', 'dec', 'ra', 'theta_jn', 'psi', 'phase', 'tilt_1', 'tilt_2']
    extrinsic_variables = ['a_1', 'a_2', 'phi_12', 'phi_jl', 'geocent_time']
    
    sourced = dict(
        source_type = 'BBH',
        aligned_spins = False,
        fd_source_model = source.lal_binary_black_hole,
        param_conversion_model = conversion.convert_to_lal_binary_black_hole_parameters,
    )

    waveform_params = dict(
        sampling_frequency = 2048,
        duration = 4,
        start_offset = 2,
        start = -2,
        waveform_apprx = 'SEOBNRv4PHM',
        minimum_frequency = 20,
        maximum_frequency = 1024,
        reference_frequency = 50,
        ifo_list = ['H1','L1','V1'],
        ifo_noise = True,
    )  
        
    d_t_array = np.zeros((n_samples, len(waveform_params['ifo_list']), 
                          waveform_params['sampling_frequency']*waveform_params['duration']))
    d_f_w_array = np.zeros((n_samples, len(waveform_params['ifo_list']*2), 4097))
    d_f_array = np.zeros((n_samples, len(waveform_params['ifo_list']*2), 4097))
    n_t_array = np.zeros((n_samples, len(waveform_params['ifo_list']), 
                          waveform_params['sampling_frequency']*waveform_params['duration']))
    n_f_w_array = np.zeros((n_samples, len(waveform_params['ifo_list']*2), 4097))
    n_f_array = np.zeros((n_samples, len(waveform_params['ifo_list']*2), 4097))
    
    logger.info(f'Generating {n_samples} random observations')
    # for idx, sample in sample_params.iterrows():
    #     
    #     injection = sample.to_dict()
    #     
    #     conf = {
    #         'source' : sourced,
    #         'waveform_params' : waveform_params,
    #         'injection' : injection,
    #         'priors' : {
    #             'int_priors' : {key: injection[key] for key in intrinsic_variables},
    #             'ext_priors' : {key: injection[key] for key in extrinsic_variables},
    #         },
    #     }
    #     
    #     simulator = Simulator(conf)
    #     obs = simulator.generate_observation()
    #     
    #     d_t_array[idx] = obs['d_t']
    #     d_f_w_array[idx] = obs['d_f_w']
    #     d_f_array[idx] = obs['d_f']
    #     n_t_array[idx] = obs['n_t']
    #     n_f_w_array[idx] = obs['n_f_w']
    #     n_f_array[idx] = obs['n_f']
    
    det = detector.InterferometerList(waveform_params['ifo_list'])
    det.set_strain_data_from_zero_noise(
        sampling_frequency = waveform_params["sampling_frequency"],
        duration = waveform_params["duration"],
        start_time = waveform_params["start"],
    )
    det.inject_signal(waveform_generator=waveform_params, parameters=params)
        
    all_results = dict(
        d_t = d_t_array,
        d_f_w = d_f_w_array,
        d_f = d_f_array,
        n_t = n_t_array,
        n_f_w = n_f_w_array,
        n_f = n_f_array,
    )

    logger.info(f'Finished generating random observations')
    
    if save_results:
        filename = f'random_{n_samples}_observations.h5'
        logger.info(f'Saving results to {filename}')
    
        # Save to an HDF5 file
        with h5py.File(filename, 'w') as f:
            df_group = f.create_group('samples')
            for column in sample_params.columns:
                df_group.create_dataset(column, data=sample_params[column])
            
            arrays_group = f.create_group('observations')
            for key, array in all_results.items():
                arrays_group.create_dataset(key, data=array)
    
    return sample_params, all_results