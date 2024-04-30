
import bilby
from bilby.gw import source, conversion
from bilby.gw import prior as prior_gw
from bilby.core import prior as prior_core
bilby.core.utils.logger.setLevel("WARNING")

sourced = dict(
    source_type = 'BBH',
    aligned_spins = False,
    fd_source_model = source.lal_binary_black_hole,
    param_conversion_model = conversion.convert_to_lal_binary_black_hole_parameters,
)

waveform_params = dict(
    sampling_frequency = 2048,
    duration = 4.0,
    start_offset = 2.0,
    start = -2.0,
    waveform_apprx = 'SEOBNRv4PHM',
    minimum_frequency = 20,
    maximum_frequency = 1024,
    reference_frequency = 50,
    ifo_list = ['H1','L1','V1'],
    ifo_noise = True,
)

default_injection = dict(
    #mass_1 = 39.536,
    #mass_2 = 34.872,
    mass_ratio = 0.8858,
    chirp_mass = 32.14,
    theta_jn = 0.4432,
    phase = 5.089,
    tilt_1 = 1.497,
    tilt_2 = 1.102,
    a_1 = 0.9702,
    a_2 = 0.8118,
    phi_12 = 6.220,
    phi_jl = 1.885,
    luminosity_distance = 900.0,
    dec = 0.071,
    ra = 5.556,
    psi = 1.100,
    geocent_time = 0.0,
)

intrinsic_variables = ['mass_ratio', 'chirp_mass', 'theta_jn', 'phase', 'tilt_1', 'tilt_2', 'a_1', 'a_2', 'phi_12', 'phi_jl']
extrinsic_variables = ['luminosity_distance', 'dec', 'ra', 'psi', 'geocent_time']

limits = dict(
    mass_ratio = [0.125, 1.0],
    chirp_mass = [25.0, 100.0],
    theta_jn = [0.0, 3.14159],
    phase = [0.0, 6.28318],
    tilt_1 = [0.0, 3.14159],
    tilt_2 = [0.0, 3.14159],
    a_1 = [0.05, 1.0],
    a_2 = [0.05, 1.0],
    phi_12 = [0.0, 6.28318],
    phi_jl = [0.0, 6.28318],
    luminosity_distance = [100, 1500],
    dec = [-1.57079, 1.57079],
    ra = [0.0, 6.28318],
    psi = [0.0, 3.14159],
    geocent_time = [-0.1, 0.1],
)

priors_gw = dict(
    mass_ratio = getattr(prior_core, 'Uniform')(minimum=0.125, maximum=1, boundary='reflective'),
    chirp_mass = getattr(prior_core, 'Uniform')(minimum=25, maximum=100, boundary='reflective'),
    theta_jn = prior_gw.BBHPriorDict()['theta_jn'],
    phase = prior_gw.BBHPriorDict()['phase'],
    tilt_1 = prior_gw.BBHPriorDict()['tilt_1'],
    tilt_2 = prior_gw.BBHPriorDict()['tilt_2'],
    a_1 = prior_gw.BBHPriorDict()['a_1'],
    a_2 = prior_gw.BBHPriorDict()['a_2'],
    phi_12 = prior_gw.BBHPriorDict()['phi_12'],
    phi_jl = prior_gw.BBHPriorDict()['phi_jl'],
    luminosity_distance = prior_gw.BBHPriorDict()['luminosity_distance'],
    dec = prior_gw.BBHPriorDict()['dec'],
    ra = prior_gw.BBHPriorDict()['ra'],
    psi = prior_gw.BBHPriorDict()['psi'],
    geocent_time = getattr(prior_core, 'Uniform')(minimum=-0.1, maximum=0.1, boundary=None),
)

default_conf = {
    'source' : sourced,
    'waveform_params' : waveform_params,
    'injection' : default_injection,
    'priors' : {
        'int_priors' : {key: priors_gw[key] for key in intrinsic_variables},
        'ext_priors' : {key: priors_gw[key] for key in extrinsic_variables},
    },
}




# {'source': {'source_type': 'BBH', 'aligned_spins': False, 'fd_source_model': <function lal_binary_black_hole at 0x14ddab911620>, 'param_conversion_model': <function convert_to_lal_binary_black_hole_parameters at 0x14ddab8f5f80>},
# 'waveform_params': {'sampling_frequency': 2048, 'duration': 4.0, 'start_offset': 2.0, 'start': -2.0, 'waveform_apprx': 'SEOBNRv4PHM', 'minimum_frequency': 20, 'maximum_frequency': 1024, 'reference_frequency': 50, 'ifo_list': ['H1', 'L1', 'V1'], 'ifo_noise': True},
# 'injection': {'mass_ratio': 0.8858, 'chirp_mass': 32.14, 'luminosity_distance': 900.0, 'dec': 0.071, 'ra': 5.556, 'theta_jn': 0.4432, 'psi': 1.1, 'phase': 5.089, 'tilt_1': 1.497, 'tilt_2': 1.102, 'a_1': 0.9702, 'a_2': 0.8118, 'phi_12': 6.22, 'phi_jl': 1.885, 'geocent_time': 0.0},
# 'priors': {'int_priors': {'mass_ratio': Uniform(minimum=0.125, maximum=1.0, name=None, latex_label=None, unit=None, boundary='reflective'), 'chirp_mass': Uniform(minimum=25.0, maximum=100.0, name=None, latex_label=None, unit=None, boundary='reflective'), 'theta_jn': Sine(minimum=0.0, maximum=3.14159, name='theta_jn', latex_label='$\\theta_{JN}$', unit=None, boundary=None), 'phase': Uniform(minimum=0.0, maximum=6.28318, name='phase', latex_label='$\\phi$', unit=None, boundary='periodic'), 'tilt_1': Sine(minimum=0.0, maximum=3.14159, name='tilt_1', latex_label='$\\theta_1$', unit=None, boundary=None), 'tilt_2': Sine(minimum=0.0, maximum=3.14159, name='tilt_2', latex_label='$\\theta_2$', unit=None, boundary=None), 'a_1': Uniform(minimum=0.05, maximum=1.0, name='a_1', latex_label='$a_1$', unit=None, boundary=None), 'a_2': Uniform(minimum=0.05, maximum=1.0, name='a_2', latex_label='$a_2$', unit=None, boundary=None), 'phi_12': Uniform(minimum=0.0, maximum=6.28318, name='phi_12', latex_label='$\\Delta\\phi$', unit=None, boundary='periodic'), 'phi_jl': Uniform(minimum=0.0, maximum=6.28318, name='phi_jl', latex_label='$\\phi_{JL}$', unit=None, boundary='periodic')},
# 'ext_priors': {'luminosity_distance': bilby.gw.prior.UniformSourceFrame(minimum=100.0, maximum=1500.0, cosmology='Planck15', name='luminosity_distance', latex_label='$d_L$', unit='Mpc', boundary=None), 'dec': Cosine(minimum=-1.57079, maximum=1.57079, name='dec', latex_label='$\\mathrm{DEC}$', unit=None, boundary=None), 'ra': Uniform(minimum=0.0, maximum=6.28318, name='ra', latex_label='$\\mathrm{RA}$', unit=None, boundary='periodic'), 'psi': Uniform(minimum=0.0, maximum=3.14159, name='psi', latex_label='$\\psi$', unit=None, boundary='periodic'), 'geocent_time': Uniform(minimum=-0.1, maximum=0.1, name=None, latex_label=None, unit=None, boundary=None)}},
# 'fixed': [], 'varying': ['mass_ratio', 'chirp_mass', 'luminosity_distance', 'dec', 'ra', 'theta_jn', 'psi', 'phase', 'tilt_1', 'tilt_2', 'a_1', 'a_2', 'phi_12', 'phi_jl', 'geocent_time'], 'param_idxs': {'mass_ratio': 0, 'chirp_mass': 1, 'theta_jn': 2, 'phase': 3, 'tilt_1': 4, 'tilt_2': 5, 'a_1': 6, 'a_2': 7, 'phi_12': 8, 'phi_jl': 9, 'luminosity_distance': 10, 'dec': 11, 'ra': 12, 'psi': 13, 'geocent_time': 14},
# 'vary_idxs': [0, 1, 10, 11, 12, 2, 13, 3, 4, 5, 6, 7, 8, 9, 14], 'param_names': {0: 'mass_ratio', 1: 'chirp_mass', 2: 'theta_jn', 3: 'phase', 4: 'tilt_1', 5: 'tilt_2', 6: 'a_1', 7: 'a_2', 8: 'phi_12', 9: 'phi_jl', 10: 'luminosity_distance', 11: 'dec', 12: 'ra', 13: 'psi', 14: 'geocent_time'},
# 'tmnre': {'infer_only': True, 'marginals': ((0, 1),), 'num_rounds': 8, '1d_only': True, 'bounds_th': 1e-05, 'resampler': False, 'shuffling': True, 'noise_targets': ['n_t', 'n_f_w'], 'generate_obs': False, 'obs_path': '/scratch-shared/scur2012/peregrine_data/bhardwaj2023/observation_highSNR'},
# 'sampling': {'sampler': 'dynesty', 'npoints': 2000, 'nsamples': 2000, 'printdt': 5, 'walks': 100, 'nact': 10, 'ntemps': 10, 'nlive': 2000, 'nwalkers': 10, 'distance': False, 'time': False, 'phase': True, 'resume_from_ckpt': True},
# 'sampler_hparams': {'bilby_mcmc': {'n_samples': 2000, 'n_temps': 10, 'printdt': 5},
# 'dynesty': {'nlive': 2000, 'walks': 100, 'nact': 10},
# 'cpnest': {'nlive': 2000, 'walks': 100, 'nact': 10},
# 'pymultinest': {'npoints': 2000},
# 'ptemcee': {'ntemps': 10, 'nwalkers': 10, 'nsamples': 2000}},
# 'zarr_params': {'run_id': 'highSNR', 'use_zarr': True, 'sim_schedule': [30000, 60000, 90000, 120000, 120000, 150000, 150000, 150000], 'chunk_size': 1000, 'run_parallel': True, 'njobs': 16, 'targets': ['z_int', 'z_ext', 'z_total', 'd_t', 'd_f', 'd_f_w', 'n_t', 'n_f', 'n_f_w'], 'store_path': '/scratch-shared/scur2012/peregrine_data/bhardwaj2023'},
# 'hparams': {'min_epochs': 30, 'max_epochs': 200, 'early_stopping': 7, 'learning_rate': 0.0005, 'num_workers': 8, 'training_batch_size': 256, 'validation_batch_size': 256, 'train_data': 0.9, 'val_data': 0.1},
# 'device_params': {'device': 'gpu', 'n_devices': 1}}
