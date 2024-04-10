
from bilby.gw import source, conversion
from bilby.gw import prior as prior_gw
from bilby.core import prior as prior_core

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

default_injection = dict(
    mass_1 = 39.536,
    mass_2 = 34.872,
    mass_ratio = 0.8858,
    chirp_mass = 32.14,
    luminosity_distance = 200,
    dec = 0.071,
    ra = 5.556,
    theta_jn = 0.4432,
    psi = 1.100,
    phase = 5.089,
    tilt_1 = 1.497,
    tilt_2 = 1.102,
    a_1 = 0.9702,
    a_2 = 0.8118,
    phi_12 = 6.220,
    phi_jl = 1.885,
    geocent_time = 0.0,
)

intrinsic_variables = ['mass_ratio', 'chirp_mass', 'luminosity_distance', 'dec', 'ra', 'theta_jn', 'psi', 'phase', 'tilt_1', 'tilt_2']
extrinsic_variables = ['a_1', 'a_2', 'phi_12', 'phi_jl', 'geocent_time']

limits = dict(
    mass_ratio = [0.125, 1.0],
    chirp_mass = [25, 100],
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
