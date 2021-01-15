# -*- coding: utf-8 -*-
"""
@author: Daniel Jiménez-Caminero Costa
"""
import cmath
import sys

sys.path.append('../../..')

import numpy as np
import math
import scipy as sp
from scipy.signal import welch
import matplotlib.pyplot as plt

# Local functions imports
from mosqito.functions.shared.load import load


def spec_loudness(signal, rs):
    """ Specific Loudness

    It describes the loudness excitation in a critical band per Bark. In this case, the function outputs an array of 53
    specific loudness values,

    Parameters
    ----------
    signal : numpy.array
        time signal values

    rs : integer
        sampling frequency

    Outputs
    -------
    specific_loudness_array: float
        'Sone per Bark'

    total_loudness: float
        'Sone per Bark'
    """

    # The array signal returned by "wavfile.read" is a numpy array with an integer data type. The data type of a
    # numpy array can not be changed in place IMPORTANT INFORMATION ABOUT "READ"
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html "Data read from WAV file.
    # Data-type is determined from the file; see Notes. Data is 1-D for 1-channel WAV, or 2-D of shape (Nsamples,
    # Nchannels) otherwise."

    # The High Order filter that is needed here is a DIGITAL BIQUAD FILTER
    filter_b = [
        [1.0159, -1.9253, 0.9221],
        [0.9589, -1.8061, 0.8764],
        [0.9614, -1.7636, 0.8218],
        [2.2258, -1.4347, -0.4982],
        [0.4717, -0.3661, 0.2441],
        [0.1153, 0.0000, -0.1153],
        [0.9880, -1.9124, 0.9261],
        [1.9522, 0.1623, -0.6680]
    ]
    filter_a = [
        [1, -1.9253, 0.9380],
        [1, -1.8061, 0.8354],
        [1, -1.7636, 0.7832],
        [1, -1.4347, 0.7276],
        [1, -0.3661, -0.2841],
        [1, -1.7960, 0.8058],
        [1, -1.9124, 0.9142],
        [1, 0.1623, 0.2842]
    ]

    # SIGNAL FILTERING
    # signal_filtered is pom(n)
    signal_filtered = signal

    # In serially-cascaded second-order filters the output from the n filter is the input of  n+1. The result of this
    # loop is a filtered signal
    for ear_filter_number in range(len(filter_a)):
        signal_filtered = sp.signal.lfilter(filter_b[ear_filter_number], filter_a[ear_filter_number], signal_filtered)

    # AUDITORY FILTERING BANK
    filter_order_k = 5
    # rs = 48000, it is passed to the function
    e_coefficients_array = [0, 1, 11, 11, 1]
    d_coefficients_array = []
    z_step_size = 0.5
    af_f0 = 81.9289
    c = 0.1618
    # Centre Frequencies F(z), 26.5/0.5 = 53
    centre_freq_array = []
    # F Bandwidth Af(z)
    f_bandwidth_array = []
    # Retardation
    t_delay_array = []
    # Exponent of the delay
    exponent_1 = (2 * filter_order_k) - 1
    # Binomial coefficient
    n_binomial = (2 * filter_order_k) - 2
    k_binomial = filter_order_k - 1
    dif_binomial = n_binomial - k_binomial
    binomial_coef_1 = (math.factorial(n_binomial) / (math.factorial(k_binomial) * math.factorial(dif_binomial)))
    # Block length and hop size
    sb_array = []
    sh_array = []
    # Coefficient Calculation
    # As we want to implement band-pass filters that mimic the auditory filters (F.9 formula, Annex F), we must use
    # the modified filter coefficients present in F.13 and F.14 formulas, Annex F
    am_mod_coefficient_array = np.zeros((53, filter_order_k), dtype=complex)
    bm_mod_coefficient_array = np.zeros((53, filter_order_k), dtype=complex)
    band_pass_signal_array = []

    # ROOT-MEAN-SQUARE VALUES
    band_pass_block_array = []
    rms_values_array = []

    # NON-LINEARITY. SOUND PRESSURE INTO SPECIFIC LOUDNESS
    p_0 = 20 * math.pow(10, (-6))
    c_N = 0.0217406
    # c_N: In sones/bark. Assures that the total loudness of a sinusoid having a frequency of 1 kHz and a sound
    # pressure level of 40 dB equals 1 sone.
    alpha = 1.5
    M_exponents = 8
    a_array = []
    v_i_array = [1, 0.6602, 0.0864, 0.6384, 0.0328, 0.4068, 0.2082, 0.3994, 0.6434]
    threshold_db_array = [15, 25, 35, 45, 55, 65, 75, 85]

    # CONSIDERATION OF THRESHOLD IN QUIET
    # The specific loudness in each band 𝑧 is zero if it is at or below a critical-band-dependent specific loudness
    # threshold LTQ(𝑧)
    ltq_z = [0.3310, 0.1625, 0.1051, 0.0757, 0.0576, 0.0453, 0.0365, 0.0298, 0.0247, 0.0207, 0.0176, 0.0151, 0.0131,
             0.0115, 0.0103, 0.0093, 0.0086, 0.0081, 0.0077, 0.0074, 0.0073, 0.0072, 0.0071, 0.0072, 0.0073, 0.0074,
             0.0076, 0.0079, 0.0082, 0.0086, 0.0092, 0.0100, 0.0109, 0.0202, 0.0217, 0.0237, 0.0263, 0.0296, 0.0339,
             0.0398, 0.0485, 0.0622]
    specific_loudness_array = []

    # TOTAL LOUDNESS
    total_loudness = 0
    a_z = 0.5

    for band_number in range(53):
        z = (band_number + 1) * z_step_size
        var = c * z
        centre_freq = (af_f0 / c) * math.sinh(var)
        centre_freq_array.append(centre_freq)

        f_bandwidth = math.sqrt(math.pow(af_f0, 2) + math.pow(c * centre_freq, 2))
        f_bandwidth_array.append(f_bandwidth)

        t_delay = (1 / (math.pow(2, exponent_1))) * binomial_coef_1 * (1 / f_bandwidth)
        t_delay_array.append(t_delay)

        d_coefficients = math.exp(-(1 / (rs * t_delay)))
        d_coefficients_array.append(d_coefficients)

        # Block length and hop size, for further calculations (Root-Mean-Square Values)
        if z >= 13:
            sb_array.append(1024)
            sh_array.append(256)
        elif 8.5 <= f_bandwidth < 13:
            sb_array.append(2048)
            sh_array.append(512)
        elif 2 <= f_bandwidth < 8.5:
            sb_array.append(4096)
            sh_array.append(1024)
        else:
            sb_array.append(8192)
            sh_array.append(2048)

        # Coefficient Calculation
        for m in range(filter_order_k):
            # Binomial coefficient
            dif_binomial_2 = filter_order_k - m
            binomial_coef_2 = (math.factorial(filter_order_k) / (math.factorial(m) * math.factorial(dif_binomial_2)))
            am_coefficient = math.pow((-(d_coefficients_array[band_number])), m) * binomial_coef_2
            # "math.exp()" doesn't support complex arguments. "cmath.exp()" is needed:
            am_mod_coefficient = am_coefficient * cmath.exp(-((1j * 2 * math.pi * centre_freq_array[band_number] * m)
                                                              / rs))
            print(am_mod_coefficient)
            am_mod_coefficient_array[band_number][m] = am_mod_coefficient

            bm_sum = 0
            for j in range(filter_order_k - 1):
                z = j + 1
                bm_sum = bm_sum + (e_coefficients_array[z] * math.pow(d_coefficients_array[band_number], z))

            bm_coefficient = (math.pow((1 - d_coefficients_array[band_number]), filter_order_k)
                              / bm_sum) * d_coefficients_array[band_number] * e_coefficients_array[m]
            bm_mod_coefficient = bm_coefficient * cmath.exp(
                -((1j * 2 * math.pi * centre_freq_array[band_number] * m) / rs))
            bm_mod_coefficient_array[band_number][m] = bm_mod_coefficient

        # Band-Pass Signals pom,z (n) centred around the critical band rate scale values z. 50% of overlap
        # We have to pass all the coefficients of each filter, each time, that is why we do not do further inspection
        # inside the array
        band_pass_signal = 2 * sp.signal.lfilter(bm_mod_coefficient_array[band_number],
                                                 am_mod_coefficient_array[band_number], signal_filtered).imag

        # Time representation of the band-passed signal Vs. The filtered signal
        # for printable in range(len(band_pass_signal)):
        #     print(band_pass_signal[printable])
        plt.figure()
        t = np.linspace(-2 * math.pow(10, -7), 2 * math.pow(10, -7), 201)
        plt.plot(band_pass_signal, 'b')
        plt.grid(True)
        plt.show()

        # RECTIFICATION for the activation of the basilar membrane using numpy's indexing function
        band_pass_signal[band_pass_signal <= 0] = 0
        band_pass_signal_array.append(band_pass_signal)

        # ROOT-MEAN-SQUARE
        factor_2_summ = 0
        sequence_product = 0

        for n_segmentation in range(sb_array[band_number]):
            # n_segmentation (n') goes from 0 to sb(band-dependent block size) - 1

            if n_segmentation < (sb_array[band_number] - (band_number * sh_array[band_number])):
                b_value = 0
                band_pass_block_array.append(b_value)
            else:
                # I have doubts about the following displacement: (It may be better to use shift, from scipy)
                position = (band_number * sh_array[band_number] + n_segmentation - sb_array[band_number])
                b_value = band_pass_signal_array[band_number][position]
                band_pass_block_array[band_number][n_segmentation].append(b_value)
                # After these operations, we have obtained an array of signals, and their block values

            factor_2_summ = factor_2_summ + math.pow(b_value, 2)

        rms_value = math.sqrt((2 / sb_array[band_number]) * factor_2_summ)
        rms_values_array.append(rms_value)
        # The result is the RMS-values of the blocks

        # NON-LINEARITY
        for exponent_position in range(M_exponents):
            exponent_number = (exponent_position + 1)
            a_exponent = ((v_i_array[j] - v_i_array[exponent_position]) / alpha)
            pt_threshold = p_0 * math.pow(10, (threshold_db_array[exponent_position] / 20))
            a_base = math.pow((1 + (rms_values_array[band_number] / pt_threshold)), alpha)
            if exponent_number == 1:
                sequence_product = math.pow(a_base, a_exponent)
            else:
                sequence_product = sequence_product * math.pow(a_base, a_exponent)

        a_value = c_N * (rms_value / p_0) * sequence_product
        a_array.append(a_value)

        # CONSIDERATION OF THRESHOLD IN QUIET
        specific_loudness = a_array[band_number] - ltq_z[band_number]
        if specific_loudness < ltq_z[band_number]:
            specific_loudness_array.append(0)
        else:
            specific_loudness_array.append(specific_loudness)

        # TOTAL LOUDNESS
        total_loudness = total_loudness + (a_array[band_number] * a_z)

    return specific_loudness_array, total_loudness


def comp_tonality(tonality_type, signal, fs):
    """ Tonality Calculation

    The concept of tonality has two interpretations: the classic one gives
    an idea of how powerful a tone is in a broadband noise, and the other
    one, psycho-acoustic tonality that tells how perceivable a tone is from
    the background noise.

    Parameters
    ----------
    tonality_type : string
        The type of tonality wanted to be analysed

    lt : int
        Overall sound pressure level of the input signal

    unit : string (unit of Lt)
        'dB' or 'dBA'

    Outputs
    -------
    Value : definition
    """
    # NOTA: El loudness lo obtenemos de las funciones de zwicker
    # No se calcula aquí, solo se pasan los parametros calculados en el test
    tonality_value = 0

    return tonality_value


# calibration = 2 * pow(2, 0.5)
# signal, fs = load(True, file, calibration)
# comp_tonality(signal, fs)
# Test signal as input for stationary loudness
# (from ISO 532-1 annex B3)
# signal = {
#     "data_file": "mosqito/tests/loudness/data/ISO_532-1/Test signal 3 (1 kHz 60 dB).wav",
#     "N": 4.019,
#     "N_specif_file": "mosqito/tests/loudness/data/ISO_532-1/test_signal_3.csv",
# }
# wav_file = {
#     "data_file": r"D:\PycharmProjects\tonality_ecma\mosqito\tests\tonality_ecma\car_engine_1.wav"
# }
wav_file = {
    "data_file": r"D:\PycharmProjects\tonality_ecma\mosqito\tests\tonality_ecma\sweep_freq.wav"
}

# Load signal
# sig, fs = load(True, signal["data_file"], calib=1)
sig, fs = load(False, wav_file["data_file"], calib=2 * 2 ** 0.5)

s_l, t_l = spec_loudness(sig, fs)

for i in range(len(s_l)):
    print(s_l[i])

print(t_l)

# narrowband = np.zeros((21), dtype = dict)
#
# narrowband[0] = { "data_file": r"data\Check_signals_DIN_45692_(Schaerfe)\Narrowband_noise (frequency group
# width)\BP250.wav", "type": "Narrow-band", "S": 0.38 } narrowband[1] = { "data_file":
# r"data\Check_signals_DIN_45692_(Schaerfe)\Narrowband_noise (frequency group width)\BP350.wav", "S": 0.49 }
# narrowband[2] = { "data_file": r"data\Check_signals_DIN_45692_(Schaerfe)\Narrowband_noise (frequency group
# width)\BP450.wav", "S": 0.6 } narrowband[3] = { "data_file": r"data\Check_signals_DIN_45692_(
# Schaerfe)\Narrowband_noise (frequency group width)\BP570.wav", "S": 0.71 }

# validation_sharpness(narrowband)

# for i in range(len(noise)):
#     # Load signal
#     sig, fs = load(True, noise[i]["data_file"], calib=1)
#
#     # Compute sharpness
#     S = comp_sharpness(True, sig, fs, method='din')
#     sharpness[i] = S['values']
#
#
# signal, fs = load(True, file, 2 * 2 ** 0.5)
