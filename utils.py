# Yibo Yang, 2020

import tensorflow.compat.v1 as tf


def read_png(filename):
    """Loads a image file as float32 HxWx3 array; tested to work on png and jpg images."""
    string = tf.read_file(filename)
    image = tf.image.decode_image(string, channels=3)
    image = tf.cast(image, tf.float32)
    image /= 255
    return image


def quantize_image(image):
    image = tf.round(image * 255)
    image = tf.saturate_cast(image, tf.uint8)
    return image


def write_png(filename, image):
    """Saves an image to a PNG file."""
    image = quantize_image(image)
    string = tf.image.encode_png(image)
    return tf.write_file(filename, string)


def convert_float_to_uint8(image):
    image = tf.round(image * 255)
    image = tf.saturate_cast(image, tf.uint8)
    return image


def convert_uint8_to_float(image):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image


import numpy as np


# for reading images in .npy format
def read_npy_file_helper(file_name_in_bytes):
    # data = np.load(file_name_in_bytes.decode('utf-8'))
    data = np.load(file_name_in_bytes)  # turns out this works too without decoding to str first
    # assert data.dtype is np.float32   # needs to match the type argument in the caller tf.data.Dataset.map
    return data


def get_runname(args_dict, record_keys=('num_filters', 'num_hfilters', 'lmbda', 'last_step'), prefix=''):
    """
    Given a dictionary of cmdline arguments, return a string that identifies the training run.
    :param args_dict:
    :return:
    """
    config_strs = []  # ['key1=val1', 'key2=val2', ...]

    # for key, val in args_dict.items():
    #     if isinstance(val, (list, tuple)):
    #         val_str = '_'.join(map(str, val))
    #         config_strs.append('%s=%s' % (key, val_str))

    for key in record_keys:
        if key == 'num_hfilters' and int(args_dict[key]) <= 0:
            continue
        config_strs.append('%s=%s' % (key, args_dict[key]))

    return '-'.join([prefix] + config_strs)


log2pi = np.log(2. * np.pi).astype('float32')


def log_normal_pdf(sample, mean, logvar, backend=tf):
    # compute normal logpdf, element-wise
    return -.5 * ((sample - mean) ** 2. * backend.exp(-logvar) + logvar + log2pi)


def gaussian_standardized_cumulative(inputs):
    # borrowed from tensorflow_compression/python/layers/entropy_models.GaussianConditional._standardized_cumulative
    # Using the complementary error function maximizes numerical precision.
    return 0.5 * tf.math.erfc(-(2 ** -0.5) * inputs)


def box_convolved_gaussian_pdf(inputs, mu, sigma):
    # Compute the pdf of inputs under the density of N(mu, sigma**2) convolved with U(-0.5, 0.5).
    # Equivalent to N(mu, sigma**2).CDF(inputs + 0.5) - N(mu, sigma**2).CDF(inputs - 0.5), but should be more numerically
    # stable.
    values = inputs
    values -= mu
    # This assumes that the standardized cumulative has the property
    # 1 - c(x) = c(-x), which means we can compute differences equivalently in
    # the left or right tail of the cumulative. The point is to only compute
    # differences in the left tail. This increases numerical stability: c(x) is
    # 1 for large x, 0 for small x. Subtracting two numbers close to 0 can be
    # done with much higher precision than subtracting two numbers close to 1.
    values = abs(values)
    upper = gaussian_standardized_cumulative((.5 - values) / sigma)
    lower = gaussian_standardized_cumulative((-.5 - values) / sigma)
    likelihood = upper - lower
    return likelihood


@tf.custom_gradient
def round_with_STE(x, STE=None):
    """
    Special rounding that uses straight-through estimator (STE) for backpropagation.
    See a discussion in https://openreview.net/pdf?id=Skh4jRcKQ.
    :param x:
    :param STE: type of proxy function whose gradient is used in place of round in the backward pass.
    :return:
    """
    output = tf.math.round(x)

    def grad(dy):  # grad fun implement the Jacobian
        if STE is None or STE == 'identity':
            return dy
        elif STE == 'relu':
            return tf.nn.relu(dy)  # max{input, 0}
        elif STE == 'crelu' or STE == 'clipped_relu':
            return tf.clip_by_value(tf.nn.relu(dy), 0., 1.)  # min{max{input, 0}, 1}
        else:
            raise NotImplementedError

    return output, grad


# Above version of round_with_STE with kwarg won't work in graph mode. Hence have to implement various STE types separately
@tf.custom_gradient
def round_with_identity_STE(x):
    output = tf.math.round(x)
    grad = lambda dy: dy
    return output, grad


@tf.custom_gradient
def round_with_relu_STE(x):
    output = tf.math.round(x)
    grad = lambda dy: tf.nn.relu(dy)
    return output, grad


@tf.custom_gradient
def round_with_crelu_STE(x):
    output = tf.math.round(x)
    grad = lambda dy: tf.clip_by_value(tf.nn.relu(dy), 0., 1.)
    return output, grad


def annealed_temperature(t, r, ub, lb=1e-8, backend=np, scheme='exp', **kwargs):
    """
    Return the temperature at time step t, based on a chosen annealing schedule.
    :param t: step/iteration number
    :param r: decay strength
    :param ub: maximum/init temperature
    :param lb: small const like 1e-8 to prevent numerical issue when temperature gets too close to 0
    :param backend: np or tf
    :param scheme:
    :param kwargs:
    :return:
    """
    default_t0 = 700
    if scheme == 'exp':
        tau = backend.exp(-r * t)
    elif scheme == 'exp0':
        # Modified version of above that fixes temperature at ub for initial t0 iterations
        t0 = kwargs.get('t0', default_t0)
        tau = ub * backend.exp(-r * (t - t0))
    elif scheme == 'linear':
        # Cool temperature linearly from ub after the initial t0 iterations
        t0 = kwargs.get('t0', default_t0)
        tau = -r * (t - t0) + ub
    else:
        raise NotImplementedError

    if backend is None:
        return min(max(tau, lb), ub)
    else:
        return backend.minimum(backend.maximum(tau, lb), ub)
