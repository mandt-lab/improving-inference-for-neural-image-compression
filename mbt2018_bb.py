"""Modified version of mean-scale hyperprior model (mbt2018), using Gaussian posterior on latents for bits-back.

See
Yibo Yang, Robert Bamler, Stephan Mandt:
"Improving Inference for Neural Image Compression", NeurIPS 2020
https://arxiv.org/pdf/2006.04240.pdf

We have a generative model of images:
z_tilde -> y_tilde -> x
where
p(z_tilde) = flexible_cdf_dist,
p(y_tilde | z_tilde) = N(y_tilde | hyper_synthesis_transform(z_tilde)) convolved with U(-0.5, 0.5),
p(x | y_tilde) = N(x | synthesis_transform(y_tilde)

and the following inference model:
x -> y_tilde  z_tilde
   \_________/^
where
q(y_tilde | x) = U(y-0.5, y+0.5), where y = analysis_transform(x)
q(z_tilde | x) = N(z_tilde | hyper_analysis_transform(y)

We hope to get E[ H[q(z_tilde | x)] ] bits back.
"""

import os

import numpy as np
import tensorflow.compat.v1 as tf
from absl import app
from tensorflow_compression.python.ops import math_ops

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

import tensorflow_compression as tfc
from nn_models import AnalysisTransform, SynthesisTransform, HyperAnalysisTransform
from nn_models import MBT2018HyperSynthesisTransform as HyperSynthesisTransform
from utils import quantize_image

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

likelihood_lowerbound = 1e-9
variance_upperbound = 1e1


def build_graph(args, x, training=True):
    """
    Build the computational graph of the model. x should be a float tensor of shape [batch, H, W, 3].
    Given original image x, the model computes a lossy reconstruction x_tilde and various other quantities of interest.
    During training we sample from box-shaped posteriors; during compression this is approximated by rounding.
    """
    # Instantiate model.
    analysis_transform = AnalysisTransform(args.num_filters)
    synthesis_transform = SynthesisTransform(args.num_filters)
    hyper_analysis_transform = HyperAnalysisTransform(args.num_filters, num_output_filters=2 * args.num_filters)
    hyper_synthesis_transform = HyperSynthesisTransform(args.num_filters, num_output_filters=2 * args.num_filters)
    # entropy_bottleneck = tfc.EntropyBottleneck()

    # Build autoencoder and hyperprior.
    y = analysis_transform(x)

    # z_tilde ~ q(z_tilde | x) = q(z_tilde | h_a(y))
    z_mean, z_logvar = tf.split(hyper_analysis_transform(y), num_or_size_splits=2, axis=-1)
    eps = tf.random.normal(shape=tf.shape(z_mean))
    z_tilde = eps * tf.exp(z_logvar * .5) + z_mean
    from utils import log_normal_pdf
    log_q_z_tilde = log_normal_pdf(z_tilde, z_mean, z_logvar)  # bits back

    # compute the pdf of z_tilde under the flexible (hyper)prior p(z_tilde) ("z_likelihoods")
    from learned_prior import BMSHJ2018Prior
    hyper_prior = BMSHJ2018Prior(z_tilde.shape[-1], dims=(3, 3, 3))
    z_likelihoods = hyper_prior.pdf(z_tilde, stop_gradient=False)
    z_likelihoods = math_ops.lower_bound(z_likelihoods, likelihood_lowerbound)

    # compute parameters of p(y_tilde|z_tilde)
    mu, sigma = tf.split(hyper_synthesis_transform(z_tilde), num_or_size_splits=2, axis=-1)
    sigma = tf.exp(sigma)  # make positive
    if training:
        sigma = math_ops.upper_bound(sigma, variance_upperbound ** 0.5)
    if not training:  # need to handle images with non-standard sizes during compression; mu/sigma must have the same shape as y
        y_shape = tf.shape(y)
        mu = mu[:, :y_shape[1], :y_shape[2], :]
        sigma = sigma[:, :y_shape[1], :y_shape[2], :]
    scale_table = np.exp(np.linspace(np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
    conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table, mean=mu)
    # sample y_tilde from q(y_tilde|x) = U(y-0.5, y+0.5) = U(g_a(x)-0.5, g_a(x)+0.5), and then compute the pdf of
    # y_tilde under the conditional prior/entropy model p(y_tilde|z_tilde) = N(y_tilde|mu, sigma^2) conv U(-0.5, 0.5)
    # Note that at test/compression time, the resulting y_tilde doesn't simply
    # equal round(y); instead, the conditional_bottleneck does something
    # smarter and slightly more optimal: y_hat=floor(y + 0.5 - prior_mean), so
    # that the mean (mu) of the prior coincides with one of the quantization bins.
    y_tilde, y_likelihoods = conditional_bottleneck(y, training=training)

    x_tilde = synthesis_transform(y_tilde)
    if not training:
        x_shape = tf.shape(x)
        x_tilde = x_tilde[:, :x_shape[1], :x_shape[2], :]  # crop reconstruction to have the same shape as input

    return locals()


def build_train_graph(args, x):
    graph = build_graph(args, x, training=True)
    y_likelihoods, z_likelihoods, x_tilde, = graph['y_likelihoods'], graph['z_likelihoods'], graph['x_tilde']
    log_q_z_tilde = graph['log_q_z_tilde']

    # Total number of bits divided by number of pixels.
    # - log p(\tilde y | \tilde z) - log p(\tilde z) - - log q(\tilde z | \tilde y)
    num_pixels = args.batchsize * args.patchsize ** 2
    bpp_back = -tf.reduce_sum(log_q_z_tilde) / (np.log(2) * num_pixels)
    y_bpp = -tf.reduce_sum(tf.log(y_likelihoods)) / (np.log(2) * num_pixels)
    z_bpp = -tf.reduce_sum(tf.log(z_likelihoods)) / (np.log(2) * num_pixels)
    # train_bpp = (-tf.reduce_sum(tf.log(y_likelihoods)) -
    #              tf.reduce_sum(tf.log(z_likelihoods)) + tf.reduce_sum(log_q_z_tilde)) / (np.log(2) * num_pixels)
    train_bpp = y_bpp + z_bpp - bpp_back

    # Mean squared error across pixels.
    train_mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))
    # Multiply by 255^2 to correct for rescaling.
    float_train_mse = train_mse
    psnr = - 10 * (tf.log(float_train_mse) / np.log(10))  # float MSE computed on float images
    train_mse *= 255 ** 2

    # The rate-distortion cost.
    train_loss = args.lmbda * train_mse + train_bpp

    # Minimize loss and auxiliary loss, and execute update op.
    step = tf.train.create_global_step()
    main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    main_step = main_optimizer.minimize(train_loss, global_step=step)

    train_op = main_step

    model_name = os.path.splitext(os.path.basename(__file__))[0]
    original = quantize_image(x)
    reconstruction = quantize_image(x_tilde)
    return locals()


from tf_boilerplate import train, parse_args


def main(args):
    # Invoke subcommand.
    assert args.command == "train", 'Only training is supported.'
    if args.command == "train":
        train(args, build_train_graph=build_train_graph)


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
