"""Mean-scale hyperprior model (no context model), as described in "Joint Autoregressive and Hierarchical Priors for
Learned Image Compression", NeurIPS2018, by Minnen, Ballé, and Toderici (https://arxiv.org/abs/1809.02736

Also see
Yibo Yang, Robert Bamler, Stephan Mandt:
"Improving Inference for Neural Image Compression", NeurIPS 2020
https://arxiv.org/pdf/2006.04240.pdf
where this is the "base" hyperprior model (M3 in Table 1 of paper).

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
q(z_tilde | x) = U(z-0.5, z+0.5), where z = hyper_analysis_transform(y)
"""

import argparse
import glob
import sys
import os

from absl import app
from absl.flags import argparse_flags
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow_compression.python.ops import math_ops

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

import tensorflow_compression as tfc
from nn_models import AnalysisTransform, SynthesisTransform, HyperAnalysisTransform
from nn_models import MBT2018HyperSynthesisTransform as HyperSynthesisTransform
from utils import read_png, quantize_image, write_png, read_npy_file_helper, get_runname

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def build_graph(args, x, training=True):
    """
    Build the computational graph of the model. x should be a float tensor of shape [batch, H, W, 3].
    Given original image x, the model computes a lossy reconstruction x_tilde and various other quantities of interest.
    During training we sample from box-shaped posteriors; during compression this is approximated by rounding.
    """
    # Instantiate model.
    analysis_transform = AnalysisTransform(args.num_filters)
    synthesis_transform = SynthesisTransform(args.num_filters)
    hyper_analysis_transform = HyperAnalysisTransform(args.num_filters)
    hyper_synthesis_transform = HyperSynthesisTransform(args.num_filters, num_output_filters=2 * args.num_filters)
    entropy_bottleneck = tfc.EntropyBottleneck()

    # Build autoencoder and hyperprior.
    y = analysis_transform(x)  # y = g_a(x)
    z = hyper_analysis_transform(y)  # z = h_a(y)

    # sample z_tilde from q(z_tilde|x) = q(z_tilde|h_a(g_a(x))), and compute the pdf of z_tilde under the flexible prior
    # p(z_tilde) ("z_likelihoods")
    z_tilde, z_likelihoods = entropy_bottleneck(z, training=training)
    mu, sigma = tf.split(hyper_synthesis_transform(z_tilde), num_or_size_splits=2, axis=-1)
    sigma = tf.exp(sigma)  # make positive
    if not training:  # need to handle images with non-standard sizes during compression; mu/sigma must have the same shape as y
        y_shape = tf.shape(y)
        mu = mu[:, :y_shape[1], :y_shape[2], :]
        sigma = sigma[:, :y_shape[1], :y_shape[2], :]
    scale_table = np.exp(np.linspace(np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
    conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table, mean=mu)
    # sample y_tilde from q(y_tilde|x) = U(y-0.5, y+0.5) = U(g_a(x)-0.5, g_a(x)+0.5), and then compute the pdf of
    # y_tilde under the conditional prior/entropy model p(y_tilde|z_tilde) = N(y_tilde|mu, sigma^2) conv U(-0.5,  0.5)
    y_tilde, y_likelihoods = conditional_bottleneck(y, training=training)
    x_tilde = synthesis_transform(y_tilde)

    if not training:
        side_string = entropy_bottleneck.compress(z)
        string = conditional_bottleneck.compress(y)
        x_shape = tf.shape(x)
        x_tilde = x_tilde[:, :x_shape[1], :x_shape[2], :]  # crop reconstruction to have the same shape as input

    return locals()


def build_train_graph(args, x):
    graph = build_graph(args, x, training=True)
    y_likelihoods, z_likelihoods, x_tilde, = graph['y_likelihoods'], graph['z_likelihoods'], graph['x_tilde']
    entropy_bottleneck = graph['entropy_bottleneck']
    # Total number of bits divided by number of pixels.
    # - log p(\tilde y | \tilde z) - log p(\tilde z)
    num_pixels = args.batchsize * args.patchsize ** 2
    y_bpp = -tf.reduce_sum(tf.log(y_likelihoods)) / (np.log(2) * num_pixels)
    z_bpp = -tf.reduce_sum(tf.log(z_likelihoods)) / (np.log(2) * num_pixels)
    # train_bpp = (-tf.reduce_sum(tf.log(y_likelihoods)) -
    #              tf.reduce_sum(tf.log(z_likelihoods))) / (np.log(2) * num_pixels)
    train_bpp = y_bpp + z_bpp

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

    aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

    train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])

    model_name = os.path.splitext(os.path.basename(__file__))[0]
    original = quantize_image(x)
    reconstruction = quantize_image(x_tilde)
    return locals()


def compress(args):
    """Compresses an image, or a batch of images of the same shape in npy format."""
    from configs import get_eval_batch_size, write_tfci_for_eval

    if args.input_file.endswith('.npy'):
        # .npy file should contain N images of the same shapes, in the form of an array of shape [N, H, W, 3]
        X = np.load(args.input_file)
    else:
        # Load input image and add batch dimension.
        from PIL import Image
        x = np.asarray(Image.open(args.input_file).convert('RGB'))
        X = x[None, ...]

    num_images = int(X.shape[0])
    num_pixels = int(np.prod(X.shape[1:-1]))
    X = X.astype('float32')
    X /= 255.

    eval_batch_size = get_eval_batch_size(num_pixels)
    dataset = tf.data.Dataset.from_tensor_slices(X)
    dataset = dataset.batch(batch_size=eval_batch_size)
    # https://www.tensorflow.org/api_docs/python/tf/compat/v1/data/Iterator
    # Importantly, each sess.run(op) call will consume a new batch, where op is any operation that depends on
    # x. Therefore if multiple ops need to be evaluated on the same batch of data, they have to be grouped like
    # sess.run([op1, op2, ...]).
    x_next = dataset.make_one_shot_iterator().get_next()

    x_ph = x = tf.placeholder('float32', (None, *X.shape[1:]))  # keep a reference around for feed_dict
    graph = build_graph(args, x, training=False)
    y_likelihoods, z_likelihoods, x_tilde = graph['y_likelihoods'], graph['z_likelihoods'], graph['x_tilde']
    string, side_string = graph['string'], graph['side_string']

    # graph = build_graph(args, x, training=False)
    # y_likelihoods, z_likelihoods, x_tilde, = graph['y_likelihoods'], graph['z_likelihoods'], graph['x_tilde']
    # string, side_string = graph['string'], graph['side_string']
    # Total number of bits divided by number of pixels.
    axes_except_batch = list(range(1, len(x.shape)))  # should be [1,2,3]
    y_bpp = tf.reduce_sum(-tf.log(y_likelihoods), axis=axes_except_batch) / (np.log(2) * num_pixels)
    z_bpp = tf.reduce_sum(-tf.log(z_likelihoods), axis=axes_except_batch) / (np.log(2) * num_pixels)
    eval_bpp = y_bpp + z_bpp  # shape (N,)

    # Bring both images back to 0..255 range.
    x *= 255
    x_tilde = tf.clip_by_value(x_tilde, 0, 1)
    x_tilde = tf.round(x_tilde * 255)

    mse = tf.reduce_mean(tf.squared_difference(x, x_tilde), axis=axes_except_batch)  # shape (N,)
    psnr = tf.image.psnr(x_tilde, x, 255)  # shape (N,)
    msssim = tf.image.ssim_multiscale(x_tilde, x, 255)  # shape (N,)
    msssim_db = -10 * tf.log(1 - msssim) / np.log(10)  # shape (N,)
    x_shape = graph['x_shape']
    y_shape = graph['y_shape']
    z_shape = tf.shape(graph['z'])

    with tf.Session() as sess:
        # Load the latest model checkpoint, get the compressed string and the tensor
        # shapes.
        save_dir = os.path.join(args.checkpoint_dir, args.runname)
        latest = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
        ckpt_last_step = int(os.path.basename(latest).split('.ckpt-')[-1])  # "model.ckpt-1000000" -> 1000000
        tf.train.Saver().restore(sess, save_path=latest)
        eval_fields = ['mse', 'psnr', 'msssim', 'msssim_db', 'est_bpp', 'est_y_bpp', 'est_z_bpp']
        eval_tensors = [mse, psnr, msssim, msssim_db, eval_bpp, y_bpp, z_bpp]
        all_results_arrs = {key: [] for key in eval_fields}  # append across all batches

        compression_tensors = [string, side_string, x_shape[1:-1], y_shape[1:-1], z_shape[1:-1]]
        batch_actual_bpp = []
        batch_sizes = []

        batch_idx = 0
        while True:
            try:
                x_val = sess.run(x_next)
                x_feed_dict = {x_ph: x_val}

                # If requested, transform the quantized image back and measure performance.
                eval_arrs = sess.run(eval_tensors, feed_dict=x_feed_dict)
                for field, arr in zip(eval_fields, eval_arrs):
                    all_results_arrs[field] += arr.tolist()

                # Write a binary file with the shape information and the compressed string.
                packed = tfc.PackedTensors()
                compression_arrs = sess.run(compression_tensors, feed_dict=x_feed_dict)

                packed.pack(compression_tensors, compression_arrs)
                if write_tfci_for_eval:
                    with open(args.output_file, "wb") as f:
                        f.write(packed.string)

                # The actual bits per pixel including overhead.
                batch_actual_bpp.append(
                    len(packed.string) * 8 / num_pixels)  # packed.string is the encoding for the entire batch
                batch_sizes.append(len(eval_arrs[0]))

                batch_idx += 1

            except tf.errors.OutOfRangeError:
                break

        for field in eval_fields:
            all_results_arrs[field] = np.asarray(all_results_arrs[field])

        all_results_arrs['batch_actual_bpp'] = np.asarray(batch_actual_bpp)
        all_results_arrs['batch_sizes'] = np.asarray(batch_sizes)

        avg_batch_actual_bpp = np.sum(batch_actual_bpp) / np.sum(batch_sizes)
        eval_fields.append('avg_batch_actual_bpp')
        all_results_arrs['avg_batch_actual_bpp'] = avg_batch_actual_bpp

        input_file = os.path.basename(args.input_file)
        results_dict = all_results_arrs
        np.savez(os.path.join(args.results_dir, 'rd-%s-last_step=%d-file=%s.npz'
                              % (args.runname, ckpt_last_step, input_file)), **results_dict)
        for field in eval_fields:
            arr = all_results_arrs[field]
            print('Avg {}: {:0.4f}'.format(field, arr.mean()))


def decompress(args):
    """Decompresses an image."""
    # Adapted from https://github.com/tensorflow/compression/blob/master/examples/bmshj2018.py
    # Read the shape information and compressed string from the binary file.
    string = tf.placeholder(tf.string, [1])
    side_string = tf.placeholder(tf.string, [1])
    x_shape = tf.placeholder(tf.int32, [2])
    y_shape = tf.placeholder(tf.int32, [2])
    z_shape = tf.placeholder(tf.int32, [2])
    with open(args.input_file, "rb") as f:
        packed = tfc.PackedTensors(f.read())
    tensors = [string, side_string, x_shape, y_shape, z_shape]
    arrays = packed.unpack(tensors)

    # Instantiate model. TODO: automate this with build_graph
    synthesis_transform = SynthesisTransform(args.num_filters)
    hyper_synthesis_transform = HyperSynthesisTransform(args.num_filters, num_output_filters=2 * args.num_filters)
    entropy_bottleneck = tfc.EntropyBottleneck(dtype=tf.float32)

    # Decompress and transform the image back.
    z_shape = tf.concat([z_shape, [args.num_filters]], axis=0)
    z_hat = entropy_bottleneck.decompress(
        side_string, z_shape, channels=args.num_filters)

    mu, sigma = tf.split(hyper_synthesis_transform(z_hat), num_or_size_splits=2, axis=-1)
    sigma = tf.exp(sigma)  # make positive
    training = False
    if not training:  # need to handle images with non-standard sizes during compression; mu/sigma must have the same shape as y
        mu = mu[:, :y_shape[0], :y_shape[1], :]
        sigma = sigma[:, :y_shape[0], :y_shape[1], :]
    scale_table = np.exp(np.linspace(np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
    conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table, mean=mu, dtype=tf.float32)
    y_hat = conditional_bottleneck.decompress(string)
    x_hat = synthesis_transform(y_hat)

    # Remove batch dimension, and crop away any extraneous padding on the bottom
    # or right boundaries.
    x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]

    # Write reconstructed image out as a PNG file.
    op = write_png(args.output_file, x_hat)

    # Load the latest model checkpoint, and perform the above actions.
    with tf.Session() as sess:
        save_dir = os.path.join(args.checkpoint_dir, args.runname)
        latest = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
        tf.train.Saver().restore(sess, save_path=latest)
        sess.run(op, feed_dict=dict(zip(tensors, arrays)))


from tf_boilerplate import train, parse_args


def main(args):
    # Invoke subcommand.
    if args.command == "train":
        train(args, build_train_graph=build_train_graph)
    elif args.command == "compress":
        if not args.output_file:
            args.output_file = args.input_file + ".tfci"
        compress(args)
        # compress_est_ideal_rate(args)
    elif args.command == "decompress":
        if not args.output_file:
            args.output_file = args.input_file + ".png"
        decompress(args)


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
