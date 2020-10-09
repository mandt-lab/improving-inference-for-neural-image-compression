"""Perform inference/compression on a pre-trained mean-scale hyperprior model modified for lossy bits-back.
Implement BB without SGA (no initial iterative inference over the latents; A5 in Table 1 of paper), in
Yibo Yang, Robert Bamler, Stephan Mandt:
"Improving Inference for Neural Image Compression", NeurIPS 2020
https://arxiv.org/pdf/2006.04240.pdf
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

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

likelihood_lowerbound = 1e-9
variance_upperbound = 2e1


def compress(args):
    """Compresses an image, or a batch of images of the same shape in npy format."""
    from configs import get_eval_batch_size

    if args.input_file.endswith('.npy'):
        # .npy file should contain N images of the same shapes, in the form of an array of shape [N, H, W, 3]
        X = np.load(args.input_file)
    else:
        # Load input image and add batch dimension.
        from PIL import Image
        x = np.asarray(Image.open(args.input_file).convert('RGB'))
        X = x[None, ...]

    num_images = int(X.shape[0])
    img_num_pixels = int(np.prod(X.shape[1:-1]))
    X = X.astype('float32')
    X /= 255.

    eval_batch_size = get_eval_batch_size(img_num_pixels)
    dataset = tf.data.Dataset.from_tensor_slices(X)
    dataset = dataset.batch(batch_size=eval_batch_size)
    # https://www.tensorflow.org/api_docs/python/tf/compat/v1/data/Iterator
    # Importantly, each sess.run(op) call will consume a new batch, where op is any operation that depends on
    # x. Therefore if multiple ops need to be evaluated on the same batch of data, they have to be grouped like
    # sess.run([op1, op2, ...]).
    # x = dataset.make_one_shot_iterator().get_next()
    x_next = dataset.make_one_shot_iterator().get_next()

    x_ph = x = tf.placeholder('float32', (None, *X.shape[1:]))  # keep a reference around for feed_dict

    #### BEGIN build compression graph ####
    from utils import log_normal_pdf
    from learned_prior import BMSHJ2018Prior
    hyper_prior = BMSHJ2018Prior(args.num_filters, dims=(3, 3, 3))

    # Instantiate model.
    analysis_transform = AnalysisTransform(args.num_filters)
    synthesis_transform = SynthesisTransform(args.num_filters)
    hyper_analysis_transform = HyperAnalysisTransform(args.num_filters, num_output_filters=2 * args.num_filters)
    hyper_synthesis_transform = HyperSynthesisTransform(args.num_filters, num_output_filters=2 * args.num_filters)
    # entropy_bottleneck = tfc.EntropyBottleneck()

    # Build autoencoder and hyperprior.
    y = analysis_transform(x)
    y_tilde = tf.round(y)

    x_tilde = synthesis_transform(y_tilde)
    x_shape = tf.shape(x)
    x_tilde = x_tilde[:, :x_shape[1], :x_shape[2], :]  # crop reconstruction to have the same shape as input

    # z_tilde ~ q(z_tilde | h_a(\tilde y))
    z_mean_init, z_logvar_init = tf.split(hyper_analysis_transform(y_tilde), num_or_size_splits=2, axis=-1)
    z_mean = tf.placeholder('float32', z_mean_init.shape)  # initialize to inference network results
    z_logvar = tf.placeholder('float32', z_logvar_init.shape)

    eps = tf.random.normal(shape=tf.shape(z_mean))
    z_tilde = eps * tf.exp(z_logvar * .5) + z_mean

    log_q_z_tilde = log_normal_pdf(z_tilde, z_mean, z_logvar)  # bits back

    # compute the pdf of z_tilde under the flexible (hyper)prior p(z_tilde) ("z_likelihoods")
    z_likelihoods = hyper_prior.pdf(z_tilde, stop_gradient=False)
    z_likelihoods = math_ops.lower_bound(z_likelihoods, likelihood_lowerbound)

    # compute parameters of p(y_tilde|z_tilde)
    mu, sigma = tf.split(hyper_synthesis_transform(z_tilde), num_or_size_splits=2, axis=-1)
    sigma = tf.exp(sigma)  # make positive

    # need to handle images with non-standard sizes during compression; mu/sigma must have the same shape as y
    y_shape = tf.shape(y)
    mu = mu[:, :y_shape[1], :y_shape[2], :]
    sigma = sigma[:, :y_shape[1], :y_shape[2], :]
    scale_table = np.exp(np.linspace(np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
    conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table, mean=mu)
    # compute the pdf of y_tilde under the conditional prior/entropy model p(y_tilde|z_tilde)
    # = N(y_tilde|mu, sigma^2) conv U(-0.5, 0.5)
    y_likelihoods = conditional_bottleneck._likelihood(y_tilde)  # p(\tilde y | \tilde z)
    if conditional_bottleneck.likelihood_bound > 0:
        likelihood_bound = conditional_bottleneck.likelihood_bound
        y_likelihoods = math_ops.lower_bound(y_likelihoods, likelihood_bound)
    #### END build compression graph ####

    # Total number of bits divided by number of pixels.
    # - log p(\tilde y | \tilde z) - log p(\tilde z) - - log q(\tilde z | \tilde y)
    axes_except_batch = list(range(1, len(x.shape)))  # should be [1,2,3]
    bpp_back = tf.reduce_sum(-log_q_z_tilde, axis=axes_except_batch) / (np.log(2) * img_num_pixels)
    y_bpp = tf.reduce_sum(-tf.log(y_likelihoods), axis=axes_except_batch) / (np.log(2) * img_num_pixels)
    z_bpp = tf.reduce_sum(-tf.log(z_likelihoods), axis=axes_except_batch) / (np.log(2) * img_num_pixels)
    eval_bpp = y_bpp + z_bpp - bpp_back  # shape (N,)
    train_bpp = tf.reduce_mean(eval_bpp)

    local_gradients = tf.gradients(train_bpp, [z_mean, z_logvar])

    # Bring both images back to 0..255 range.
    x *= 255
    x_tilde = tf.clip_by_value(x_tilde, 0, 1)
    x_tilde = tf.round(x_tilde * 255)

    mse = tf.reduce_mean(tf.squared_difference(x, x_tilde), axis=axes_except_batch)  # shape (N,)
    psnr = tf.image.psnr(x_tilde, x, 255)  # shape (N,)
    msssim = tf.image.ssim_multiscale(x_tilde, x, 255)  # shape (N,)
    msssim_db = -10 * tf.log(1 - msssim) / np.log(10)  # shape (N,)

    with tf.Session() as sess:
        # Load the latest model checkpoint, get compression stats
        save_dir = os.path.join(args.checkpoint_dir, args.runname)
        latest = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
        tf.train.Saver().restore(sess, save_path=latest)
        eval_fields = ['mse', 'psnr', 'msssim', 'msssim_db', 'est_bpp', 'est_y_bpp', 'est_z_bpp', 'est_bpp_back']
        eval_tensors = [mse, psnr, msssim, msssim_db, eval_bpp, y_bpp, z_bpp, bpp_back]
        all_results_arrs = {key: [] for key in eval_fields}  # append across all batches

        batch_idx = 0
        while True:
            try:
                x_val = sess.run(x_next)
                x_feed_dict = {x_ph: x_val}
                z_mean_cur, z_logvar_cur = sess.run([z_mean_init, z_logvar_init], feed_dict=x_feed_dict)  # np arrays

                opt_obj_hist = []
                opt_grad_hist = []
                lr = 0.005
                local_its = 1000
                from adam import Adam
                np_adam_optimizer = Adam(lr=lr)
                for it in range(local_its):
                    grads, obj = sess.run([local_gradients, train_bpp],
                                          feed_dict={z_mean: z_mean_cur, z_logvar: z_logvar_cur, **x_feed_dict})
                    z_mean_cur, z_logvar_cur = np_adam_optimizer.update([z_mean_cur, z_logvar_cur], grads)
                    if it % 100 == 0:
                        print('negative local ELBO', obj)
                    opt_obj_hist.append(obj)
                    opt_grad_hist.append(np.mean(np.abs(grads)))
                print()

                # If requested, transform the quantized image back and measure performance.
                eval_arrs = sess.run(eval_tensors, feed_dict={z_mean: z_mean_cur,
                                                              z_logvar: z_logvar_cur,
                                                              **x_feed_dict})
                for field, arr in zip(eval_fields, eval_arrs):
                    all_results_arrs[field] += arr.tolist()

                batch_idx += 1

            except tf.errors.OutOfRangeError:
                break

        for field in eval_fields:
            all_results_arrs[field] = np.asarray(all_results_arrs[field])

        input_file = os.path.basename(args.input_file)
        results_dict = all_results_arrs
        trained_script_name = args.runname.split('-')[0]
        script_name = os.path.splitext(os.path.basename(__file__))[0]  # current script name, without extension
        save_file = 'rd-%s-file=%s.npz' % (args.runname, input_file)
        if script_name != trained_script_name:
            save_file = 'rd-%s+%s-file=%s.npz' % (
                script_name, args.runname, input_file)
        np.savez(os.path.join(args.results_dir, save_file), **results_dict)

        for field in eval_fields:
            arr = all_results_arrs[field]
            print('Avg {}: {:0.4f}'.format(field, arr.mean()))


from tf_boilerplate import parse_args


def main(args):
    # Invoke subcommand.
    assert args.command == "compress", 'Only compression is supported.'
    compress(args)


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
