"""Perform inference/compression on a pre-trained mean-scale hyperprior model modified for lossy bits-back.
Implement SGA + BB (M2 in Table 1 of paper), in
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
Z_DENSITY = 1 << 4  # Should probably be a power of 2 to avoid rounding errors.

import tensorflow_compression as tfc
from nn_models import AnalysisTransform, SynthesisTransform, HyperAnalysisTransform
from nn_models import MBT2018HyperSynthesisTransform as HyperSynthesisTransform

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

likelihood_lowerbound = 1e-9
variance_upperbound = 2e1


class Model:
    def __init__(self, args, X):
        """Constructs the computational graph for the entire model in a reproducible way."""
        from configs import get_eval_batch_size

        self.img_num_pixels = int(np.prod(X.shape[1:-1]))

        eval_batch_size = get_eval_batch_size(self.img_num_pixels)
        dataset = tf.data.Dataset.from_tensor_slices(X)
        dataset = dataset.batch(batch_size=eval_batch_size)
        # https://www.tensorflow.org/api_docs/python/tf/compat/v1/data/Iterator
        # Importantly, each sess.run(op) call will consume a new batch, where op is any operation that depends on
        # x. Therefore if multiple ops need to be evaluated on the same batch of data, they have to be grouped like
        # sess.run([op1, op2, ...]).
        # x = dataset.make_one_shot_iterator().get_next()
        self.x_next = dataset.make_one_shot_iterator().get_next()

        self.x_ph = x = tf.placeholder('float32', (None, *X.shape[1:]), name='x_ph')  # keep a reference around for feed_dict

        #### BEGIN build compression graph ####
        from utils import log_normal_pdf

        from learned_prior import BMSHJ2018Prior
        self.hyper_prior = BMSHJ2018Prior(args.num_filters, dims=(3, 3, 3))

        # Instantiate model.
        analysis_transform = AnalysisTransform(args.num_filters)
        synthesis_transform = SynthesisTransform(args.num_filters)
        hyper_analysis_transform = HyperAnalysisTransform(args.num_filters, num_output_filters=2 * args.num_filters)
        hyper_synthesis_transform = HyperSynthesisTransform(args.num_filters, num_output_filters=2 * args.num_filters)
        # entropy_bottleneck = tfc.EntropyBottleneck()

        # Initial optimization (where we still have access to x)
        # Soft-to-hard rounding with Gumbel-softmax trick; for each element of z_tilde, let R be a 2D auxiliary one-hot
        # random vector, such that R=[1, 0] means rounding DOWN and [0, 1] means rounding UP.
        # Let the logits of each outcome be -(z - z_floor) / T and -(z_ceil - z) / T (i.e., Boltzmann distribution with
        # energies (z - floor(z)) and (ceil(z) - z), so p(R==[1,0]) = softmax((z - z_floor) / T), ...
        # Let z_tilde = p(R==[1,0]) * floor(z) + p(R==[0,1]) * ceil(z), so z_tilde -> round(z) as T -> 0.
        import tensorflow_probability as tfp
        self.T = tf.placeholder('float32', shape=[], name='temperature')
        self.y_init = analysis_transform(x)
        self.y = tf.placeholder('float32', self.y_init.shape, name='y')
        y_floor = tf.floor(self.y)
        y_ceil = tf.ceil(self.y)
        y_bds = tf.stack([y_floor, y_ceil], axis=-1)
        epsilon = 1e-5
        logits = tf.stack([-tf.math.atanh(tf.clip_by_value(self.y - y_floor, -1 + epsilon, 1 - epsilon)) / self.T,
                        -tf.math.atanh(tf.clip_by_value(y_ceil - self.y, -1 + epsilon, 1 - epsilon)) / self.T],
                        axis=-1)  # last dim are logits for DOWN or UP; clip to prevent NaN as temperature -> 0
        rounding_dist = tfp.distributions.RelaxedOneHotCategorical(self.T,
                                                                logits=logits)  # technically we can use a different temperature here
        sample_concrete = rounding_dist.sample()
        self.y_tilde = tf.reduce_sum(y_bds * sample_concrete, axis=-1)  # inner product in last dim
        x_tilde = synthesis_transform(self.y_tilde)
        x_shape = tf.shape(x)
        x_tilde = x_tilde[:, :x_shape[1], :x_shape[2], :]  # crop reconstruction to have the same shape as input

        # z_tilde ~ q(z_tilde | h_a(\tilde y))
        self.z_mean_init, self.z_logvar_init = tf.split(hyper_analysis_transform(self.y_tilde), num_or_size_splits=2, axis=-1)
        self.z_mean = tf.placeholder('float32', self.z_mean_init.shape, name='z_mean')  # initialize to inference network results
        self.z_logvar = tf.placeholder('float32', self.z_logvar_init.shape, name='z_logvar')

        eps = tf.random.normal(shape=tf.shape(self.z_mean))
        self.z_tilde = eps * tf.exp(self.z_logvar * .5) + self.z_mean

        log_q_z_tilde = log_normal_pdf(self.z_tilde, self.z_mean, self.z_logvar)  # bits back

        # compute the pdf of z_tilde under the flexible (hyper)prior p(z_tilde) ("z_likelihoods")
        z_likelihoods = self.hyper_prior.pdf(self.z_tilde, stop_gradient=False)
        z_likelihoods = math_ops.lower_bound(z_likelihoods, likelihood_lowerbound)

        # compute parameters of p(y_tilde|z_tilde)
        self.mu, self.sigma = tf.split(hyper_synthesis_transform(self.z_tilde), num_or_size_splits=2, axis=-1)
        self.sigma = tf.exp(self.sigma)  # make positive

        # need to handle images with non-standard sizes during compression; mu/sigma must have the same shape as y
        y_shape = tf.shape(self.y_tilde)
        self.mu = self.mu[:, :y_shape[1], :y_shape[2], :]
        self.sigma = self.sigma[:, :y_shape[1], :y_shape[2], :]
        scale_table = np.exp(np.linspace(np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
        conditional_bottleneck = tfc.GaussianConditional(self.sigma, scale_table, mean=self.mu)
        # compute the pdf of y_tilde under the conditional prior/entropy model p(y_tilde|z_tilde)
        # = N(y_tilde|mu, sigma^2) conv U(-0.5, 0.5)
        y_likelihoods = conditional_bottleneck._likelihood(self.y_tilde)  # p(\tilde y | \tilde z)
        if conditional_bottleneck.likelihood_bound > 0:
            likelihood_bound = conditional_bottleneck.likelihood_bound
            y_likelihoods = math_ops.lower_bound(y_likelihoods, likelihood_bound)
        #### END build compression graph ####

        # Total number of bits divided by number of pixels.
        # - log p(\tilde y | \tilde z) - log p(\tilde z) - - log q(\tilde z | \tilde y)
        axes_except_batch = list(range(1, len(x.shape)))  # should be [1,2,3]
        batch_log_q_z_tilde = tf.reduce_sum(log_q_z_tilde, axis=axes_except_batch)
        self.bpp_back = -batch_log_q_z_tilde / (np.log(2) * self.img_num_pixels)
        batch_log_cond_p_y_tilde = tf.reduce_sum(tf.log(y_likelihoods), axis=axes_except_batch)
        self.y_bpp = -batch_log_cond_p_y_tilde / (np.log(2) * self.img_num_pixels)
        batch_log_p_z_tilde = tf.reduce_sum(tf.log(z_likelihoods), axis=axes_except_batch)
        self.z_bpp = -batch_log_p_z_tilde / (np.log(2) * self.img_num_pixels)
        self.eval_bpp = self.y_bpp + self.z_bpp - self.bpp_back  # shape (N,)
        self.train_bpp = tf.reduce_mean(self.eval_bpp)
        self.net_z_bpp = tf.reduce_mean(self.z_bpp - self.bpp_back)

        z_scale_extra_bits = np.log2(Z_DENSITY) * int(np.prod(z_likelihoods.shape.as_list()[1:]))
        self.bits_back = z_scale_extra_bits - batch_log_q_z_tilde / np.log(2)
        self.y_bits = -batch_log_cond_p_y_tilde / np.log(2)
        self.z_bits = z_scale_extra_bits - batch_log_p_z_tilde / np.log(2)
        self.eval_bits = self.y_bits + self.z_bits - self.bits_back # shape (N,)

        # Mean squared error across pixels.
        self.train_mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))
        # Multiply by 255^2 to correct for rescaling.
        # float_train_mse = train_mse
        # psnr = - 10 * (tf.log(float_train_mse) / np.log(10))  # float MSE computed on float images
        self.train_mse *= 255 ** 2

        try:
            lmbda = args.lmbda
        except:
            # For `encode_latents` and `decode_latents` subcommands
            lmbda = -1

        # The rate-distortion cost.
        if lmbda < 0:
            lmbda = float(args.runname.split('lmbda=')[1].split('-')[0])  # re-use the lmbda as used for training
            print('Defaulting lmbda (mse coefficient) to %g as used in model training.' % lmbda)
        if lmbda > 0:
            self.rd_loss = lmbda * self.train_mse + self.train_bpp
        else:
            self.rd_loss = self.train_bpp
        self.rd_gradients = tf.gradients(self.rd_loss, [self.y, self.z_mean, self.z_logvar])
        self.r_gradients = tf.gradients(self.net_z_bpp, [self.z_mean, self.z_logvar])

        x *= 255
        x_tilde = tf.clip_by_value(x_tilde, 0, 1)
        x_tilde = tf.round(x_tilde * 255)

        self.mse = tf.reduce_mean(tf.squared_difference(x, x_tilde), axis=axes_except_batch)  # shape (N,)
        self.psnr = tf.image.psnr(x_tilde, x, 255)  # shape (N,)
        self.msssim = tf.image.ssim_multiscale(x_tilde, x, 255)  # shape (N,)
        self.msssim_db = -10 * tf.log(1 - self.msssim) / np.log(10)  # shape (N,)

def compress(args):
    """Compresses an image, or a batch of images of the same shape in npy format."""

    if args.input_file.endswith('.npy'):
        # .npy file should contain N images of the same shapes, in the form of an array of shape [N, H, W, 3]
        X = np.load(args.input_file)
    else:
        # Load input image and add batch dimension.
        from PIL import Image
        x = np.asarray(Image.open(args.input_file).convert('RGB'))
        X = x[None, ...]

    X = X.astype('float32')
    X /= 255.

    model = Model(args, X)
    # Bring both images back to 0..255 range, for evaluation only.

    with tf.Session() as sess:
        # Load the latest model checkpoint, get compression stats
        save_dir = os.path.join(args.checkpoint_dir, args.runname)
        latest = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
        tf.train.Saver().restore(sess, save_path=latest)
        eval_fields = ['mse', 'psnr', 'msssim', 'msssim_db', 'est_bpp', 'est_y_bpp', 'est_z_bpp', 'est_bpp_back']
        eval_tensors = [model.mse, model.psnr, model.msssim, model.msssim_db, model.eval_bpp, model.y_bpp, model.z_bpp, model.bpp_back]
        all_results_arrs = {key: [] for key in eval_fields}  # append across all batches

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        log_itv = 100
        rd_lr = 0.005
        # rd_opt_its = args.sga_its
        rd_opt_its = 2000
        annealing_scheme = 'exp0'
        annealing_rate = args.annealing_rate  # default annealing_rate = 1e-3
        t0 = args.t0  # default t0 = 700
        T_ub = 0.5  # max/initial temperature
        from utils import annealed_temperature
        r_lr = 0.003
        r_opt_its = 2000
        from adam import Adam

        batch_idx = 0
        while True:
            try:
                x_val = sess.run(model.x_next)
                x_feed_dict = {model.x_ph: x_val}
                # 1. Perform R-D optimization conditioned on ground truth x
                print('----RD Optimization----')
                y_cur = sess.run(model.y_init, feed_dict=x_feed_dict)  # np arrays
                z_mean_cur, z_logvar_cur = sess.run([model.z_mean_init, model.z_logvar_init], feed_dict={model.y_tilde: y_cur})
                rd_loss_hist = []
                adam_optimizer = Adam(lr=rd_lr)

                opt_record = {'its': [], 'T': [], 'rd_loss': [], 'rd_loss_after_rounding': []}
                for it in range(rd_opt_its):
                    temperature = annealed_temperature(it, r=annealing_rate, ub=T_ub, scheme=annealing_scheme, t0=t0)
                    grads, obj, mse_, train_bpp_, psnr_ = sess.run(
                        [model.rd_gradients, model.rd_loss, model.train_mse, model.train_bpp, model.psnr],
                        feed_dict={model.y: y_cur, model.z_mean: z_mean_cur, model.z_logvar: z_logvar_cur, **x_feed_dict, model.T: temperature})
                    y_cur, z_mean_cur, z_logvar_cur = adam_optimizer.update([y_cur, z_mean_cur, z_logvar_cur], grads)
                    if it % log_itv == 0 or it + 1 == rd_opt_its:
                        psnr_ = psnr_.mean()
                        if args.verbose:
                            bpp_after_rounding, psnr_after_rounding, rd_loss_after_rounding = sess.run(
                                [model.train_bpp, model.psnr, model.rd_loss],
                                feed_dict={
                                    model.y_tilde: np.round(y_cur),
                                    model.z_mean: z_mean_cur,
                                    model.z_logvar: z_logvar_cur,
                                    **x_feed_dict})
                            psnr_after_rounding = psnr_after_rounding.mean()
                            print(
                                'it=%d, T=%.3f rd_loss=%.4f mse=%.3f bpp=%.4f psnr=%.4f\t after rounding: rd_loss=%.4f, bpp=%.4f psnr=%.4f'
                                % (
                                    it, temperature, obj, mse_, train_bpp_, psnr_, rd_loss_after_rounding,
                                    bpp_after_rounding,
                                    psnr_after_rounding))
                        else:
                            print('it=%d, T=%.3f rd_loss=%.4f mse=%.3f bpp=%.4f psnr=%.4f' % (
                                it, temperature, obj, mse_, train_bpp_, psnr_))
                    rd_loss_hist.append(obj)
                print()

                # 2. Fix y_tilde, perform rate optimization w.r.t. z_mean and z_logvar.
                y_tilde_cur = np.round(y_cur)  # this is the latents we end up transmitting
                # rate_feed_dict = {y_tilde: y_tilde_cur, **x_feed_dict}
                rate_feed_dict = {model.y_tilde: y_tilde_cur}
                np.random.seed(seed)
                tf.set_random_seed(seed)
                print('----Rate Optimization----')
                # Reinitialize based on the value of y_tilde
                z_mean_cur, z_logvar_cur = sess.run([model.z_mean_init, model.z_logvar_init], feed_dict=rate_feed_dict)  # np arrays

                r_loss_hist = []
                # rate_grad_hist = []

                adam_optimizer = Adam(lr=r_lr)
                for it in range(r_opt_its):
                    grads, obj = sess.run([model.r_gradients, model.net_z_bpp],
                                          feed_dict={model.z_mean: z_mean_cur, model.z_logvar: z_logvar_cur, **rate_feed_dict})
                    z_mean_cur, z_logvar_cur = adam_optimizer.update([z_mean_cur, z_logvar_cur], grads)
                    if it % log_itv == 0 or it + 1 == r_opt_its:
                        print('it=', it, '\trate=', obj)
                    r_loss_hist.append(obj)
                    # rate_grad_hist.append(np.mean(np.abs(grads)))
                print()

                # fig, axes = plt.subplots(nrows=2, sharex=True)
                # axes[0].plot(rd_loss_hist)
                # axes[0].set_ylabel('RD loss')
                # axes[1].plot(r_loss_hist)
                # axes[1].set_ylabel('Rate loss')
                # axes[1].set_xlabel('SGD iterations')
                # plt.savefig('plots/local_q_opt_hist-%s-input=%s-b=%d.png' %
                #             (args.runname, os.path.basename(args.input_file), batch_idx))

                # If requested, transform the quantized image back and measure performance.
                eval_arrs = sess.run(
                    eval_tensors,
                    feed_dict={model.y_tilde: y_tilde_cur, model.z_mean: z_mean_cur,
                               model.z_logvar: z_logvar_cur, **x_feed_dict})
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
        save_file = 'rd-%s-input=%s.npz' % (args.runname, input_file)
        if script_name != trained_script_name:
            save_file = 'rd-%s-lmbda=%g+%s-input=%s.npz' % (
                script_name, args.lmbda, args.runname, input_file)
        np.savez(os.path.join(args.results_dir, save_file), **results_dict)

        if args.save_latents:
            prefix = 'latents'
            save_file = '%s-%s-input=%s.npz' % (prefix, args.runname, input_file)
            if script_name != trained_script_name:
                save_file = '%s-%s-lmbda=%g+%s-input=%s.npz' % (
                    prefix, script_name, args.lmbda, args.runname, input_file)
            np.savez(
                os.path.join(args.results_dir, save_file),
                y_tilde_cur = y_tilde_cur.astype(np.int32),
                z_mean_cur = z_mean_cur,
                z_logvar_cur = z_logvar_cur,
                img_dimensions = np.array(X.shape[1:-1], dtype=np.int32))

        for field in eval_fields:
            arr = all_results_arrs[field]
            print('Avg {}: {:0.4f}'.format(field, arr.mean()))


def encode_latents(args):
    """Entropy code latent variables previously generated with `compress --save_latents` and write to a file."""

    # Encoding and decoding has to run on CPU because all calculations have to be exactly
    # reproducible. GPU calculations turn out to differ by tiny amounts between encoder and
    # decoder executions (probably due to different orderings of ops). While the discrepancies
    # appear only in the least significant bits, they are large enough to completely confuse
    # the entropy coder.
    assert not tf.test.is_gpu_available(), 'Encoding and decoding currently doesn\'t support GPU acceleration. Run with "CUDA_VISIBLE_DEVICES=-1".'

    import ans
    import sys
    from learned_prior import BMSHJ2018Prior
    from utils import uint_to_bytes, bytes_to_uint
    from configs import get_eval_batch_size

    # Load the latents
    latents_file = np.load(args.input_file)
    y = latents_file['y_tilde_cur']
    z_mean = Z_DENSITY * latents_file['z_mean_cur']
    z_std = Z_DENSITY * np.exp(0.5 * latents_file['z_logvar_cur'])
    batch_size = y.shape[0]
    width, height = latents_file['img_dimensions']
    del latents_file

    # Generate random side information and decode into z using q(z|y)
    rng = np.random.RandomState(1234)
    num_z = len(z_mean[0].ravel()) if args.separate else len(z_mean.ravel())
    # 32 bits per parameter should do; if not, just increase the size.
    side_information = rng.randint(0, 1 << 32, dtype=np.uint32, size=(10 + num_z,))
    # Highest bit must be set (and therefore does not count towards the size of the side information.)
    side_information[-1] |= 1 << 31
    side_information_bits = 32 * len(side_information) - 1

    # Find maximum range for grid in z-space:
    encoder_ranges_z = np.array([15, 31, 63, 127])
    max_z_abs = (np.abs(z_mean) + 3.0 * z_std + 0.5).max()
    z_grid_range_index = min(len(encoder_ranges_z) -1, np.sum(Z_DENSITY * encoder_ranges_z < max_z_abs))
    z_grid_range = Z_DENSITY * encoder_ranges_z[z_grid_range_index]

    # Instantiate the model.
    fake_X = np.zeros((batch_size, width, height, 3), dtype=np.float32)
    model = Model(args, fake_X)
    _, z_width, z_height, _ = model.z_mean.shape.as_list()

    # Rasterize the hyperprior.
    z_grid = tf.tile(
        tf.reshape(tf.range(-z_grid_range, z_grid_range + 1), (-1, 1, 1, 1)),
        (1, 1, 1, args.num_filters))
    z_grid = (1.0 / Z_DENSITY) * tf.cast(z_grid, tf.float32)
    z_grid_likelihood = tf.reshape(model.hyper_prior.pdf(z_grid), (2 * z_grid_range + 1, args.num_filters))
    # (`z_grid_likelihood` is normalized to Z_DENSITY rather than one but that shouldn't make a
    # difference since the coder doesn't require it to be normalized.)

    def encode_tensors(y, output_file, byteorder, Coder, sess):
        # Rerun final optimization in z-space.
        # (We can't use the result from the `compress` subcommand because compression
        # is better done on a GPU but, unfortunately GPU calculations turn out to be
        # not exactly reproducible. So we rerun the z-space optimization on CPU here.)

        eval_batch_size = get_eval_batch_size(model.img_num_pixels)
        dataset = tf.data.Dataset.from_tensor_slices(y)
        dataset = dataset.batch(batch_size=eval_batch_size)
        y_next = dataset.make_one_shot_iterator().get_next()
        y_val = sess.run(y_next)

        y_tilde_cur = y_val.astype(np.float32)
        rate_feed_dict = {model.y_tilde: y_tilde_cur}
        np.random.seed(seed)
        tf.set_random_seed(seed)
        print('----Rate Optimization----')
        z_mean_cur, z_logvar_cur = sess.run([model.z_mean_init, model.z_logvar_init], feed_dict={model.y_tilde: y_tilde_cur})

        r_lr = 0.003
        r_opt_its = 2000
        from adam import Adam
        adam_optimizer = Adam(lr=r_lr)
        for it in range(r_opt_its):
            grads, obj = sess.run([model.r_gradients, model.net_z_bpp],
                                    feed_dict={model.z_mean: z_mean_cur, model.z_logvar: z_logvar_cur, **rate_feed_dict})
            z_mean_cur, z_logvar_cur = adam_optimizer.update([z_mean_cur, z_logvar_cur], grads)
            if it % 100 == 0 or it + 1 == r_opt_its:
                print('it=', it, '\trate=', obj)
        print()

        est_bits_back, est_y_bits, est_z_bits, est_net_bits = sess.run(
            [model.bits_back, model.y_bits, model.z_bits, model.eval_bits],
            feed_dict={model.z_mean: z_mean_cur, model.z_logvar: z_logvar_cur, **rate_feed_dict})

        # Find range for entropy coding:
        z_mean_scaled = Z_DENSITY * z_mean_cur
        z_std_scaled = Z_DENSITY * np.exp(0.5 * z_logvar_cur)
        max_z_abs = (np.abs(z_mean_scaled) + 3.0 * z_std_scaled + 0.5).max()
        encoder_z_range_index = min(z_grid_range_index, np.sum(Z_DENSITY * encoder_ranges_z < max_z_abs))
        encoder_z_range = Z_DENSITY * encoder_ranges_z[encoder_z_range_index]

        encoder_ranges_y = np.array([31, 63, 127, 255])
        max_y_abs = np.abs(y).max()
        encoder_y_range_index = np.sum(encoder_ranges_y < max_y_abs)
        if encoder_y_range_index == len(encoder_ranges_y):
            raise "Values for y out of range."
        encoder_y_range = encoder_ranges_y[encoder_y_range_index]

        coder = Coder(side_information)
        batch_size, _, _, _ = z_mean_scaled.shape
        z = np.empty(z_mean_scaled.shape, dtype=np.int32)
        coder.pop_gaussian_symbols(
            -encoder_z_range, encoder_z_range,
            z_mean_scaled.ravel().astype(np.float64),
            z_std_scaled.ravel().astype(np.float64),
            z.ravel(),
            True)
        num_bits_after_popping_z = coder.num_bits()

        fake_y = np.zeros([batch_size] + model.y.shape[1:].as_list(), dtype=np.float32)
        y_mean, y_std = sess.run(
            [model.mu, model.sigma], 
            {
                model.z_tilde: (1.0 / Z_DENSITY) * z.astype(np.float32),
                model.y: fake_y,
                model.T: 1.0
            })

        # Entropy coder cannot deal with infinite or zero standard deviations.
        y_std = np.maximum(np.minimum(y_std, 16 * encoder_y_range), 1e-6)

        coder.push_gaussian_symbols(
            y.ravel().astype(np.int32),
            -encoder_y_range, encoder_y_range,
            y_mean.ravel().astype(np.float64),
            y_std.ravel().astype(np.float64),
            True)
        num_bits_after_pushing_y = coder.num_bits()

        z_grid_cutoff_left = z_grid_range - encoder_z_range
        z_grid_cutoff_right = 2 * z_grid_range + 1 - z_grid_cutoff_left
        _, num_filters = z_grid_likelihood.shape
        for i in range(num_filters):
            coder.push_iid_categorical_symbols(
                z[..., i].flatten().astype(np.int32),
                -encoder_z_range, encoder_z_range, -encoder_z_range,
                z_grid_likelihood[z_grid_cutoff_left:z_grid_cutoff_right, i].flatten().astype(np.float64))
        num_bits_after_pushing_z = coder.num_bits()

        compressed = np.empty((coder.num_words(),), dtype=np.uint32)
        coder.copy_compressed(compressed)
        if byteorder == 'big':
            compressed.byteswap()

        with open(output_file, 'wb') as f:
            batch_size, z_width, z_height, _ = z.shape
            assert batch_size != 0 and z_width != 0 and z_height != 0

            batch_size = uint_to_bytes(batch_size)
            x_width = uint_to_bytes(width)
            x_height = uint_to_bytes(height)
            if len(x_width) > len(x_height):
                x_height = bytearray([0] * (len(x_width) - len(x_height)) + list(x_height))
            elif len(x_height) > len(x_width):
                x_width = bytearray([0] * (len(x_height) - len(x_width)) + list(x_width))
            
            # first byte: 2 bits each for length of batch size length, length of dimensions, encoder_z_range_index, encoder_y_range_index
            first_byte = ((len(batch_size) - 1) << 6) | ((len(x_width) - 1) << 4) | (encoder_z_range_index << 2) | encoder_y_range_index
            header = bytearray([first_byte] + list(batch_size) + list(x_width) + list(x_height))

            f.write(header)
            compressed.tofile(f)

        print('Compressed data written to file %s' % output_file)
        print()
        print('Expected bit rates based on information content:')
        print('- expected bits back: %.2f bits' % np.sum(est_bits_back))
        print('- expected bits for encoding y|z: %.2f bits' % np.sum(est_y_bits))
        print('- expected bits for encoding z: %.2f bits' % np.sum(est_z_bits))
        print('- expected net file size: %.2f bits' % np.sum(est_net_bits))
        print()
        print('Actual bit rates from entropy coder:')
        print('- started with %d bits of random side information' % side_information_bits)
        print('- actual bits back: %d bits' % (side_information_bits - num_bits_after_popping_z))
        print('- actual bits for encoding y|z: %d bits' % (num_bits_after_pushing_y - num_bits_after_popping_z))
        print('- actual bits for encoding z: %d bits' % (num_bits_after_pushing_z - num_bits_after_pushing_y))
        print('- header size: %d bits' % (8 * len(header)))
        print(
            '- actual net file size (including header): %d bits'
            % (8 * len(header) + num_bits_after_pushing_z - side_information_bits))

    with tf.Session() as sess:
        # Load the latest model checkpoint.
        save_dir = os.path.join(args.checkpoint_dir, args.runname)
        latest = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
        tf.train.Saver().restore(sess, save_path=latest)

        # Replace tensorflow op with its corresponding value.
        z_grid_likelihood = sess.run(z_grid_likelihood)

        output_file = args.output_file
        if output_file is None:
            output_file = args.input_file + '.compressed'

        if args.separate:
            batch_size, _, _, _ = z_mean.shape
            for i in range(batch_size):
                # (z_mean, z_std, y, mu, sigma, output_file, z_grid_range, byteorder, Coder, sess):
                encode_tensors(
                    y[i:i+1, ...],
                    '%s.%d' % (output_file, i), sys.byteorder, ans.Coder, sess)
        else:
            encode_tensors(y, output_file, sys.byteorder, ans.Coder, sess)


def decode_latents(args):
    """Decode compressed latents generated with `encode_latents`, write them to a file, and verify side information."""

    # Encoding and decoding has to run on CPU because all calculations have to be exactly
    # reproducible. GPU calculations turn out to differ by tiny amounts between encoder and
    # decoder executions (probably due to different orderings of ops). While the discrepancies
    # appear only in the least significant bits, they are large enough to completely confuse
    # the entropy coder.
    assert not tf.test.is_gpu_available(), 'Encoding and decoding currently doesn\'t support GPU acceleration. Run with "CUDA_VISIBLE_DEVICES=-1".'

    import ans
    import sys
    from learned_prior import BMSHJ2018Prior
    from utils import uint_to_bytes, bytes_to_uint, log_normal_pdf
    from configs import get_eval_batch_size

    # Read file header and payload.
    with open(args.input_file, 'rb') as f:
        # first byte: 2 bits each for length of batch size length, length of dimensions, encoder_z_range_index, encoder_y_range_index
        first_byte = f.read(1)[0]
        batch_size_len = (first_byte >> 6) + 1
        dimensions_len = ((first_byte >> 4) & 3) + 1
        encoder_z_range_index = (first_byte >> 2) & 3
        encoder_y_range_index = first_byte & 3

        batch_size = bytes_to_uint(f.read(batch_size_len))
        x_width = bytes_to_uint(f.read(dimensions_len))
        x_height = bytes_to_uint(f.read(dimensions_len))
        # y_width = z_width * 4
        # y_height = z_height * 4

        compressed = np.fromfile(f, dtype=np.uint32)

    if sys.byteorder == 'big':
        compressed.byteswap()
    coder = ans.Coder(compressed)

    # Find range for entropy coding:
    encoder_z_ranges = np.array([15, 31, 63, 127])
    encoder_z_range = Z_DENSITY * encoder_z_ranges[encoder_z_range_index]
    encoder_y_ranges = np.array([31, 63, 127, 255])
    encoder_y_range = encoder_y_ranges[encoder_y_range_index]

    # Instantiate the model.
    fake_X = np.zeros((batch_size, x_width, x_height, 3), dtype=np.float32)
    model = Model(args, fake_X)
    _, z_width, z_height, _ = model.z_mean.shape.as_list()

    # Rasterize the hyperprior.
    z_grid = tf.tile(
        tf.reshape(tf.range(-encoder_z_range, encoder_z_range + 1), (-1, 1, 1, 1)),
        (1, 1, 1, args.num_filters))
    z_grid = (1.0 / Z_DENSITY) * tf.cast(z_grid, tf.float32)
    z_grid_likelihood = tf.reshape(model.hyper_prior.pdf(z_grid), (2 * encoder_z_range + 1, args.num_filters))
    # (`z_grid_likelihood` is normalized to Z_DENSITY rather than one but that shouldn't make a
    # difference since the coder doesn't require it to be normalized.)

    with tf.Session() as sess:
        # Load the latest model checkpoint.
        save_dir = os.path.join(args.checkpoint_dir, args.runname)
        latest = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
        tf.train.Saver().restore(sess, save_path=latest)

        # Replace tensorflow op with its corresponding value.
        z_grid_likelihood = sess.run(z_grid_likelihood)

        z = np.empty((args.num_filters, batch_size, z_width, z_height), dtype=np.int32)
        for i in reversed(range(args.num_filters)):
            coder.pop_iid_categorical_symbols(
                -encoder_z_range, encoder_z_range, -encoder_z_range,
                z_grid_likelihood[:, i].flatten().astype(np.float64),
                z[i, ...].ravel())
        z = z.transpose((1, 2, 3, 0))

        fake_y = np.zeros([batch_size] + model.y.shape[1:].as_list(), dtype=np.float32)
        y_mean, y_std = sess.run(
            [model.mu, model.sigma], 
            {
                model.z_tilde: (1.0 / Z_DENSITY) * z.astype(np.float32),
                model.y: fake_y,
                model.T: 1.0
            })

        # Entropy coder cannot deal with infinite or zero standard deviations.
        y_std = np.maximum(np.minimum(y_std, 16 * encoder_y_range), 1e-6)

        y = np.empty(y_mean.shape, dtype=np.int32)
        coder.pop_gaussian_symbols(
            -encoder_y_range, encoder_y_range,
            y_mean.ravel().astype(np.float64),
            y_std.ravel().astype(np.float64),
            y.ravel(),
            True)
        compressed = np.empty((coder.num_words(),), dtype=np.uint32)
        coder.copy_compressed(compressed)

        # Write reconstructed y.
        output_file = args.output_file
        if output_file is None:
            output_file = args.input_file + '.reconstructed.npz'    
        np.savez(output_file, y_tilde_cur=y, y_mean=y_mean, y_std=y_std)

        # Rerun final optimization in z-space
        eval_batch_size = get_eval_batch_size(model.img_num_pixels)
        dataset = tf.data.Dataset.from_tensor_slices(y)
        dataset = dataset.batch(batch_size=eval_batch_size)
        y_next = dataset.make_one_shot_iterator().get_next()

        while True:
            try:
                y_val = sess.run(y_next)

                y_tilde_cur = y_val.astype(np.float32)
                rate_feed_dict = {model.y_tilde: y_tilde_cur}
                np.random.seed(seed)
                tf.set_random_seed(seed)
                print('----Rate Optimization----')
                z_mean_cur, z_logvar_cur = sess.run([model.z_mean_init, model.z_logvar_init], feed_dict={model.y_tilde: y_tilde_cur})

                r_lr = 0.003
                r_opt_its = 2000
                from adam import Adam
                adam_optimizer = Adam(lr=r_lr)
                for it in range(r_opt_its):
                    grads, obj = sess.run([model.r_gradients, model.net_z_bpp],
                                            feed_dict={model.z_mean: z_mean_cur, model.z_logvar: z_logvar_cur, **rate_feed_dict})
                    z_mean_cur, z_logvar_cur = adam_optimizer.update([z_mean_cur, z_logvar_cur], grads)
                    if it % 100 == 0 or it + 1 == r_opt_its:
                        print('it=', it, '\trate=', obj)
                print()
            except tf.errors.OutOfRangeError:
                break

    # Get back side information.
    coder.push_gaussian_symbols(
        z.ravel(),
        -encoder_z_range, encoder_z_range,
        (Z_DENSITY * z_mean_cur).ravel().astype(np.float64),
        (Z_DENSITY * np.exp(0.5 * z_logvar_cur)).ravel().astype(np.float64),
        True)

    # Verify side information.
    rng = np.random.RandomState(1234)
    side_information_size = 10 + len(z.ravel())
    assert side_information_size == coder.num_words()
    side_information = rng.randint(0, 1 << 32, dtype=np.uint32, size=(side_information_size,))
    # Highest bit must be set.
    side_information[-1] |= 1 << 31
    remaining_compressed = np.empty((side_information_size,), dtype=np.uint32)
    coder.copy_compressed(remaining_compressed)
    assert np.all(side_information == remaining_compressed)
    print('Reconstructed tensors written to file %s' % output_file)
    print('Successfully reconstructed expected side information.')


from tf_boilerplate import parse_args


def main(args):
    # Invoke subcommand.
    if args.command == "compress":
        compress(args)
    elif args.command == "encode_latents":
        encode_latents(args)
    elif args.command == "decode_latents":
        decode_latents(args)
    else:
        raise 'Only compression, encoding, and is supported.'


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
