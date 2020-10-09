# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
try:
    from tensorflow_compression.python.ops import math_ops
except ModuleNotFoundError:  # in case tfc dependency not installed
    import math_ops


class BMSHJ2018Prior(tf.keras.Model):
    """
    Flexible prior model from appendix of "Variational image compression with a scale hyperprior"
    """

    def __init__(self, channels, dims=(3, 3, 3), init_scale=10., **kwargs):
        """

        :param channels: number of distinct prior distributions to use
        :param dims: iterable of ints, specifying the sizes of intermediate dimensions in the sequence of transforms that
            define the CDF.
        :param init_scale: A scaling factor determining the initial width of the
            probability densities. This should be chosen big enough so that the
            range of values of the layer inputs roughly falls within the interval
            [`-init_scale`, `init_scale`] at the beginning of training.
        :param kwargs:
        """
        super(BMSHJ2018Prior, self).__init__(**kwargs)
        self._channels = int(channels)  # one distinct distribution per channel
        self._init_scale = float(init_scale)
        self._dims = tuple(int(f) for f in dims)
        # self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)

        # Creates the variables for the network modeling the densities.
        dims = (1,) + self.dims + (1,)
        scale = self.init_scale ** (1 / (len(self.dims) + 1))

        # Create variables.
        self._matrices = []
        self._matrices_ = []
        self._biases = []
        self._factors = []
        self._factors_ = []
        for i in range(len(self.dims) + 1):
            init = np.log(np.expm1(1 / scale / dims[i + 1]))
            matrix = self.add_weight(
                "matrix_{}".format(i), dtype=self.dtype,
                shape=(channels, dims[i + 1], dims[i]),  # tf.matmul happens in the inner-most two dimensions
                initializer=tf.initializers.constant(init))
            self._matrices_.append(matrix)
            matrix = tf.nn.softplus(matrix)
            self._matrices.append(matrix)

            bias = self.add_weight(
                "bias_{}".format(i), dtype=self.dtype,
                shape=(channels, dims[i + 1], 1),
                initializer=tf.initializers.random_uniform(-.5, .5))
            self._biases.append(bias)

            if i < len(self.dims):
                factor = self.add_weight(
                    "factor_{}".format(i), dtype=self.dtype,
                    shape=(channels, dims[i + 1], 1),
                    initializer=tf.initializers.zeros())
                self._factors_.append(factor)
                factor = tf.math.tanh(factor)
                self._factors.append(factor)

    @property
    def init_scale(self):
        return self._init_scale

    @property
    def dims(self):
        # iterable of ints, specifying the sizes of intermediate dimensions in the sequence of transforms that define
        # the CDF.
        return self._dims

    def _logits_cdf(self, inputs, stop_gradient):
        """Evaluate logits of the cumulative densities.

        Arguments:
          inputs: The values at which to evaluate the cumulative densities, expected
            to be a `Tensor` of shape `(channels, 1, batch)` (this shape makes batch matrix-multiplication easier in
            tensorflow). For channel n of input, inputs[n], we evaluate the CDF logits using parameters of the nth
            channel distribution ([m[n] for m in self._matrices], [b[n] for b in self._biases], etc.) in parallel.
            The last (innermost) axis of inputs is simply broadcasted over to allow batch evaluation of multiple inputs.
          stop_gradient: Boolean. Whether to add `tf.stop_gradient` calls so
            that the gradient of the output with respect to the density model
            parameters is disconnected (the gradient with respect to `inputs` is
            left untouched).

        Returns:
          A `Tensor` of the same shape as `inputs`, containing the logits of the
          cumulative densities evaluated at the given inputs.
        """
        logits = inputs

        for i in range(len(self.dims) + 1):
            if tf.executing_eagerly():
                matrix = tf.nn.softplus(self._matrices_[i])
            else:
                matrix = self._matrices[i]
            if stop_gradient:
                matrix = tf.stop_gradient(matrix)
            logits = tf.linalg.matmul(matrix, logits)  # tf.matmul happens in the inner-most two dimensions

            bias = self._biases[i]
            if stop_gradient:
                bias = tf.stop_gradient(bias)
            logits += bias

            if i < len(self._factors):
                if tf.executing_eagerly():
                    factor = tf.math.tanh(self._factors_[i])
                else:
                    factor = self._factors[i]
                if stop_gradient:
                    factor = tf.stop_gradient(factor)
                logits += factor * tf.math.tanh(logits)

        return logits

    def cdf(self, inputs, stop_gradient):
        """
        Compute model CDF for a batch of inputs. The innermost axis of inputs will be aligned with the "channels"
        (independent distributions) modeled by the prior, so inputs[...,n] will be evaluated by the nth prior model
        distribution (dimensions within inputs[...,n] are treated as i.i.d., i.e., all coming from the nth prior model).
        :param inputs: Tensor, whose last dimension = number of channels in the prior model; this is typically
            [batch, z_height, z_width, channels] for a batch of latent image representations in convolutional VAEs.
        :param stop_gradient: Boolean. Whether to add `tf.stop_gradient` calls so
            that the gradient of the output with respect to the density model
            parameters is disconnected (the gradient with respect to `inputs` is
            left untouched).
        :return:
        """
        # <<<< Input reshaping
        input_shape = inputs.get_shape()
        ndim = input_shape.ndims
        channel_axis = ndim - 1  # assuming channel last
        assert int(input_shape[channel_axis]) == self._channels, \
            'Innermost dimension of inputs = %d, does not match number of channels = %d' % \
            (int(input_shape[channel_axis]), self._channels)
        # Convert to (channels, 1, batch) format by commuting channels to front
        # and then collapsing.
        order = list(range(ndim))
        order.pop(channel_axis)
        order.insert(0, channel_axis)
        inputs = tf.transpose(inputs, order)
        shape = tf.shape(inputs)
        inputs = tf.reshape(inputs, (shape[0], 1, -1))  # (channels, 1, ?)
        # Input reshaping >>>>

        logits_cdf = self._logits_cdf(inputs, stop_gradient=stop_gradient)
        cdf = tf.nn.sigmoid(logits_cdf)

        # Convert back to input tensor shape.
        order = list(range(1, ndim))
        order.insert(channel_axis, 0)
        cdf = tf.reshape(cdf, shape)
        cdf = tf.transpose(cdf, order)

        return cdf

    def pdf(self, inputs, stop_gradient=False):
        """
        Compute model PDF for a batch of inputs. The innermost axis of inputs will be aligned with the "channels"
        (independent distributions) modeled by the prior, so inputs[...,n] will be evaluated by the nth prior model
        distribution (dimensions within inputs[...,n] are treated as i.i.d., i.e., all coming from the nth prior model).
        :param inputs: Tensor, whose last dimension = number of channels in the prior model; this is typically
            [batch, z_height, z_width, channels] for a batch of latent image representations in convolutional VAEs.
        :param stop_gradient: Boolean. Whether to add `tf.stop_gradient` calls so
            that the gradient of the output with respect to the density model
            parameters is disconnected (the gradient with respect to `inputs` is
            left untouched).
        :return:
        """
        if tf.executing_eagerly():
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                cdf = self.cdf(inputs, stop_gradient=stop_gradient)
            pdf = tape.gradient(cdf, inputs)
        else:
            cdf = self.cdf(inputs, stop_gradient=stop_gradient)
            pdf = tf.gradients(cdf, inputs)[0]
        return pdf

    def inverse_cdf(self, xi, method='bisection', max_iterations=1000, tol=1e-9, **kwargs):
        float_type = 'float32'
        if method == 'bisection':
            # https://calculus.subwiki.org/wiki/Bisection_method
            # if kwargs.get('init_interval', None):
            #     init_interval = kwargs['init_interval']
            # else:
            #     init_interval = [-1, 1]
            init_interval = [-1, 1]
            left_endpoints = tf.ones_like(xi, dtype=float_type) * init_interval[0]
            right_endpoints = tf.ones_like(xi, dtype=float_type) * init_interval[1]

            def f(z):
                return self.cdf(z, stop_gradient=True) - xi

            while True:
                if tf.reduce_all(f(left_endpoints) < 0):
                    break
                else:
                    left_endpoints = left_endpoints * 2
            while True:
                if tf.reduce_all(f(right_endpoints) > 0):
                    break
                else:
                    right_endpoints = right_endpoints * 2

            for i in range(max_iterations):
                mid_pts = 0.5 * (left_endpoints + right_endpoints)
                mid_vals = f(mid_pts)
                pos = mid_vals > 0
                non_pos = tf.logical_not(pos)
                neg = mid_vals < 0
                non_neg = tf.logical_not(neg)
                # pos, non_pos, neg, non_neg = map(lambda x: tf.cast(x, float_type), (pos, non_pos, neg, non_neg))
                # note that end points are not updated for coordinates with f(mid_pts)==0 (solution already found at mid)
                left_endpoints = left_endpoints * tf.cast(non_neg, float_type) + mid_pts * tf.cast(neg, float_type)
                right_endpoints = right_endpoints * tf.cast(non_pos, float_type) + mid_pts * tf.cast(pos, float_type)
                if tf.reduce_all(tf.logical_and(non_pos, non_neg)) or \
                                tf.reduce_min(right_endpoints - left_endpoints) <= tol:
                    print('bisection terminated after %d its' % i)
                    break

            if kwargs.get('return_np', False):
                mid_pts = mid_pts.numpy()

            return mid_pts

        elif method == 'newton':
            # https://calculus.subwiki.org/wiki/Newton%27s_method_for_root-finding_for_a_function_of_one_variable
            # from scipy.stats import norm
            # x = tf.constant(norm.ppf(xi, loc=0, scale=self.init_scale / 4))
            # record = []
            # for i in range(max_iterations):
            #     x_prev = x.numpy()
            #     cdf, pdf = self.cdf_pdf(xi)
            #     x = x - (cdf - xi) / pdf
            #     diff = np.linalg.norm(x.numpy() - x_prev)
            #     record.append(diff)
            #
            # return record
            raise NotImplementedError

    def logpdf(self, inputs, pdf_lower_bound=1e-10, stop_gradient=False):
        """
        Compute log PDF; see self.pdf
        :param inputs:
        :param lower_bound: threshold the pdf to be at least value before
        taking log, to avoid getting NaN.
        :param stop_gradient:
        :return:
        """
        pdf = self.pdf(inputs=inputs, stop_gradient=stop_gradient)
        if pdf_lower_bound:
            pdf = math_ops.lower_bound(pdf, pdf_lower_bound)
        return tf.math.log(pdf)

    def cdf_pdf(self, inputs, stop_gradient=False):
        """
        Compute model CDF and PDF (manually) for a batch of inputs. The innermost axis of inputs will be aligned with the "channels"
        (independent distributions) modeled by the prior, so inputs[...,n] will be evaluated by the nth prior model
        distribution (dimensions within inputs[...,n] are treated as i.i.d., i.e., all coming from the nth prior model).
        :param inputs: Tensor, whose last dimension = number of channels in the prior model; this is typically
            [batch, z_height, z_width, channels] for a batch of latent image representations in convolutional VAEs.
        :param stop_gradient: Boolean. Whether to add `tf.stop_gradient` calls so
            that the gradient of the output with respect to the density model
            parameters is disconnected (the gradient with respect to `inputs` is
            left untouched).
        :return:
        :param inputs:
        :param stop_gradient:
        :return:
        """
        # <<<< Input reshaping
        input_shape = inputs.get_shape()
        ndim = input_shape.ndims
        channel_axis = ndim - 1  # assuming channel last
        assert int(input_shape[channel_axis]) == self._channels, \
            'Innermost dimension of inputs = %d, does not match number of channels = %d' % \
            (int(input_shape[channel_axis]), self._channels)
        # Convert to (channels, 1, batch) format by commuting channels to front
        # and then collapsing.
        order = list(range(ndim))
        order.pop(channel_axis)
        order.insert(0, channel_axis)
        inputs = tf.transpose(inputs, order)
        shape = tf.shape(inputs)
        inputs = tf.reshape(inputs, (shape[0], 1, -1))  # (channels, 1, ?)
        # Input reshaping >>>>

        logits = inputs  # has shape [channels, d_1, batch], with d_1 = 1
        pdf = None
        for i in range(len(self.dims) + 1):
            if tf.executing_eagerly():
                matrix = tf.nn.softplus(self._matrices_[i])
            else:
                matrix = self._matrices[i]
            # has shape r_k by d_k (using appendix's notation, where k=i+1)
            if stop_gradient:
                matrix = tf.stop_gradient(matrix)
            logits = tf.linalg.matmul(matrix, logits)  # has shape [channels, r_k, batch]

            bias = self._biases[i]
            if stop_gradient:
                bias = tf.stop_gradient(bias)
            logits += bias

            # activations = tf.linalg.matmul(matrix, logits) + bias
            # logits = activations

            if i < len(self._factors):
                if tf.executing_eagerly():
                    factor = tf.math.tanh(self._factors_[i])
                else:
                    factor = self._factors[i]
                if stop_gradient:
                    factor = tf.stop_gradient(factor)
                # tanh = tf.math.tanh(activations)
                tanh = tf.math.tanh(logits)
                logits += factor * tanh

                tanh_grad = 1 - tanh ** 2
                nonlinearity_grad = 1 + factor * tanh_grad
            else:
                cdf = tf.nn.sigmoid(logits)
                nonlinearity_grad = cdf * (1 - cdf)  # σ'(x) = σ(x) * (1 - σ(x))
                # alternatively could have computed the above directly in terms
                # of logistic PDF, perhaps with more numerical stability:
                # exp_neg = tf.exp(-logits)
                # nonlinearity_grad = exp_neg / (1 + exp_neg)**2
            # Now nonlinearity_grad has the same shape as logits, i.e., has shape [channels, r_k, batch];
            # we convert its shape to [batch, channels, r_k, 1] to multiply row-wise with the r_k x d_k matrix
            nonlinearity_grad = tf.transpose(nonlinearity_grad, [2, 0, 1])  # [batch, channels, r_k]
            nonlinearity_grad = tf.expand_dims(nonlinearity_grad, -1)  # broadcast for multiplication
            jacobian = nonlinearity_grad * matrix  # [batch, channels, r_k, d_k]

            if pdf is None:
                pdf = jacobian
            else:
                pdf = tf.matmul(jacobian, pdf)

        pdf = pdf[..., 0]  # [batch, channels, 1, 1] -> [batch, channels, 1]
        pdf = tf.transpose(pdf, [1, 2, 0])  # [batch, channels, 1] -> [channels, 1, batch]

        # <<<< Output reshaping
        # Convert back to input tensor shape.
        order = list(range(1, ndim))
        order.insert(channel_axis, 0)
        cdf = tf.reshape(cdf, shape)
        cdf = tf.transpose(cdf, order)
        pdf = tf.reshape(pdf, shape)
        pdf = tf.transpose(pdf, order)
        # >>>> Output reshaping

        return cdf, pdf


def get_runname(args_dict):
    """
    Given a dictionary of cmdline arguments, return a string that identifies the training run.
    :param args_dict:
    :return:
    """
    import os
    config_strs = []  # ['key1=val1', 'key2=val2', ...]

    for key, val in args_dict.items():
        if isinstance(val, (list, tuple)):
            val_str = '_'.join(map(str, val))
            config_strs.append('%s=%s' % (key, val_str))

    for key in ('init_scale', 'lr', 'its', 'tol'):
        config_strs.append('%s=%s' % (key, args_dict[key]))

    script_name = os.path.splitext(os.path.basename(__file__))[0]  # current script name, without extension
    return '-'.join([script_name] + config_strs)


def create_model(args):
    model = BMSHJ2018Prior(args.num_channels, dims=args.dims, init_scale=args.init_scale)
    return model


def train(args):
    tf.reset_default_graph()

    num_channels = args.num_channels
    dims = args.dims
    init_scale = args.init_scale
    model = BMSHJ2018Prior(num_channels, dims=dims, init_scale=init_scale)

    # if hasattr(args, 'runname') and args['runname']:
    #     runname = args.runname
    # else:
    #     import time, datetime
    #     ts = time.time()
    #     runname = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

    runname = get_runname(vars(args))
    data = np.load(args.data_path)

    lr = args.lr
    its = args.its
    tol = args.tol
    logging_freq = args.logging_freq
    plot = args.plot
    if plot:
        import matplotlib.pyplot as plt

    checkpoint_dir = args.checkpoint_dir

    import os
    save_dir_name = runname
    save_dir = os.path.join(checkpoint_dir, save_dir_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_name = os.path.join(save_dir, 'prior_model')

    import json
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:  # will overwrite existing
        json.dump(vars(args), f, indent=4, sort_keys=True)

    assert not tf.executing_eagerly()
    X = tf.placeholder(data.dtype, [None, num_channels])
    pdf_lower_bound = 1e-10
    # pdf = model.pdf(X, False)
    # [cdf, pdf] = model.cdf_pdf(X, False)
    # pdf = math_ops.lower_bound(pdf, pdf_lower_bound)
    # log_likelihood = tf.math.log(pdf)
    # log_likelihood = model.logpdf(X, False)
    pdf = model.pdf(X, stop_gradient=False)

    pdf = math_ops.lower_bound(pdf, pdf_lower_bound)
    log_likelihood = tf.math.log(pdf)
    print(log_likelihood)
    loss = - tf.reduce_mean(log_likelihood)

    optimizer = tf.train.AdamOptimizer(lr)
    # optimizer = tf.train.AdadeltaOptimizer()
    train_step = optimizer.minimize(loss)

    record = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        prev_loss = float('inf')
        for it in range(its):
            sess.run(train_step, feed_dict={X: data})
            loss_ = sess.run(loss, feed_dict={X: data})
            loss_ = float(loss_)

            if abs(prev_loss - loss_) / abs(loss_) < tol:
                break

            if it % logging_freq == 0 or it + 1 == its:
                print('it=%d,\t\tloss=%g' % (it, loss_))
                record.append(dict(it=it, loss=loss_))

                if plot:
                    # plot p(x)
                    xlim = [-5, 5]
                    xs = np.linspace(*xlim)
                    # figsize=None
                    figsize = (12, 8)
                    plt.figure(figsize=figsize)

                    xs_feed = np.tile(xs[..., None], num_channels)  # len(xs) by num_channels
                    q_xs = sess.run(pdf, feed_dict={X: xs_feed})

                    h, v = 2, 4
                    for k in range(h * v):
                        plt.subplot(h, v, k + 1)
                        plt.plot(xs, q_xs[:, k], label='$q(x)$')

                        bins = 31
                        plt.hist(data[:, k].ravel(), bins=bins, density=True, alpha=0.4, label='$\hat q(z)$')

                        plt.xlim(xlim)
                        plt.title('channel %d, it %d' % (k, it))
                    # plt.ylim([0, 2])

                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, runname + '_it=%d.png' % it))
                    # plt.show()

        model.save_weights(model_name)

    with open(os.path.join(save_dir, 'record.json'), 'w') as f:  # will overwrite existing
        json.dump(record, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    # High-level options.
    # parser.add_argument(
    #     "--verbose", "-V", action="store_true",
    #     help="If set, use tf.logging.INFO; otherwise use tf.logging.WARN, for tf.logging")

    parser.add_argument(
        "--checkpoint_dir", default="checkpoints",
        help="Directory where to save/load model checkpoints.")

    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for reproducibility")

    # Model architecture
    parser.add_argument("--num_channels", type=int)
    parser.add_argument('--dims', nargs='*', type=int, default=[3, 3, 3])
    parser.add_argument("--init_scale", default=1.)
    parser.add_argument('--data_path')
    # parser.add_argument('--runnname', default=None)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--its", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--logging_freq", type=int, default=10)

    parser.add_argument('--plot', action="store_true", help='If set, will plot')

    args = parser.parse_args()

    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)

    train(args)
