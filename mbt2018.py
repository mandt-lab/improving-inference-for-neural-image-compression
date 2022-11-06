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


Based on https://github.com/tensorflow/compression/blob/f9edc949fa58381ffafa5aa8cb37dc8c3ce50e7f/models/bmshj2018.py,
originally by the authors of tensorflow-compression.
Yibo Yang, 2021
"""

import argparse
import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
from absl import app
from utils import reshape_spatially_as


class AnalysisTransform(tf.keras.Sequential):
    """The analysis transform."""

    def __init__(self, num_filters):
        super().__init__(name="analysis")
        self.add(tf.keras.layers.Lambda(lambda x: x / 255.))
        self.add(tfc.SignalConv2D(
            num_filters, (5, 5), name="layer_0", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="gdn_0")))
        self.add(tfc.SignalConv2D(
            num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="gdn_1")))
        self.add(tfc.SignalConv2D(
            num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="gdn_2")))
        self.add(tfc.SignalConv2D(
            num_filters, (5, 5), name="layer_3", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=None))


class SynthesisTransform(tf.keras.Sequential):
    """The synthesis transform."""

    def __init__(self, num_filters):
        super().__init__(name="synthesis")
        self.add(tfc.SignalConv2D(
            num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_0", inverse=True)))
        self.add(tfc.SignalConv2D(
            num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_1", inverse=True)))
        self.add(tfc.SignalConv2D(
            num_filters, (5, 5), name="layer_2", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_2", inverse=True)))
        self.add(tfc.SignalConv2D(
            3, (5, 5), name="layer_3", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=None))
        self.add(tf.keras.layers.Lambda(lambda x: x * 255.))


class HyperAnalysisTransform(tf.keras.Sequential):
    """The analysis transform for the entropy model parameters."""

    def __init__(self, num_filters, num_output_filters=None):
        super().__init__(name="hyper_analysis")
        if not num_output_filters:
            num_output_filters = num_filters
        self.add(tfc.SignalConv2D(
            num_filters, (3, 3), name="layer_0", corr=True, strides_down=1,
            padding="same_zeros", use_bias=True,
            activation=tf.nn.relu))
        self.add(tfc.SignalConv2D(
            num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=tf.nn.relu))
        self.add(tfc.SignalConv2D(
            num_output_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
            padding="same_zeros", use_bias=False,
            activation=None))


# Architecture (mean-scale, no context model) based on Table 1 of https://papers.nips.cc/paper/8275-joint-autoregressive-and-hierarchical-priors-for-learned-image-compression.pdf
class HyperSynthesisTransform(tf.keras.Sequential):
    """The synthesis transform for the entropy model parameters."""

    def __init__(self, num_filters, num_output_filters=None):
        super().__init__(name="hyper_synthesis")
        if not num_output_filters:
            num_output_filters = num_filters * 2
        self.add(tfc.SignalConv2D(
            num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True, kernel_parameter="variable",
            activation=tf.nn.relu))
        self.add(tfc.SignalConv2D(
            int(num_filters * 1.5), (5, 5), name="layer_1", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True, kernel_parameter="variable",
            activation=tf.nn.relu))
        self.add(tfc.SignalConv2D(
            num_output_filters, (3, 3), name="layer_2", corr=False, strides_up=1,
            padding="same_zeros", use_bias=True, kernel_parameter="variable",
            activation=None))


class MBT2018Model(tf.keras.Model):
    """Main model class."""

    def __init__(self, lmbda, num_filters, num_scales, scale_min, scale_max):
        super().__init__()
        self.lmbda = lmbda
        self.num_scales = num_scales
        offset = tf.math.log(scale_min)
        factor = (tf.math.log(scale_max) - tf.math.log(scale_min)) / (
                num_scales - 1.)
        self.scale_fn = lambda i: tf.math.exp(offset + factor * i)
        self.analysis_transform = AnalysisTransform(num_filters)
        self.synthesis_transform = SynthesisTransform(num_filters)
        self.hyper_analysis_transform = HyperAnalysisTransform(num_filters)
        self.hyper_synthesis_transform = HyperSynthesisTransform(num_filters, num_output_filters=2 * num_filters)
        self.hyperprior = tfc.NoisyDeepFactorized(batch_shape=(num_filters,))
        self.build((None, None, None, 3))

    @classmethod
    def create_model(cls, args):
        return cls(args.lmbda, args.num_filters, args.num_scales, args.scale_min, args.scale_max)

    def call(self, x, training):
        """Computes rate and distortion losses."""
        entropy_model = tfc.LocationScaleIndexedEntropyModel(
            tfc.NoisyNormal, self.num_scales, self.scale_fn, coding_rank=3,
            compression=False)
        side_entropy_model = tfc.ContinuousBatchedEntropyModel(
            self.hyperprior, coding_rank=3, compression=False)

        y = self.analysis_transform(x)
        z = self.hyper_analysis_transform(y)  # no need for abs(y) as in bmshj2018.py
        z_hat, z_bits = side_entropy_model(z, training=training)
        mu, sigma = tf.split(self.hyper_synthesis_transform(z_hat), num_or_size_splits=2, axis=-1)
        if not training:
            mu = reshape_spatially_as(mu, y)
            sigma = reshape_spatially_as(sigma, y)
        sigma = tf.exp(sigma)  # make positive; will be clipped then quantized to scale_table anyway
        loc, indexes = mu, sigma
        y_hat, y_bits = entropy_model(y, indexes, loc=loc, training=training)
        x_hat = self.synthesis_transform(y_hat)
        if not training:
            x_hat = reshape_spatially_as(x_hat, x)

        # Total number of bits divided by total number of pixels.
        num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), y_bits.dtype)
        bits = y_bits + z_bits
        bpp = tf.reduce_sum(bits) / num_pixels
        # Mean squared error across pixels.
        axes_except_batch = list(range(1, len(x.shape)))  # should be [1,2,3]
        mses = tf.reduce_mean(tf.math.squared_difference(x, x_hat), axis=axes_except_batch)  # per img
        mse = tf.reduce_mean(mses)
        psnrs = 20 * np.log10(255) - 10 * tf.math.log(mses) / np.log(10)  # PSNR for each img in batch
        psnr = tf.reduce_mean(psnrs)
        # The rate-distortion Lagrangian.
        loss = bpp + self.lmbda * mse

        return dict(loss=loss, bpp=bpp, mse=mse, bits=bits, y_bits=y_bits, z_bits=z_bits, x_hat=x_hat, psnr=psnr)

    def train_step(self, x):
        with tf.GradientTape() as tape:
            res = self(x, training=True)
        variables = self.trainable_variables
        loss = res['loss']
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        for m in self.my_metrics:
            m.update_state(res[m.name])
        return {m.name: m.result() for m in self.my_metrics}

    def test_step(self, x):
        res = self(x, training=False)
        for m in self.my_metrics:
            m.update_state(res[m.name])
        return {m.name: m.result() for m in self.my_metrics}

    def predict_step(self, x):
        raise NotImplementedError("Prediction API is not supported.")

    def compile(self, **kwargs):
        super().compile(
            loss=None,
            metrics=None,
            loss_weights=None,
            weighted_metrics=None,
            **kwargs,
        )
        self.metric_names = ('loss', 'bpp', 'psnr')
        self.my_metrics = [tf.keras.metrics.Mean(name=name) for name in self.metric_names]  # can't use self.metrics

    def fit(self, *args, **kwargs):
        retval = super().fit(*args, **kwargs)
        # After training, fix range coding tables.
        self.set_entropy_model()  # the resulting self.entropy_model won't actually be saved if using model.save_weights
        return retval

    def set_entropy_model(self):
        self.entropy_model = tfc.LocationScaleIndexedEntropyModel(
            tfc.NoisyNormal, self.num_scales, self.scale_fn, coding_rank=3,
            compression=True)
        self.side_entropy_model = tfc.ContinuousBatchedEntropyModel(
            self.hyperprior, coding_rank=3, compression=True)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
    ])
    def compress(self, x):
        """Compresses an image."""
        # Add batch dimension and cast to float.
        x = tf.expand_dims(x, 0)
        x = tf.cast(x, dtype=tf.float32)
        y = self.analysis_transform(x)
        z = self.hyper_analysis_transform(y)  # no need for abs(y) as in bmshj2018.py
        # Preserve spatial shapes of image and latents.
        x_shape = tf.shape(x)[1:-1]
        y_shape = tf.shape(y)[1:-1]
        z_shape = tf.shape(z)[1:-1]
        z_hat, _ = self.side_entropy_model(z, training=False)
        mu, sigma = tf.split(self.hyper_synthesis_transform(z_hat), num_or_size_splits=2, axis=-1)
        sigma = tf.exp(
            sigma)  # make positive; will be clipped then quantized to scale_table anyway; see https://github.com/tensorflow/compression/blob/f9edc949fa58381ffafa5aa8cb37dc8c3ce50e7f/tensorflow_compression/python/entropy_models/continuous_indexed.py#L255
        loc, indexes = mu, sigma
        indexes = indexes[:, :y_shape[0], :y_shape[1], :]
        loc = loc[:, :y_shape[0], :y_shape[1], :]
        side_string = self.side_entropy_model.compress(z)
        string = self.entropy_model.compress(y, indexes, loc=loc)
        return string, side_string, x_shape, y_shape, z_shape

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(1,), dtype=tf.string),
        tf.TensorSpec(shape=(1,), dtype=tf.string),
        tf.TensorSpec(shape=(2,), dtype=tf.int32),
        tf.TensorSpec(shape=(2,), dtype=tf.int32),
        tf.TensorSpec(shape=(2,), dtype=tf.int32),
    ])
    def decompress(self, string, side_string, x_shape, y_shape, z_shape):
        """Decompresses an image."""
        z_hat = self.side_entropy_model.decompress(side_string, z_shape)
        mu, sigma = tf.split(self.hyper_synthesis_transform(z_hat), num_or_size_splits=2, axis=-1)
        sigma = tf.exp(sigma)  # make positive; will be clipped then quantized to scale_table anyway
        loc, indexes = mu, sigma
        indexes = indexes[:, :y_shape[0], :y_shape[1], :]
        loc = loc[:, :y_shape[0], :y_shape[1], :]
        y_hat = self.entropy_model.decompress(string, indexes, loc=loc)
        x_hat = self.synthesis_transform(y_hat)
        # Remove batch dimension, and crop away any extraneous padding.
        x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]
        # Then cast back to 8-bit integer.
        return tf.saturate_cast(tf.round(x_hat), tf.uint8)


def get_runname(args):
    from utils import config_dict_to_str
    model_name = os.path.splitext(os.path.basename(__file__))[0]
    runname = config_dict_to_str(vars(args), record_keys=('num_filters', 'lmbda'), prefix=model_name)
    return runname


# Unavoidable boilerplate below.
import boilerplate
from functools import partial

# Note: needed to specify as kwargs in partial; o/w would be incorrect ('argv' would receive the value for create_model)
main = partial(boilerplate.main, create_model=MBT2018Model.create_model, get_runname=get_runname)
parse_args = partial(boilerplate.parse_args, add_model_specific_args=MBT2018Model.add_model_specific_args)

if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
