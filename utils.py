# Common utility methods used across the codebase.
# Yibo Yang, 2022


from keras.callbacks import ReduceLROnPlateau


class MyReduceLROnPlateauCallback(ReduceLROnPlateau):
    """
    My fancier version of keras.callbacks.ReduceLROnPlateau that allows a 'warmup' period during which no action is taken.
    """

    def __init__(self,
                 monitor='val_loss',
                 factor=0.1,
                 patience=10,
                 verbose=0,
                 mode='auto',
                 min_delta=1e-4,
                 warmup=0,
                 cooldown=0,
                 min_lr=0,
                 **kwargs):
        """
        Same as super class, except no action is taken until the epoch >= `warmup`.
        :param monitor:
        :param factor:
        :param patience:
        :param verbose:
        :param mode:
        :param min_delta:
        :param warmup:
        :param cooldown:
        :param min_lr:
        :param kwargs:
        """
        super().__init__(
            monitor=monitor,
            factor=factor,
            patience=patience,
            verbose=verbose,
            mode=mode,
            min_delta=min_delta,
            cooldown=cooldown,
            min_lr=min_lr,
            **kwargs)
        self.warmup = warmup

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.warmup:
            super().on_epoch_end(epoch, logs)


# My custom logging code for logging in JSON lines ("jsonl") format
import json


class MyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyJSONEncoder, self).default(obj)


def get_json_logging_callback(log_file_path, buffering=1, **preprocess_float_kwargs):
    log_file = open(log_file_path, mode='wt', buffering=buffering)
    json_logging_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs_dict: log_file.write(
            json.dumps({'epoch': epoch, **preprocess_float_dict(logs_dict, **preprocess_float_kwargs)},
                       cls=MyJSONEncoder) + '\n'),
        on_train_end=lambda logs: log_file.close()
    )
    return json_logging_callback


# Commonly used utility routines for organizing/keeping track of my experiments.
def get_runname(args_dict, record_keys=tuple(), prefix=''):
    """
    Given a dictionary of cmdline arguments, return a string that identifies the training run.
    :param args_dict:
    :param record_keys: a tuple/list of keys that is a subset of keys in args_dict that will be used to form the runname
    :return:
    """
    kv_strs = []  # ['key1=val1', 'key2=val2', ...]

    for key in record_keys:
        val = args_dict[key]
        if isinstance(val, (list, tuple)):  # e.g., 'num_layers: [10, 8, 10] -> 'num_layers=10_8_10'
            val_str = '_'.join(map(str, val))
        else:
            val_str = str(val)
        kv_strs.append('%s=%s' % (key, val_str))

    return '-'.join([prefix] + kv_strs)


class AttrDict(dict):
    # https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_args_as_obj(args):
    """
    Get an object specifying options/hyper-params from a JSON file or a Python dict; simulates the result of argparse.
    No processing is done if the input is of neither type (assumed to already be in an obj format).
    :param args: either a dict-like object with attributes specifying the model, or the path to some args.json file
    containing the args (which will be loaded and converted to a dict).
    :return:
    """
    if isinstance(args, str):
        import json
        with open(args) as f:
            args = json.load(f)
    if isinstance(args, dict):
        args = AttrDict(args)
    return args


def config_dict_to_str(args_dict, record_keys=tuple(), leave_out_falsy=True, prefix=None,
                       primary_delimiter='-', secondary_delimiter='_'):
    """
    Given a dictionary of cmdline arguments, return a string that identifies the training run.
    :param args_dict:
    :param record_keys: a tuple/list of keys that is a subset of keys in args_dict that will be used to form the runname
    :param leave_out_falsy: whether to skip keys whose values evaluate to falsy (0, None, False, etc.)
    :param primary_delimiter: the char to delimit different key-value paris
    :param secondary_delimiter: the delimiter within each key or value string (e.g., when the value is a list of numbers)
    :return:
    """
    kv_strs = []  # ['key1=val1', 'key2=val2', ...]

    for key in record_keys:
        val = args_dict[key]
        if leave_out_falsy and not val:
            continue
        if isinstance(val, (list, tuple)):  # e.g., 'num_layers: [10, 8, 10] -> 'num_layers=10_8_10'
            val_str = secondary_delimiter.join(map(str, val))
        else:
            val_str = str(val)
        kv_strs.append('%s=%s' % (key, val_str))

    if prefix:
        substrs = [prefix] + kv_strs
    else:
        substrs = kv_strs
    return primary_delimiter.join(substrs)


def preprocess_float_dict(d, format_str='.6g', as_str=False):
    # preprocess the floating values in a dict so that json.dump(dict) looks nice
    import tensorflow as tf
    import numpy as np
    res = {}
    for (k, v) in d.items():
        if isinstance(v, (float, np.floating)) or tf.is_tensor(v):
            if as_str:
                res[k] = format(float(v), format_str)
            else:
                res[k] = float(format(float(v), format_str))
        else:  # if not some kind of float, leave it be
            res[k] = v
    return res


def get_time_str():
    import datetime
    try:
        from configs import strftime_format
    except ImportError:
        strftime_format = "%Y_%m_%d~%H_%M_%S"

    time_str = datetime.datetime.now().strftime(strftime_format)
    return time_str


def get_np_datasets(np_file, batchsize, append_channel_dim=False, get_validation_data=True):
    assert np_file.endswith('.npy') or np_file.endswith('.npz')

    import numpy as np
    import tensorflow as tf
    import os

    def get_dataset(ar_path, repeat):
        X = np.load(ar_path).astype('float32')
        if append_channel_dim:  # convolutional models often require data to have a channel dim
            X = X[..., np.newaxis]
        dataset = tf.data.Dataset.from_tensor_slices(X)
        dataset = dataset.shuffle(len(X), reshuffle_each_iteration=True)
        if repeat:
            dataset = dataset.repeat()
        dataset = dataset.batch(batchsize)
        return dataset

    train_dataset = get_dataset(np_file, repeat=True)

    if not get_validation_data:
        return train_dataset
    else:
        validation_dataset = None
        if 'train' in np_file:  # dataset named as such comes with a validation set
            val_dataset = None
            if os.path.isfile(np_file.replace('train', 'val')):
                val_dataset = np_file.replace('train', 'val')
            elif os.path.isfile(np_file.replace('train', 'test')):
                val_dataset = np_file.replace('train', 'test')
            if val_dataset:
                validation_dataset = get_dataset(val_dataset, repeat=False)
                print(f'Validating on {val_dataset}')

        if validation_dataset is None:
            print(f"Couldn't find validation data for {np_file}; validating on a subset of train data")
            validation_dataset = train_dataset
        return train_dataset, validation_dataset


# For experiments on images.
import numpy as np
import tensorflow as tf


def reshape_spatially_as(x, y):
    """
    Crop away extraneous padding from upsampled tfc.SignalConv2D; used by the decoder for decompression.
    :param x: 3D tensor to be reshaped spatially
    :param y: target 3D tensor
    :return:  reshaped x
    """
    y_shape = tf.shape(y)
    return x[:, :y_shape[1], :y_shape[2], :]


def read_png(filename, channels=3):
    """Loads a PNG image file."""
    string = tf.io.read_file(filename)
    return tf.image.decode_image(string, channels=channels)


def write_png(filename, image):
    """Saves an image to a PNG file."""
    string = tf.image.encode_png(image)
    tf.io.write_file(filename, string)


def check_image_size(image, patchsize):
    shape = tf.shape(image)
    return shape[0] >= patchsize and shape[1] >= patchsize and shape[-1] == 3


def process_image(image, patchsize=None, img_channels=3):
    if patchsize is not None:
        image = tf.image.random_crop(image, (patchsize, patchsize, img_channels))
    return tf.cast(image, tf.float32)


def augment_image(image):
    # also maybe apply random rotation
    return tf.image.random_flip_left_right(image, seed=None)


def maybe_pad_img(x, factor: int):
    """
    Given an image x, add the minimum amount of padding around it such that the padded img has height and width that
    are both divisible by a number 'factor'. This is useful for img compression with convolutional autoencoders, where
    'factor' represents the total downsampling factor of the encoder: when the dimensions of x can't be evenly
    divided by 'factor', the reconstruction of x as output by the decoder can have a shape that doesn't match
    x, causing troubles with later computations (e.g., when the img is 34 x 34 and the encoder downsamples by 4, the
    reconstruction might turn out to be 32 x 32, or 36 x 36, depending on how the convolutions are done).
    To avoid this, we can first pad the img so that its dimensions can be divided evenly by 'factor', feed the padded
    img to the autoencoder, then later crop the padding out of the reconstruction.
    Return x_padded, offset
    :param x: a single [H, W, C] img-like tensor
    :param factor: integer factor used to determine the amount of padding. No padding is done if 'factor' already evenly
    divides into the dimensions of x.
    :return: x_padded, offset; x_padded is a potentially padded version of x whose height and width are divisible by
    div, and such that, x_padded[offset[0]: (offset[0] + x_size[0]), offset[1]:(offset[1] + x_size[1])] == x
    """
    assert len(x.shape) == 3, 'must be a single RGB image'
    img_shape = tf.shape(x)[:2]
    factor = tf.constant([factor, factor], dtype=tf.int32)
    ratio = tf.math.ceil(img_shape / factor)  # say cel([768, 512] / [100, 100]) = [8, 6]
    ratio = tf.cast(ratio, tf.int32)
    padded_shape = tf.multiply(ratio, factor)
    if tf.reduce_all(padded_shape == img_shape):  # special case, no need for padding
        return x, tf.constant([0, 0], dtype=tf.int32)

    # offset as in the top left corner of the crop; https://www.tensorflow.org/api_docs/python/tf/image/crop_to_bounding_box
    offset = tf.cast(tf.math.floor((padded_shape - img_shape) / 2), tf.int32)
    paddings = np.zeros([3, 2], dtype='int32')
    slack = padded_shape - img_shape  # e.g., [800, 600] - [768, 512] = [32, 88]
    # pad around center
    paddings[0:2, 0] = np.floor(slack / 2)  # e.g., [16, 44]
    paddings[0:2, 1] = slack - np.floor(slack / 2)
    x_padded = tf.pad(x, paddings, 'reflect')

    assert tf.reduce_all(x_padded[offset[0]: (offset[0] + img_shape[0]), offset[1]:(offset[1] + img_shape[1])] == x)
    return x_padded, offset


def read_npy_file_helper(file_name_in_bytes):
    ### for reading images in .npy format
    # data = np.load(file_name_in_bytes.decode('utf-8'))
    data = np.load(file_name_in_bytes)  # turns out this works too without decoding to str first
    # assert data.dtype is np.float32   # needs to match the type argument in the caller tf.data.Dataset.map
    return data


def get_custom_dataset(split, file_glob, args):
    """Creates input data pipeline from custom PNG images.
    :param split:
    :param file_glob:
    :param args:
    """
    import glob
    with tf.device("/cpu:0"):
        files = sorted(glob.glob(file_glob))
        if not files:
            raise RuntimeError(f"No images found with glob '{file_glob}'.")
        dataset = tf.data.Dataset.from_tensor_slices(files)
        if split == 'eval':
            drop_remainder = False
        else:  # for train or validation
            dataset = dataset.shuffle(len(files), reshuffle_each_iteration=True)
            drop_remainder = True  # as set in the original tfc source code; perhaps done for optimization purposes

        if split == "train":
            dataset = dataset.repeat()

        # if '.npy' in args.train_glob:  # reading numpy arrays directly instead of from images
        #    dataset = dataset.map(  # https://stackoverflow.com/a/49459838
        #        lambda item: tuple(tf.numpy_function(read_npy_file_helper, [item], [tf.float32, ])),
        #        num_parallel_calls=args.preprocess_threads)
        # else:
        #    dataset = dataset.map(
        #        read_png, num_parallel_calls=args.preprocess_threads)
        # dataset = dataset.map(lambda x: crop_image(x, args.patchsize))
        if not hasattr(args, 'patchsize'):
            args.patchsize = None
        if not hasattr(args, 'preprocess_threads'):
            args.preprocess_threads = 16
        if '.npy' in file_glob:  # reading numpy arrays directly instead of from images
            dataset = dataset.map(  # https://stackoverflow.com/a/49459838
                lambda file_name: tuple(tf.numpy_function(read_npy_file_helper, [file_name], [tf.float32, ])),
                num_parallel_calls=args.preprocess_threads)
            dataset = dataset.map(lambda x: process_image(x, args.patchsize),
                                  num_parallel_calls=args.preprocess_threads)
        else:
            dataset = dataset.map(
                lambda x: process_image(read_png(x), args.patchsize),
                num_parallel_calls=args.preprocess_threads)

        dataset = dataset.batch(args.batchsize, drop_remainder=drop_remainder)
    return dataset


def psnr_to_float_mse(psnr):
    return 10 ** (-psnr / 10)


def float_mse_to_psnr(float_mse):
    return -10 * np.log10(float_mse)


# Math routines
def softplus_inverse(x):
    """Helper which computes the function inverse of `tf.nn.softplus`."""
    return tf.math.log(tf.math.expm1(x))


softplus_inv_1 = softplus_inverse(1.0)


def transform_scale_indexes(indexes, type='softplus'):
    """
    (Optionally) transform the nn output that is used as 'indexes' for building a tfc.LocationScaleIndexedEntropyModel.
    For the conditional entropy model implemented by tfc, the 'indexes' is ideally an integer in {0,1,...,num_scales-1}.
    In tfc examples, the nn output is directly used as 'indexes' in a conditional entropy model, which then clips
    'indexes' into the correct range (and rounds to integers at test time). However it might make sense to preprocess
    the nn output first to ensure it's in the right range, or at least to ensure it's positive (e.g., with softplus
    or exp).
    :param indexes:
    :return:
    """
    if type == 'softplus':
        return tf.nn.softplus(indexes + softplus_inv_1)
    elif type == 'exp':
        return tf.exp(indexes)
    else:
        return indexes  # this is what's used in tfc examples -- raw nn output is used as 'indexes' for indexed entropy model


def diag_normal_from_features(features, name=None, split_axis=-1, scale_lb=None, scale_lb_reparam=False, scale_ub=None):
    """
    Constructs a diagonal Gaussian, by extracting mean and std parameters from a tensor of features.
    :param features: either a tf tensor, or a tuple of 2 tf tensors corresponding to (mean, scale)
    :param split_axis: which axis to split features along, in order to form mean and scale tensors
    :return:
    """
    import tensorflow_probability as tfp
    from tensorflow_compression.python.ops import math_ops

    tfd = tfp.distributions

    if isinstance(features, tuple):
        mu, sigma = features
    else:
        mu, sigma = tf.split(features, num_or_size_splits=2, axis=split_axis)
    sigma = tf.nn.softplus(sigma + softplus_inv_1)
    if scale_lb is not None:
        if scale_lb_reparam:  # using reparameterization to enforce lb
            sigma += scale_lb
        else:
            sigma = math_ops.lower_bound(sigma, scale_lb)
    if scale_ub is not None:
        sigma = math_ops.upper_bound(sigma, scale_ub)
    return tfd.Normal(loc=mu, scale=sigma, name=name)


def diag_gaussian_rdf(variances, num_points=50, distortion='mse'):
    """
    Compute rate-distortion function of a Gaussian source with a diagonal
    covariance mat, under either squared or mean squared distortion.
    The R(D) in this case has no closed-form expression in general (as a
    function of D), but can still be traced out analytically by the reverse
    water filling algorithm (see Cover and Thomas textbook Ch 10.3.3).
    This procedure will produce pairs of (D, R) points that span the whole R(D)
    curve.
    :param variances:
    :param num_points:
    :param distortion:
    :return:
    """
    distortion = distortion.lower()
    assert distortion in ('se', 'mse')
    max_var = np.max(variances)
    n = len(variances)
    lambs = np.linspace(0, max_var, num_points)
    # vars_rep = np.stack([variances] * num_lambdas, axis=0)  # each row is the vector of variances
    vars_rep = np.repeat([variances], num_points, axis=0)  # each row is the vector of variances
    lambs_rep = np.repeat([lambs], n, axis=0).T  # each column is a copy of lambs

    D_mat = np.minimum(vars_rep, lambs_rep)  # reverse water filling
    Rs = 0.5 * np.sum(np.log(vars_rep) - np.log(D_mat), axis=-1)
    Ds = np.sum(D_mat, axis=-1)

    if distortion == 'mse':
        Ds /= n
    return (Ds, Rs)


# LB estimator stuff
def get_xi_samples(E_log_us, log_Ck_samples):
    r"""
    E_log_us and log_Ck_samples are two arrays, each of length 2M, as produced by the
    rdlb.eval. Each pair of E_log_u and log_Ck_sample is produced by one global optimization
    run on a separate batch of k samples of X.
    We use the first M entries of log_Ck_samples to set alpha, and the second M parallel
    entries of E_log_us and log_Ck_samples to compute M \xi samples, with
    xi_samples[i] := - E_log_us[i] - exp(log_Ck_samples[i] - log_alpha)  - log_alpha + 1
    ($\xi = - \frac{1}{k} \sum_i \log u (X_i) - C_k/alpha - \log \alpha + 1$). The
    first M entries of E_log_us are simply ignored.

    :param E_log_us:
    :param log_Ck_samples:
    :return:
    """
    assert len(E_log_us) == len(log_Ck_samples)
    M = int(len(E_log_us) / 2)
    assert M * 2 == len(E_log_us)
    from scipy.special import logsumexp
    log_alpha = logsumexp(log_Ck_samples[:M]) - np.log(float(M))
    xi_samples = -E_log_us[M:] - np.exp(log_Ck_samples[M:] - log_alpha) - log_alpha + 1
    return xi_samples


def parse_lamb(path, strip_pardir=False):
    # search for a numeric string (possibly in scientific notation) for the lamb value
    import os, re
    if strip_pardir:
        path = os.path.basename(path)
    return re.search('lamb=(\d*\.?\d+(?:e[+-]?\d+)?)', path).group(1)


def aggregate_lb_results(res_files):
    lambs_to_res_files = {}  # map each lamb to the list of files run with the same lamb
    for path in res_files:
        lamb_str = parse_lamb(path, strip_pardir=True)
        if lamb_str not in lambs_to_res_files:
            lambs_to_res_files[lamb_str] = []
        lambs_to_res_files[lamb_str].append(path)

    # Go through each lamb and collect all results with the same lamb (but maybe different seeds)
    res_dict = {}
    for lamb_str, npz_files in lambs_to_res_files.items():
        E_log_us = []
        log_Ck_samples = []
        for file in npz_files:  # these are all the same eval runs but with different seeds
            res = np.load(file)
            E_log_us.append(res['E_log_us'])
            log_Ck_samples.append(res['log_Ck_samples'])
        E_log_us = np.concatenate(E_log_us)
        log_Ck_samples = np.concatenate(log_Ck_samples)
        xi_samples_for_lamb = get_xi_samples(E_log_us, log_Ck_samples)
        res_dict[float(lamb_str)] = xi_samples_for_lamb

    return res_dict
