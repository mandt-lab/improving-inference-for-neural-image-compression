import argparse
import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
from absl import app
from absl.flags import argparse_flags
import configs
from utils import read_png, write_png, get_custom_dataset


def parse_args(argv, add_model_specific_args=None):
    """
    Parses command line arguments.
    :param argv: A non-empty list of the command line arguments including program name, sys.argv is used if None.
    :param add_model_specific_args: a callable that adds model specific args to the parser.
    :return:
    """
    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # High-level options.
    parser.add_argument(
        "--verbose", "-V", action="store_true",
        help="Report progress and metrics when training or compressing.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint_dir", default="./checkpoints",
        help="Directory where to save/load model checkpoints.")
    if add_model_specific_args:
        # inspired by https://pytorch-lightning.readthedocs.io/en/latest/common/hyperparameters.html#argparser-best-practices
        sub_parser = parser.add_argument_group("Model")
        add_model_specific_args(sub_parser)

    subparsers = parser.add_subparsers(
        title="commands", dest="command",
        help="What to do: 'train' loads training data and trains (or continues "
             "to train) a new model. 'compress' reads an image file (lossless "
             "PNG format) and writes a compressed binary file. 'decompress' "
             "reads a binary file and reconstructs the image (in PNG format). "
             "input and output filenames need to be provided for the latter "
             "two options. Invoke '<command> -h' for more information.")

    # 'train' subcommand.
    train_cmd = subparsers.add_parser(
        "train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Trains (or continues to train) a new model. Note that this "
                    "model trains on a continuous stream of patches drawn from "
                    "the training image dataset. An epoch is always defined as "
                    "the same number of batches given by --steps_per_epoch. "
                    "For neural compression models, the validation"
                    "rate-distortion performance is computed with actual "
                    "quantization rather than the differentiable proxy loss. "
                    "Note that when using custom training images, the validation "
                    "set is simply a random sampling of patches from the "
                    "training set.")
    train_cmd.add_argument(
        "--batchsize", type=int, default=8,
        help="Batch size for training and validation.")
    train_cmd.add_argument(
        "--patchsize", type=int, default=None,
        help="Size of image patches for training; default (None) uses whole images, no random crops.")
    train_cmd.add_argument(
        "--epochs", type=int, default=100,
        help="Train up to this number of epochs. (One epoch is here defined as "
             "the number of steps given by --steps_per_epoch, not iterations "
             "over the full training dataset.)")
    train_cmd.add_argument(
        "--steps_per_epoch", type=int, default=10000,
        help="Perform validation and produce logs after this many batches.")
    train_cmd.add_argument(
        "--lr", type=float, default=1e-4,
        help="Initial lr to configure the optimizer with.")
    train_cmd.add_argument(
        "--lr_decay_factor", type=float, default=0.5,
        help="Mult lr by this everytime lr is reduced")
    train_cmd.add_argument(
        "--patience", type=int, default=10,
        help="Number of epochs of non-improvement before reducing lr.")
    train_cmd.add_argument(
        "--warmup", type=int, default=100,
        help="Don't start decaying lr until the number of epochs hits `warmup`.")
    train_cmd.add_argument(
        "--validation_steps", type=int, default=16,
        help="Total number of steps (batches of samples) to validate before stopping.")
    train_cmd.add_argument(
        "--preprocess_threads", type=int, default=16,
        help="Number of CPU threads to use for parallel decoding of training "
             "images.")

    # 'compress' subcommand.
    compress_cmd = subparsers.add_parser(
        "compress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads a PNG file, compresses it, and writes a TFCI file.")

    # 'decompress' subcommand.
    decompress_cmd = subparsers.add_parser(
        "decompress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads a TFCI file, reconstructs the image, and writes back "
                    "a PNG file.")

    # Arguments for both 'compress' and 'decompress'.
    for cmd, ext in ((compress_cmd, ".tfci"), (decompress_cmd, ".png")):
        cmd.add_argument(
            "input_file",
            help="Input filename.")
        cmd.add_argument(
            "output_file", nargs="?",
            help=f"Output filename (optional). If not provided, appends '{ext}' to "
                 f"the input filename.")

    # 'eval' subcommand.
    eval_cmd = subparsers.add_parser(
        "eval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluates model on a dataset of images and outputs results in .npz.")
    eval_cmd.add_argument(
        "--batchsize", type=int, default=1,
        help="Batch size; must use 1 for images with different sizes. Larger batch size might run faster.")

    eval_cmd.add_argument(
        "--cum_downsample_factor", type=int, default=64,
        help="Cumulative downsample factor by the analysis transform, to help decide on the amount of padding.")
    eval_cmd.add_argument(
        "--ckpt", type=str, default=None,
        help="Path to the checkpoint (either the directory containing the checkpoint (will use the latest), or"
             "full checkpoint name (should not have the .index extension)) to load;"
             "by default (None) uses the latest ckpt in the auto-generated run dir in checkpoint_dir/runname")
    eval_cmd.add_argument(
        "--results_dir", default="./results",
        help="Directory for storing compression stats/results; set to empty string '' to disable.")
    eval_cmd.add_argument(
        "--bits", default=False, action='store_true',
        help="Whether to actually compress to bits and also report bpp based on file size ('file_bpp')")
    eval_cmd.add_argument(
        "--no_cast_xhat", default=False, action='store_true',
        help="Don't cast img reconstruction to uint8 for evaluation (but still clip to the right range)."
             "This is useful, e.g., when the model output would be scaled to [0, 1] and compare with a"
             "float img in that space, like on GAN imgs.")

    for cmd, mode in zip((train_cmd, eval_cmd), ('training', 'evaluation')):
        cmd.add_argument(
            "--dataset", default=None,
            help=f"Name of dataset for {mode}; accepts a GAN class, a glob key defined in configs.dataset_to_globs"
                 f"(use --data_glob to provide custom glob string instead), or a path to a numpy data array.")
        cmd.add_argument(
            "--data_glob", type=str, default=None,
            help=f"Glob pattern identifying custom {mode} data. This pattern should"
                 "expand to a list of RGB image files.")
        cmd.add_argument(
            "--data_dim", type=int, default=None,
            help="Intrinsic data dimension for custom data generator.")

    # Parse arguments.
    args = parser.parse_args(argv[1:])
    args.script_name = argv[0]
    if args.command is None:
        parser.print_usage()
        sys.exit(2)
    return args


def train(args, create_model, get_runname):
    """
    Instantiates and trains the model.
    :param args: an object containing hparams, typically returned by argparse
    :param create_model: a callable that returns a model instance given args
    :param get_runname: a callable that returns a string identifying the run
    :return:
    """
    model = create_model(args)
    if args.lr:
        init_lr = args.lr
    else:
        init_lr = 1e-4
        print(f'No initial lr provided, defaulting to {init_lr}')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=init_lr))

    from utils import get_custom_dataset
    if args.dataset is not None:
        import configs
        if args.dataset in configs.biggan_class_names_to_ids:  # train on GAN
            import biggan
            # scale from [-1, 1] to [0, 255] to conform to img compression models
            post_process_fun = lambda x: (x + 1.) * 127.5
            sampler = biggan.get_sampler(args.dataset, args.data_dim, post_process_fun=post_process_fun)

            def gen():
                while True:
                    yield sampler(args.batchsize)

            train_data_generator = gen()
            train_dataset = None
        elif args.dataset in configs.dataset_to_globs.keys():
            file_glob = configs.dataset_to_globs[args.dataset]
            train_dataset = get_custom_dataset("train", file_glob, args)
            validation_dataset = get_custom_dataset("validation", file_glob, args)
        elif args.dataset.endswith('.npy') or args.dataset.endswith('.npz'):
            from utils import get_np_datasets
            train_dataset, validation_dataset = get_np_datasets(args.dataset, args.batchsize)
        else:
            raise NotImplementedError(f'No idea how to load dataset {args.dataset}')
    else:
        assert args.data_glob is not None  # train on custom images
        train_dataset = get_custom_dataset("train", args.data_glob, args)
        validation_dataset = get_custom_dataset("validation", args.data_glob, args)

    ##################### BEGIN: Good old bookkeeping #########################
    runname = get_runname(args)
    save_dir = os.path.join(args.checkpoint_dir, runname)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    from utils import get_time_str
    time_str = get_time_str()

    # log to file during training
    log_file_path = os.path.join(save_dir, f'record-{time_str}.jsonl')
    from utils import get_json_logging_callback
    file_log_callback = get_json_logging_callback(log_file_path)
    print(f'Logging to {log_file_path}')
    ##################### END: Good old bookkeeping #########################

    ### BEGIN: Set up train/val data for model.fit to run on either tf dataset or generator ###
    if train_dataset is not None:
        train_data = train_dataset.prefetch(tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.take(args.validation_steps)
        validation_data = validation_dataset.cache()
    else:  # train on infinite generator
        train_data = train_data_generator
        validation_dataset = tf.data.Dataset.from_tensor_slices([x for (i, x) in zip(range(args.validation_steps),
                                                                                     train_data_generator)])
        validation_data = validation_dataset.cache()
    ### END: Set up train/val data for model.fit to run on either tf dataset or generator ###

    #### BEGIN: set up learning rate schedule ####
    # https://keras.io/api/callbacks/reduce_lr_on_plateau/
    from utils import MyReduceLROnPlateauCallback
    if validation_data:
        monitor_loss = 'val_loss'
    else:  # monitor train loss
        monitor_loss = 'loss'
    reduce_lr = MyReduceLROnPlateauCallback(monitor=monitor_loss,
                                            mode='min',
                                            factor=args.lr_decay_factor,
                                            warmup=args.warmup,
                                            patience=args.patience,  # patience in terms of epochs
                                            min_delta=1e-4,
                                            min_lr=1e-6,
                                            verbose=1)
    #### END: set up learning rate schedule ####

    tmp_save_dir = os.path.join('/tmp/rdvae', save_dir)

    hist = model.fit(
        train_data,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_data=validation_data,
        validation_freq=1,
        verbose=int(args.verbose),
        callbacks=[
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.TensorBoard(
                log_dir=tmp_save_dir,
                histogram_freq=1, update_freq="epoch"),
            tf.keras.callbacks.experimental.BackupAndRestore(tmp_save_dir),
            file_log_callback,
            reduce_lr
        ],
    )
    records = hist.history
    ckpt_path = os.path.join(save_dir, f"ckpt-lmbda={args.lmbda}-epoch={args.epochs}-loss={records['loss'][-1]:.3f}")
    model.save_weights(ckpt_path)
    print('Saved checkpoint to', ckpt_path)
    return hist


def evaluate(args, create_model, get_runname):
    """
    Evaluate on a dataset
    :param args:
    :param create_model:
    :param get_runname:
    :return:
    """

    runname = get_runname(args)
    if not args.ckpt:  # use the latest checkpoint in run dir
        ckpt_dir = os.path.join(args.checkpoint_dir, runname)  # run dir
        restore_ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
        assert restore_ckpt_path is not None, f'No checkpoints found in {ckpt_dir}'
    else:
        if os.path.isdir(args.ckpt):
            ckpt_dir = args.ckpt
            restore_ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
            assert restore_ckpt_path is not None, f'No checkpoints found in {ckpt_dir}'
        else:
            restore_ckpt_path = args.ckpt

    model = create_model(args)
    load_status = model.load_weights(restore_ckpt_path).expect_partial()
    # load_status.assert_consumed()
    print('Loaded model weights from', restore_ckpt_path)
    if args.bits:
        model.set_entropy_model()

    max_pxl_val = 255.
    if args.dataset in configs.biggan_class_names_to_ids:
        import biggan
        # scale from [-1, 1] to [0, 255] to conform to img compression models
        post_process_fun = lambda x: (x + 1.) * 127.5
        sampler = biggan.get_sampler(args.dataset, args.data_dim, post_process_fun=post_process_fun)
        dataset = [sampler(args.batchsize)]  # just eval a single batch of images
        pad_img = False
    else:  # custom imgs
        pad_img = True
        args.batchsize = 1
        if args.data_glob:
            file_glob = args.data_glob
            args.dataset = 'custom_glob'
        else:
            assert args.dataset in configs.dataset_to_globs.keys()
            file_glob = configs.dataset_to_globs[args.dataset]
        dataset = get_custom_dataset("eval", file_glob, args)  # load eval dataset in special 'eval' mode

        if hasattr(model, 'cum_downsample_factors'):
            cum_downsample_factor = model.cum_downsample_factor
        elif hasattr(model, 'downsample_factors'):
            cum_downsample_factor = np.prod(model.downsample_factors)
        else:
            cum_downsample_factor = args.cum_downsample_factor
        print(f'Using cum_downsample_factor = {cum_downsample_factor}')

    from utils import maybe_pad_img

    batch_res_list = []  # list of dicts
    for x in dataset:
        batch_res = {}
        im_size = tf.shape(x)[1:-1]
        if pad_img:
            assert args.batchsize == 1, 'Currently padding only works on single images'
            # the padding stuff currently requires single image
            x_padded, pad_offset = maybe_pad_img(x[0], cum_downsample_factor)
            x_padded = tf.expand_dims(x_padded, 0)  # add back batch dimension
            # actually_padded = not tf.reduce_all(x_padded == x)
        else:
            x_padded = x
            pad_offset = None

        num_pixels_per_img = tf.cast(tf.reduce_prod(im_size), x.dtype)  # all imgs in x have the same shape
        # compress the (possibly) padded image
        if args.bits:
            assert len(x_padded) == 1
            # Borrowed from compress()
            tensors = model.compress(tf.cast(x_padded[0], tf.uint8))
            # Get a bitstring with the shape information and the compressed string.
            packed = tfc.PackedTensors()
            packed.pack(tensors)
            # with open(args.output_file, "wb") as f:
            #     f.write(packed.string)
            batch_res['bits'] = np.array([len(packed.string) * 8])
            batch_res['bpp'] = (batch_res['bits'] / num_pixels_per_img)  # [batchsize]

            x_hat = model.decompress(*tensors)
            x_hat = x_hat[None, ...]  # add batch dimension
        else:
            out = model(x_padded, training=False)
            if 'bits' in out:  # this should have shape [batchsize]
                batch_res['bits'] = out['bits']
                batch_res['bpp'] = (batch_res['bits'] / num_pixels_per_img)  # [batchsize]
            else:  # for handling models that don't implement per-image 'bits' or 'bpps'
                assert len(x_padded) == 1
                batch_res['bpp'] = np.array([float(out['bpp'])])  # just use the aggregate bpp of this one-image batch
            x_hat = out['x_hat']  # float
            if args.no_cast_xhat:
                # no casting to uint8, but still clip to the right range
                x_hat = tf.clip_by_value(x_hat, 0., max_pxl_val)
            else:
                x_hat = tf.saturate_cast(tf.round(x_hat), tf.uint8)  # decoder rounding, as in model.decompress

        if pad_offset is not None:
            # Only keep the valid img corresponding to x.
            x_hat = x_hat[:, pad_offset[0]:pad_offset[0] + im_size[0], pad_offset[1]:pad_offset[1] + im_size[1], :]

        x_hat = tf.cast(x_hat, tf.float32)  # cast back to float to compute metrics; still in [0, 255]
        batch_res['mse'] = tf.reduce_mean(tf.math.squared_difference(x, x_hat), axis=[1, 2, 3])  # [batchsize]
        batch_res['psnr'] = -10 * (np.log10(batch_res['mse']) - 2 * np.log10(max_pxl_val))
        if im_size[0] < 160 and im_size[1] < 160:  # hack to avoid tf.image.ssim_multiscale crashing on smaller imgs
            batch_res['msssim'] = tf.image.ssim(x, x_hat, max_pxl_val)  # technically this is just one scale, not 'ms'
        else:
            batch_res['msssim'] = tf.image.ssim_multiscale(x, x_hat, max_pxl_val)
        batch_res['msssim_db'] = -10. * tf.math.log(1 - batch_res['msssim']) / tf.math.log(10.)

        batch_res_list.append(batch_res)

    res_keys = batch_res.keys()
    results_arrs = {key: [] for key in res_keys}  # one np array for each eval field
    for key in res_keys:
        results_arrs[key] = np.concatenate([np.array(b[key]) for b in batch_res_list])

    # record the important fields in the name
    avg_rate_str = f"-bpp={results_arrs['bpp'].mean():.4g}"
    avg_distortion_str = f"-psnr={results_arrs['psnr'].mean():.4g}"

    prefix = 'brd' if args.bits else 'rd'  # 'br' meaning bitrate for when compress to file

    save_dir = args.results_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dataset_str = f'-dataset={args.dataset}'
    if hasattr(args, 'data_dim') and args.data_dim:  # record intrinsic dimension for GAN imgs
        dataset_str += f'-data_dim={args.data_dim}'
    save_path = os.path.join(save_dir,
                             f'{prefix}-{runname}{dataset_str}'
                             f'{avg_rate_str}{avg_distortion_str}.npz')
    np.savez(save_path, **results_arrs)

    for key in res_keys:
        arr = results_arrs[key]
        print('Avg {}: {:0.4f}'.format(key, arr.mean()))

    print('Saved results to', save_path)


def compress(args, create_model, get_runname):
    """Compresses an image."""
    # Load model and use it to compress the image.
    runname = get_runname(args)
    save_dir = os.path.join(args.checkpoint_dir, runname)
    ckpt = tf.train.latest_checkpoint(save_dir)
    model = create_model(args)
    model.load_weights(ckpt)
    model.set_entropy_model()
    if args.input_file.endswith('.npy'):
        x = np.load(args.input_file)
        assert len(x.shape) == 3, "Needs to be a single [H,W,C] image for model.compress to work"
    else:
        x = read_png(args.input_file)
    tensors = model.compress(x)

    # Write a binary file with the shape information and the compressed string.
    packed = tfc.PackedTensors()
    packed.pack(tensors)
    with open(args.output_file, "wb") as f:
        f.write(packed.string)

    # If requested, decompress the image and measure performance.
    if args.verbose:
        x_hat = model.decompress(*tensors)

        # Cast to float in order to compute metrics.
        x = tf.cast(x, tf.float32)
        x_hat = tf.cast(x_hat, tf.float32)
        mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
        psnr = tf.squeeze(tf.image.psnr(x, x_hat, 255))
        im_size = tf.shape(x)[:-1]
        if im_size[0] < 160 and im_size[1] < 160:  # hack to avoid tf.image.ssim_multiscale crashing on smaller imgs
            msssim = tf.squeeze(tf.image.ssim(x, x_hat, 255))
        else:
            msssim = tf.squeeze(tf.image.ssim_multiscale(x, x_hat, 255))

        msssim_db = -10. * tf.math.log(1 - msssim) / tf.math.log(10.)

        # The actual bits per pixel including entropy coding overhead.
        num_pixels = tf.reduce_prod(tf.shape(x)[:-1])
        bpp = len(packed.string) * 8 / num_pixels

        print(f"Mean squared error: {mse:0.4f}")
        print(f"PSNR (dB): {psnr:0.2f}")
        print(f"Multiscale SSIM: {msssim:0.4f}")
        print(f"Multiscale SSIM (dB): {msssim_db:0.2f}")
        print(f"Bits per pixel: {bpp:0.4f}")


def decompress(args, create_model, get_runname):
    """Decompresses an image."""
    # Load the model and determine the dtypes of tensors required to decompress.
    runname = get_runname(args)
    save_dir = os.path.join(args.checkpoint_dir, runname)
    ckpt = tf.train.latest_checkpoint(save_dir)
    model = create_model(args)
    model.load_weights(ckpt)
    model.set_entropy_model()

    dtypes = [t.dtype for t in model.decompress.input_signature]

    # Read the shape information and compressed string from the binary file,
    # and decompress the image using the model.
    with open(args.input_file, "rb") as f:
        packed = tfc.PackedTensors(f.read())
    tensors = packed.unpack(dtypes)
    x_hat = model.decompress(*tensors)

    # Write reconstructed image out as a PNG file.
    # This seamlessly handles both grayscale and color images (i.e., x_hat
    # having either 1 or 3 channels).
    write_png(args.output_file, x_hat)


def main(args, create_model, get_runname):
    # Invoke subcommand.
    seed = args.seed
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if args.command == "train":
        train(args, create_model, get_runname)
    elif args.command == "eval":
        evaluate(args, create_model, get_runname)
    elif args.command == "compress":
        if not args.output_file:
            args.output_file = args.input_file + ".tfci"
        compress(args, create_model, get_runname)
    elif args.command == "decompress":
        if not args.output_file:
            args.output_file = args.input_file + ".png"
        decompress(args, create_model, get_runname)
