import glob
import os
import sys

import tensorflow.compat.v1 as tf

from utils import read_png, read_npy_file_helper, get_runname


def train(args, build_train_graph):
    """Trains the model."""

    if args.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)
    else:
        tf.logging.set_verbosity(tf.logging.ERROR)

    # Create input data pipeline.
    with tf.device("/cpu:0"):
        train_files = glob.glob(args.train_glob)
        if not train_files:
            raise RuntimeError(
                "No training images found with glob '{}'.".format(args.train_glob))
        train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
        train_dataset = train_dataset.shuffle(buffer_size=len(train_files)).repeat()
        if 'npy' in args.train_glob:  # reading numpy arrays directly instead of from images
            train_dataset = train_dataset.map(  # https://stackoverflow.com/a/49459838
                lambda item: tuple(tf.numpy_function(read_npy_file_helper, [item], [tf.float32, ])),
                num_parallel_calls=args.preprocess_threads)
        else:
            train_dataset = train_dataset.map(
                read_png, num_parallel_calls=args.preprocess_threads)
        train_dataset = train_dataset.map(lambda x: tf.random_crop(x, (args.patchsize, args.patchsize, 3)))
        train_dataset = train_dataset.batch(args.batchsize)
        train_dataset = train_dataset.prefetch(32)

    # num_pixels = args.batchsize * args.patchsize ** 2

    # Get training patch from dataset.
    x = train_dataset.make_one_shot_iterator().get_next()
    res = build_train_graph(args, x)
    train_loss = res['train_loss']
    train_op = res['train_op']
    model_name = res['model_name']

    # boiler plate code for logging
    runname = get_runname(vars(args), record_keys=('num_filters', 'num_hfilters', 'lmbda'), prefix=model_name)
    save_dir = os.path.join(args.checkpoint_dir, runname)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    import json
    import datetime
    with open(os.path.join(save_dir, 'record.txt'), 'a') as f:  # keep more detailed record in text file
        f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n')
        f.write(json.dumps(vars(args), indent=4, sort_keys=True) + '\n')
        f.write('\n')
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:  # will overwrite existing
        json.dump(vars(args), f, indent=4, sort_keys=True)

    # save a copy of the script that defined the model
    from shutil import copy
    copied_path = copy(model_name + '.py', save_dir)
    print('Saved a copy of %s.py to %s' % (model_name, copied_path))

    hooks = [
        tf.train.StopAtStepHook(last_step=args.last_step),
        tf.train.NanTensorHook(train_loss),
    ]

    save_summary_secs = args.save_summary_secs
    if args.logdir != '':
        for key in res:
            if 'bpp' in key or 'loss' in key or key in ('mse', 'psnr'):
                tf.summary.scalar(key, res[key])
            elif key in ('original', 'reconstruction'):
                tf.summary.image(key, res[key], max_outputs=2)

        summary_op = tf.summary.merge_all()
        tf_log_dir = os.path.join(args.logdir, runname)
        summary_hook = tf.train.SummarySaverHook(save_secs=save_summary_secs, output_dir=tf_log_dir,
                                                 summary_op=summary_op)
        hooks.append(summary_hook)

    with tf.train.MonitoredTrainingSession(
            hooks=hooks, checkpoint_dir=save_dir,
            save_checkpoint_secs=args.save_checkpoint_secs, save_summaries_secs=save_summary_secs) as sess:
        while not sess.should_stop():
            sess.run(train_op)


def parse_args(argv):
    """Parses command line arguments."""
    import argparse
    # from absl import app
    from absl.flags import argparse_flags

    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # High-level options.
    parser.add_argument(
        "--verbose", "-V", action="store_true",
        help="Report bitrate and distortion when training or compressing.")
    parser.add_argument(
        "--num_filters", type=int, default=-1,
        help="Number of filters in the latents.")
    parser.add_argument(
        "--num_hfilters", type=int, default=-1,
        help="Number of filters in the hyper latents.")
    parser.add_argument(
        "--checkpoint_dir", default="./checkpoints",
        help="Directory where to save/load model checkpoints.")
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
        description="Trains (or continues to train) a new model.")
    train_cmd.add_argument(
        "--train_glob", default="images/*.png",
        help="Glob pattern identifying training data. This pattern must expand "
             "to a list of RGB images in PNG format.")
    train_cmd.add_argument(
        "--batchsize", type=int, default=8,
        help="Batch size for training.")
    train_cmd.add_argument(
        "--patchsize", type=int, default=256,
        help="Size of image patches for training.")
    train_cmd.add_argument(
        "--lambda", type=float, default=0.01, dest="lmbda",
        help="Lambda for rate-distortion tradeoff.")
    train_cmd.add_argument(
        "--last_step", type=int, default=1000000,
        help="Train up to this number of steps.")
    train_cmd.add_argument(
        "--preprocess_threads", type=int, default=16,
        help="Number of CPU threads to use for parallel decoding of training "
             "images.")
    train_cmd.add_argument(
        "--logdir", default="/tmp/tf_logs",  # '--log_dir' seems to conflict with absl.flags's existing
        help="Directory for storing Tensorboard logging files; set to empty string '' to disable Tensorboard logging.")
    train_cmd.add_argument(
        "--save_checkpoint_secs", type=int, default=300,
        help="Seconds elapsed b/w saving models.")
    train_cmd.add_argument(
        "--save_summary_secs", type=int, default=60,
        help="Seconds elapsed b/w saving tf summaries.")

    # 'compress' subcommand.
    compress_cmd = subparsers.add_parser(
        "compress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads a PNG file, compresses it, and writes a TFCI file.")
    compress_cmd.add_argument(
        "--results_dir", default="./results",
        help="Directory for storing compression stats/results; set to empty string '' to disable.")
    compress_cmd.add_argument(
        "--lambda", type=float, default=-1, dest="lmbda",
        help="Lambda for rate-distortion tradeoff.")
    compress_cmd.add_argument(
        "--sga_its", type=int, default=2000,
        help="Number of SGA (Stochastic Gumbel Annealing) iterations .")
    compress_cmd.add_argument(
        "--annealing_rate", type=float, default=1e-3,
        help="Annealing rate for SGA.")
    compress_cmd.add_argument(
        "--t0", type=int, default=700,
        help="Number of 'soft-quantization' optimization iterations before annealing in SGA.")
    compress_cmd.add_argument(
        "--save_latents", action="store_true",
        help="Save the optimized latent variables (or variational parameters) to a `.npz` file.")


    # 'decompress' subcommand.
    decompress_cmd = subparsers.add_parser(
        "decompress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads a TFCI file, reconstructs the image, and writes back "
                    "a PNG file.")

    # 'encode_latents' subcommand.
    encode_latents_cmd = subparsers.add_parser(
        "encode_latents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Entropy code latent variables previously generated with `compress --save_latents` and write to a file.")
    encode_latents_cmd.add_argument(
        "--separate", action="store_true",
        help="Compress each batch item into an independent file.")

    # 'decode_latents' subcommand.
    decode_latents_cmd = subparsers.add_parser(
        "decode_latents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Decode latent variables previously encoded with `encode_latents`.")


    # Arguments for 'compress', 'decompress', 'encode_latents', and 'decode_latents'.
    for cmd, ext in ((compress_cmd, ".tfci"), (decompress_cmd, ".png"), (encode_latents_cmd, ".compressed"), (decode_latents_cmd, ".reconstructed.npz")):
        cmd.add_argument(
            "runname",
            help="Model name identifier constructed from run config, like 'bmshj2018-num_filters=...'"
        )
        cmd.add_argument(
            "input_file",
            help="Input filename.")
        cmd.add_argument(
            "output_file", nargs="?",
            help="Output filename (optional). If not provided, appends '{}' to "
                 "the input filename.".format(ext))

    # Parse arguments.
    args = parser.parse_args(argv[1:])
    if args.command is None:
        parser.print_usage()
        sys.exit(2)
    return args
