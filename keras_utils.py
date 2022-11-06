# keras hacks to make life easier.
# Yibo Yang, 2021

from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging
import os
import glob

from keras.callbacks import ModelCheckpoint


class MyModelCheckpointCallback(ModelCheckpoint):
    """
    My fancier version of keras.callbacks.ModelCheckpoint that saves a maximum number of checkpoint files.
    """

    def __init__(self, filepath, max_to_keep, monitor='val_loss', verbose=0, save_best_only=False,
                 save_weights_only=False,
                 mode='auto', save_freq='epoch', options=None, **kwargs):
        super().__init__(filepath=filepath, monitor=monitor, verbose=verbose, save_best_only=save_best_only,
                         save_weights_only=save_weights_only, mode=mode, save_freq=save_freq, options=options,
                         **kwargs)
        assert max_to_keep >= 1
        self.max_to_keep = max_to_keep
        self.most_recent_checkpoint_names = []

    def _save_model(self, epoch, logs):
        """Saves the model. Maintains max_to_keep many checkpoints by deleting old ones once it has saved more than
        max_to_keep many checkpoints; otherwise identical to the base class method.

        Args:
            epoch: the epoch this iteration is in.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}

        save_successful = False
        if isinstance(self.save_freq,
                      int) or self.epochs_since_last_save >= self.period:
            # Block only when saving interval is reached.
            logs = tf_utils.sync_to_numpy_or_python_type(logs)
            self.epochs_since_last_save = 0
            filepath = self._get_file_path(epoch, logs)

            try:
                if self.save_best_only:
                    current = logs.get(self.monitor)
                    if current is None:
                        logging.warning('Can save best model only with %s available, '
                                        'skipping.', self.monitor)
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                      ' saving model to %s' % (epoch + 1, self.monitor,
                                                               self.best, current, filepath))
                            self.best = current
                            if self.save_weights_only:
                                self.model.save_weights(
                                    filepath, overwrite=True, options=self._options)
                            else:
                                self.model.save(filepath, overwrite=True, options=self._options)
                            save_successful = True
                        else:
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s did not improve from %0.5f' %
                                      (epoch + 1, self.monitor, self.best))
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                    if self.save_weights_only:
                        self.model.save_weights(
                            filepath, overwrite=True, options=self._options)
                    else:
                        self.model.save(filepath, overwrite=True, options=self._options)
                    save_successful = True

                self._maybe_remove_file()
            except IOError as e:
                # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
                if 'is a directory' in str(e.args[0]).lower():
                    raise IOError('Please specify a non-directory filepath for '
                                  'ModelCheckpoint. Filepath used is an existing '
                                  'directory: {}'.format(filepath))
                # Re-throw the error for any other causes.
                raise e
        if save_successful:
            self.most_recent_checkpoint_names.append(filepath)
            if len(self.most_recent_checkpoint_names) > self.max_to_keep:
                assert len(self.most_recent_checkpoint_names) \
                       == self.max_to_keep + 1, 'should have deleted as soon as over the limit'
                checkpoint_name_to_remove = self.most_recent_checkpoint_names.pop(0)
                checkpoint_files_to_remove = glob.glob(
                    checkpoint_name_to_remove + '.*')  # filepath.index, filepath.data-00000-of-00001, etc.
                for checkpoint_file in checkpoint_files_to_remove:
                    os.remove(checkpoint_file)
                if self.verbose > 0:
                    print("Removed old checkpoint files", checkpoint_files_to_remove)


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
