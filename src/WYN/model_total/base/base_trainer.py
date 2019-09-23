import torch
from abc import ABC, abstractmethod
from visualizer import TensorboardWriter


class BaseTrainer(ABC):
    """
    Base class for all trainers.
    """
    def __init__(self, model, loss, metrics, optimizer, config):
        self._config = config

        self._logger = config.get_logger(
            'trainer', config['trainer']['verbosity'])

        # Setup GPU for training if available
        self._device, gpu_ids = self._get_device(config['n_gpu'])
        self._model = model.to(self._device)
        if len(gpu_ids) > 1:
            self._model = torch.nn.DataParallel(model, device_ids=gpu_ids)

        self._loss = loss
        self._metrics = metrics
        self._optimizer = optimizer

        cfg_trainer = config['trainer']
        self._epochs = cfg_trainer['epochs']
        self._save_period = cfg_trainer['save_period']
        self._monitor = cfg_trainer.get('monitor', 'off')

        # Config for monitoring model performance and saving the best model
        if self._monitor == 'off':
            self._mnt_mode = 'off'
            self._mnt_best = 0
        else:
            self._mnt_mode, self._mnt_metric = self._monitor.split()

            assert self._mnt_mode in ['min', 'max']

            self._mnt_best = (float('inf') if self._mnt_mode == 'min'
                              else -float('inf'))
            self._early_stop = cfg_trainer.get('early_stop', float('inf'))

        self._start_epoch = 1

        self._checkpoint_dir = config.save_dir

        # Setup visualization writer instance
        self._writer = TensorboardWriter(config.log_dir, self._logger,
                                         cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch.
        Args:
            epoch: Current epoch number.
        Return:
            A log dict that contains all information you want to save.
        Note:
            1.  If you have additional information to record, for example:
                > additional_log = {'x': x, 'y': y}
                merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log
            2.  The metrics in log must have the key 'metrics'.
        """
        return NotImplemented

    def train(self):
        """
        Full training logic.
        """
        not_improved_count = 0
        for epoch in range(self._start_epoch, self._epochs + 1):
            raw_log = self._train_epoch(epoch)

            # Extract metrics and save results into dict
            log = {'epoch': epoch}
            for key, value in raw_log.items():
                if key == 'metrics':
                    log.update({metric.__name__: value[i]
                                for i, metric in enumerate(self._metrics)})
                elif key == 'val_metrics':
                    log.update({'val_{}'.format(metric.__name__): value[i]
                                for i, metric in enumerate(self._metrics)})
                else:
                    log[key] = value

            # Print logged informations
            for key, value in log.items():
                self._logger.info('    {:15s}: {}'.format(str(key), value))

            # Evaluate model performance according to configured metric,
            # and save best checkpoint as model_best
            best = False
            if self._mnt_mode != 'off':
                try:
                    # Check whether the performance of model is improved
                    # or not by specified metric
                    if self._mnt_mode == 'min':
                        improved = (
                            True if log[self._mnt_metric] <= self._mnt_best
                            else False
                        )
                    else:  # self._mnt_mode == 'max'
                        improved = (
                            True if log[self._mnt_metric] >= self._mnt_best
                            else False
                        )
                except KeyError:
                    self._logger.warning(
                        "Warning: Metric '{}' not found. "
                        "Model performance monitoring is disabled."
                        .format(self._mnt_metric)
                    )
                    self._mnt_mode = 'off'
                    improved = False

                if improved:
                    self._mnt_best = log[self._mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count >= self._early_stop:
                    self._logger.info(
                        "Validation performance didn't improve for {} "
                        "epochs. Training stops."
                        .format(not_improved_count)
                    )
                    break

            if epoch % self._save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best)

    def _get_device(self, n_gpu_use):
        """
        Check the number of GPUs available and return the device and GPU list
        to use.
        """
        n_gpu_avail = torch.cuda.device_count()

        if n_gpu_use > 0 and n_gpu_avail == 0:
            self._logger.warning('Warning: No GPU available, using CPU.')
            n_gpu_use = 0
        elif n_gpu_use > n_gpu_avail:
            self._logger.warning('Warning: The number of GPUs configured to '
                                 'use is {}, but only {} available.'
                                 .format(n_gpu_use, n_gpu_avail))
            n_gpu_use = n_gpu_avail

        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        gpu_ids = list(range(n_gpu_use))

        return device, gpu_ids

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints.
        """
        state = {
            'model': type(self._model).__name__,
            'epoch': epoch,
            'state_dict': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'monitor_best': self._mnt_best,
            'config': self._config
        }

        if epoch % self._save_period == 0:
            ckeckpoint_path = str(
                self._checkpoint_dir / 'ckeckpoint_epoch{}.pth'.format(epoch))
            torch.save(state, ckeckpoint_path)

            self._logger.info('Saving checkpoint to {} ...'
                            .format(ckeckpoint_path))

        if save_best:
            best_path = str(self._checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self._logger.info('Saving current best to {} ...'
                              .format(best_path))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints.
        """
        resume_path = str(resume_path)

        self._logger.info('Loading checkpoint from {} ...'.format(resume_path))
        checkpoint = torch.load(resume_path)

        self._start_epoch = checkpoint['epoch'] + 1
        self._mnt_best = checkpoint['monitor_best']

        # Load model state from checkpoint
        if checkpoint['config']['model'] != self._config['model']:
            self._logger.warning(
                'Warning: Model configuration given in config is different '
                'from that of the checkpoint. This may yield an exception '
                'while loading model state.'
            )

        self._model.load_state_dict(checkpoint['state_dict'])

        # Load optimizer state from checkpoint only when optimizer type is
        # not changed
        if (checkpoint['config']['optimizer']['type']
                != self._config['optimizer']['type']):
            self._logger.warning(
                'Warning: Optimizer is not resumed, since optimizer type '
                'given in config is different from that of the checkpoint.'
            )
        else:
            self._optimizer.load_state_dict(checkpoint['optimizer'])

        self._logger.info('Checkpoint loaded. Resume training from epoch {}'
                          .format(self._start_epoch))
