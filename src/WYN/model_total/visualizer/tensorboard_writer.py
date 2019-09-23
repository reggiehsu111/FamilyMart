
from utils import Timer


class TensorboardWriter:
    def __init__(self, log_dir, logger, enabled):
        if enabled:
            log_dir = str(log_dir)

            # Setup tensorboard writer
            try:
                from torch.utils import tensorboard
            except ImportError:
                message = (
                    'Warning: Tensorboard is configured to use, but currently '
                    'not installed. Please upgrade PyTorch to version >= 1.1 '
                    'for using `torch.utils.tensorboard` or turn off the option '
                    'in the `config.json` file.'
                )
                logger.warning(message)
            else:
                self._writer = tensorboard.SummaryWriter(log_dir)

        self._step = 0
        self._mode = ''

        self._tag_mode_exceptions = {'add_histogram', 'add_embedding'}
            
        self._timer = Timer()

    def set_step(self, step, mode='train'):
        self._step = step
        self._mode = mode

        if step == 0:
            self._timer.reset()
        else:
            duration = self._timer.check()
            self.add_scalar('steps_per_sec', 1 / duration)

    def __getattr__(self, name):
        """
        If tensorboard is configured to use, return methods of tensorboard
        with additional information (mode, step) added.
        Otherwise, return a blank function.
        """
        add_data = getattr(self._writer, name, None)

        def wrapper(tag, data, *args, **kwargs):
            if add_data is not None:
                if name not in self._tag_mode_exceptions:
                    tag = '{}/{}'.format(tag, self._mode)
                add_data(tag, data, self._step, *args, **kwargs)
        
        return wrapper
