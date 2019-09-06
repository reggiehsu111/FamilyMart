import os
import logging
from pathlib import Path
from datetime import datetime
from functools import reduce
from operator import getitem

from logger import setup_logging
from utils import read_json, write_json


class ConfigParser:
    def __init__(self, args, options='', timestamp=True):
        # Parse default and custom cli options
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        args = args.parse_args()

        if args.device:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.device

        if args.resume:
            self._resume = Path(args.resume)
            self._cfg_fname = self._resume.parent / 'config.json'
        else:
            msg_no_cfg = ("Configuration file needs to be specified. "
                          "(Add '-c <config file>')")
            assert args.config is not None, msg_no_cfg

            self._resume = None
            self._cfg_fname = Path(args.config)

        # Load config file and apply custom cli options
        self._config = self._load_n_update_config(options, args)

        # Set save_dir where trained model and log will be saved.
        save_dir = Path(self._config['trainer']['save_dir'])
        timestamp = (datetime.now().strftime(r'%m%d_%H%M%S')
                     if timestamp else '')

        exp_name = self._config['name']
        self._save_dir = save_dir / 'models' / exp_name / timestamp
        self._log_dir = save_dir / 'log' / exp_name / timestamp

        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Save updated config file to the checkpoint dir
        write_json(self._config, self._save_dir / 'config.json')

        # Configure logging module
        setup_logging(self._log_dir)
        self._log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    def __getitem__(self, name):
        return self._config[name]

    def initialize(self, name, module, *args, **kwargs):
        """
        Find the function or class by the name given as "type" in config,
        and return the instance initialized with corresponding keyword args
        given as "args" in config.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])

        msg_overwrite_cfg = ('Overwriting kwargs given in config file '
                             'is not allowed.')
        assert all([k not in module_args for k in kwargs]), msg_overwrite_cfg

        module_args.update(kwargs)

        return getattr(module, module_name)(*args, **module_args)

    def get_logger(self, name, level=2):
        msg_level_invalid = (
            'verbosity option {} is invalid. Valid options are {}.'
            .format(level, self._log_levels.keys()))
        assert level in self._log_levels, msg_level_invalid

        logger = logging.getLogger(name)

        logger.setLevel(self._log_levels[level])

        return logger

    # Set read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def resume(self):
        return self._resume

    # Helper functions
    def _load_n_update_config(self, options, args):
        config = read_json(self._cfg_fname)

        for opt in options:
            value = getattr(args, self._get_opt_name(opt.flags))
            if value is not None:
                self._set_val(config, opt.target, value)

        return config

    def _get_opt_name(self, flags):
        for flag in flags:
            if flag.startswith('--'):
                return flag.replace('--', '')

    def _set_val(self, tree, keys, value):
        """Set a value in a nested object in tree by a sequence of keys."""
        reduce(getitem, keys[:-1], tree)[keys[-1]] = value
