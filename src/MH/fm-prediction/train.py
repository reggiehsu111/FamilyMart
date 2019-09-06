import argparse
import torch
from collections import namedtuple

import data_loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_net
from config_parser import ConfigParser
from trainer import Trainer


def main(config):
    logger = config.get_logger('train')

    # Setup data_loader
    data_loader = config.initialize('data_loader', module_data)
    valid_data_loader = data_loader.get_validation_data_loader()

    # Setup model
    model = config.initialize('model', module_net)
    logger.info(model)

    # Get loss and metrics
    loss = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, metric) for metric in config['metrics']]

    # Build optimizer, learning rate scheduler. To disable scheduler, comment
    # all lines containing `lr_scheduler`
    optimizer = config.initialize('optimizer', torch.optim, model.parameters())
    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler,
                                     optimizer)

    trainer = Trainer(model, loss, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    # One of "--config" and "--resume" must be specified
    # Specifying "--resume" automatically finds the config file of the
    # checkpoint provided
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')

    # Replace the use of "CUDA_VISIBLE_DEVICES"
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # Custom cli options to modify configuration given in the config json file
    CustomArgs = namedtuple('CustomArgs', ['flags', 'type', 'target'])
    options = [
        CustomArgs(['--lr', '--learning_rate'], float,
                   ('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], int,
                   ('data_loader', 'args', 'batch_size'))
    ]

    config = ConfigParser(args, options)

    main(config)
