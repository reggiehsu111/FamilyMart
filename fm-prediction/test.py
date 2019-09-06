import argparse
import torch
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

import data_loader.data_loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from config_parser import ConfigParser


def get_formatted_time(idx):
    days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    year = idx // 365

    days = idx % 365

    month = None
    for m, days_in_m in enumerate(days_in_months):
        if days < days_in_m:
            month = m + 1
            break

        days -= days_in_m

    date = days + 1

    return int('201{}{:02d}{:02d}'.format(year + 7, month, date))

def main(config):
    logger = config.get_logger('test')
    print(config['data_loader']['args']['data_dir'])
    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=128,
        shuffle=False,
        train=False,
        validation_split_ratio=0.0,
        num_workers=8
    )

    # build model architecture
    model = config.initialize('model', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    # if config['n_gpu'] > 1:
    #     model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    with open(Path(config['data_loader']['args']['data_dir'])
                   / 'sales_mean.pkl', 'rb') as file:
        mean = pickle.load(file)
    with open(Path(config['data_loader']['args']['data_dir'])
                   / 'sales_std.pkl', 'rb') as file:
        std = pickle.load(file)

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    hist_pred = []
    hist_real = []
    with torch.no_grad():
        for i, (data, target, _) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            hist_pred.append(torch.round(output * std + mean))
            # hist_pred.append(output * std + mean)
            hist_real.append(target * std + mean)

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            total_loss += loss.item()
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target, mean, std)

    hist_pred = torch.cat(hist_pred, 0)
    hist_real = torch.cat(hist_real, 0)

    plt.xticks(
        range(0, 140, 10),
        [get_formatted_time(i) for i in range(2 * 365 - 142, 2 * 365 - 142 + 140, 10)],
        rotation=30,
        size=8
    )
    plt.plot(range(0, 140), hist_pred[:140, 659])
    plt.plot(hist_real[:140, 659])
    plt.savefig('saved/vis/0.png')
    # print(hist_pred[0, :100], hist_real[0, :100])

    n_samples = len(data_loader.dataset)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser(args)
    main(config)