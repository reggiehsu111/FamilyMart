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
        batch_size=1024,
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
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    with open(Path(config['data_loader']['args']['data_dir'])
                   / 'test_mean.pkl', 'rb') as file:
        mean = pickle.load(file)
    with open(Path(config['data_loader']['args']['data_dir'])
                   / 'test_std.pkl', 'rb') as file:
        std = pickle.load(file)

    with open(Path(config['data_loader']['args']['data_dir'])
                   / 'train_mean.pkl', 'rb') as file:
        train_mean = pickle.load(file)
    with open(Path(config['data_loader']['args']['data_dir'])
                   / 'train_std.pkl', 'rb') as file:
        train_std = pickle.load(file)

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    num_commodity = len(data_loader.dataset) // 140

    hist_pred = [ [] for i in range(num_commodity) ]
    hist_real = [ [] for i in range(num_commodity) ]
    idx_counter = 0

    with torch.no_grad():
        # for i, (data, target, target_2) in enumerate(tqdm(data_loader)):
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            
            output = model(data)

            output_1 = torch.round(output * std + mean)

            # output_1_pad = torch.cat([output_1, data[:, -5:, 0]], 1)

            # output_2 = model(torch.cat([data[:, :, 1:], output_1_pad.unsqueeze(2)], 2))

            for j,num in enumerate(output_1):
                # print(num, int(num), int(target[j] * std + mean))
                # input()
                hist_pred[ idx_counter ].append(num.numpy()[0])
                # hist_pred.append(output * std + mean)
                num2 = target[j] * std + mean
                hist_real[ idx_counter ].append(num2.numpy()[0])
                idx_counter += 1
                if idx_counter == num_commodity:
                    idx_counter = 0

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            total_loss += loss.item()
            for i, metric in enumerate(metric_fns):
                # total_metrics[i] += metric(output_2, target_2, mean, std)
                total_metrics[i] += metric(output, target, mean, std)

    plt.xticks(
        range(0, 140, 10),
        [get_formatted_time(i) for i in range(2 * 365 - 142, 2 * 365 - 142 + 140, 10)],
        rotation=30,
        size=8
    )

    n_samples = len(data_loader.dataset)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)

    diff = 0
    summation = 0
    for i in range(num_commodity):
        for j in range(140):
            diff += abs(hist_pred[i][j] - hist_real[i][j])
            summation += hist_real[i][j]
    print(diff / summation)
    for i in range(num_commodity):
        
        diff = 0
        summation = 0
        if len(hist_pred[i]) != 140:
            print("error!")
            break
        for j in range(140):
            diff += abs(hist_pred[i][j] - hist_real[i][j])
            summation += hist_real[i][j]
        # print(i, diff, summation, sep=",")
        # plt.scatter(range(0, 140), hist_pred[i], label="pred", marker='s')
        # plt.scatter(range(0, 140), hist_real[i], label="true", marker='o' )
        # plt.legend()
        # fig = plt.gcf()
        # fig.set_size_inches(18.5, 10.5)
        # plt.savefig('saved/vis/0.png', dpi=100)
        # plt.show()
        # print(hist_pred[0, :100], hist_real[0, :100])


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser(args)
    main(config)