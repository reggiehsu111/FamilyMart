import torch
import pickle
import math
from pathlib import Path
from torchvision.utils import make_grid

from base import BaseTrainer
from utils import inf_loop


class Trainer(BaseTrainer):
    """
    Trainer example.
    """
    def __init__(self, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, loss, metrics, optimizer, config)

        if len_epoch is None:
            # Epoch-based training
            self._data_loader = data_loader
            self._len_epoch = len(self._data_loader)
        else:
            # Iteration-based training
            self._data_loader = inf_loop(data_loader)
            self._len_epoch = len_epoch

        self._valid_data_loader = valid_data_loader
        self._do_validation = self._valid_data_loader is not None

        self._lr_scheduler = lr_scheduler
        self._log_step = int(math.sqrt(self._len_epoch))

        with open(Path(config['data_loader']['args']['data_dir'])
                  / 'sales_mean.pkl', 'rb') as file:
            self._mean = pickle.load(file)
        with open(Path(config['data_loader']['args']['data_dir'])
                  / 'sales_std.pkl', 'rb') as file:
            self._std = pickle.load(file)

    def _train_epoch(self, epoch):
        self._model.train()

        total_loss_1 = 0
        total_loss_2 = 0
        total_metrics_1 = torch.zeros(len(self._metrics))
        total_metrics_2 = torch.zeros(len(self._metrics))
        for batch_idx, (data, target1, target2) in enumerate(self._data_loader):
            data, target1, target2 = data.to(self._device), target1.to(self._device), target2.to(self._device)
            self._optimizer.zero_grad()
            
            output1 = self._model(data)
            loss1 = self._loss(output1, target1)
            loss1.backward()
            temp_data = torch.cat((data[:,:759,1:], output1.unsqueeze(-1)), 2)
            temp_data = torch.cat((temp_data, data[:,759:, :]), 1)
            output2 = self._model(temp_data)

            loss2 = self._loss(output2, target2)

            self._optimizer.step()

            total_loss_1 += loss1.item()
            total_loss_2 += loss2.item()
            total_metrics_1 += self._eval_metrics(output1, target1)
            total_metrics_2 += self._eval_metrics(output2, target2)

            # self._writer.set_step((epoch - 1) * self._len_epoch + batch_idx)
            # self._writer.add_scalar('loss', loss.item())

            if batch_idx % self._log_step == 0:
                self._logger.debug('{} Loss1: {:.6f}'.format(self._progress(batch_idx),loss1.item() 
                    / len(data)))
                # self._writer.add_image('input', make_grid(data.cpu(), nrow=8,
                #                        normalize=True))

            if batch_idx == self._len_epoch:
                break

        log = {
            'loss1': total_loss_1 / len(self._data_loader.dataset),
            'loss2': total_loss_2 / len(self._data_loader.dataset),
            'err1' : (total_metrics_1 / len(self._data_loader.dataset)).item(),
            'err2' : (total_metrics_2 / len(self._data_loader.dataset)).item(),
            'metrics': (total_metrics_1 / len(self._data_loader.dataset)).tolist()
        }

        if self._do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)

        if self._lr_scheduler is not None:
            self._lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :return: A log that contains information about validation
        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self._model.eval()

        total_val_loss_1 = 0
        total_val_loss_2 = 0
        total_val_metrics_1 = torch.zeros(len(self._metrics))
        total_val_metrics_2 = torch.zeros(len(self._metrics))
        with torch.no_grad():
            for batch_idx, (data, target1, target2) in enumerate(self._valid_data_loader):
                data, target1, target2 = data.to(self._device), target1.to(self._device), target2.to(self._device)
            
                output1 = self._model(data)
                loss1 = self._loss(output1, target1)
                temp_data = torch.cat((data[:,:759,1:], output1.unsqueeze(-1)), 2)
                temp_data = torch.cat((temp_data, data[:,759:, :]), 1)
                output2 = self._model(temp_data)

                loss2 = self._loss(output2, target2)
                
                total_val_loss_1 += loss1.item()
                total_val_loss_2 += loss2.item()
                total_val_metrics_1 += self._eval_metrics(output1, target1)
                total_val_metrics_2  += self._eval_metrics(output2, target2)

                # self._writer.set_step(
                #     (epoch - 1) * len(self._valid_data_loader) + batch_idx,
                #     'valid'
                # )
                # self._writer.add_scalar('loss', loss.item())
                
                # self._writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        # for name, p in self._model.named_parameters():
            # self._writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss_1': total_val_loss_1 / len(self._valid_data_loader.dataset),
            'val_loss_2': total_val_loss_2 / len(self._valid_data_loader.dataset),
            'val_err_1': (total_val_metrics_1 / len(self._valid_data_loader.dataset)).item(),
            'val_err_2' : (total_val_metrics_2 / len(self._valid_data_loader.dataset)).item(),
            'val_metrics': (total_val_metrics_1 / len(self._valid_data_loader.dataset)).tolist()
        }

    def _eval_metrics(self, output, target):
        acc_metrics = torch.zeros(len(self._metrics))

        for i, metric in enumerate(self._metrics):
            acc_metrics[i] += metric(output, target, self._mean, self._std)
            # self._writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])

        return acc_metrics

    def _progress(self, batch_idx):
        base_str = '[{}/{} ({:.0f}%)]'

        try:
            current = batch_idx * self._data_loader.batch_size
            total = len(self._data_loader.dataset)
        except AttributeError:
            current = batch_idx
            total = self._len_epoch
            
        return base_str.format(current, total, 100 * current / total)
