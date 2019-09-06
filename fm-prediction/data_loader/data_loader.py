import pickle
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset

from base import BaseDataLoader

class FamilyMartDataset(Dataset):
    def __init__(self, data_dir, train):
        # Window size is the number of consecutive days before the target
        # day to choose for the model to predict the sales of the target
        # day. i.e., the target day is not included in window size.
        self._time_window_size = 20

        with open(data_dir / 'sales_data.pkl', 'rb') as file:
            self._sales_data = pickle.load(file)
        with open(data_dir / 'commodity_codes.pkl', 'rb') as file:
            self._commodity_codes = pickle.load(file)
        with open(data_dir / 'store_codes.pkl', 'rb') as file:
            self._store_codes = pickle.load(file)

        self._train = train
    
    def __len__(self):
        # return (2 * 365 - self._time_window_size) * len(self._store_codes)
        if self._train is True:
            return (2 * 365 - self._time_window_size - 142) * len(self._store_codes)
        else:  # Testing
            return 142 * len(self._store_codes)

    def __getitem__(self, idx):
        if self._train is False:  # Testing
            idx += (2 * 365 - self._time_window_size - 142) * len(self._store_codes)

        interval = len(self._store_codes) * len(self._commodity_codes)

        x_index = idx * len(self._commodity_codes)

        x_indices = x_index + torch.tensor(
            [range(i, self._time_window_size * interval, interval)
             for i in range(len(self._commodity_codes))])

        store_code_onehot = torch.zeros(len(self._store_codes), self._time_window_size)
        store_code_onehot[idx % len(self._store_codes)] = 1

        x = torch.cat([self._sales_data[x_indices], store_code_onehot], 0)

        y_index = (idx + self._time_window_size) * len(self._commodity_codes)

        y_indices = y_index + torch.tensor(
            [i for i in range(len(self._commodity_codes))])
        
        y = self._sales_data[y_indices]

        return x, y

class FamilyMartDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, train=True,
                 validation_split_ratio=0.0, num_workers=1):

        dataset = FamilyMartDataset(Path(data_dir), train)

        super().__init__(dataset, batch_size, shuffle, validation_split_ratio,
                         num_workers)
