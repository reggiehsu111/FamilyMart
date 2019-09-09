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
        print(self._sales_data.shape)

        self._train = train
    
    def __len__(self):
        # return (2 * 365 - self._time_window_size) * len(self._store_codes)
        if self._train is True:
            return (2 * 365 - self._time_window_size - 142 - 2) * len(self._store_codes)
        else:  # Testing
            return 142 * len(self._store_codes)

    def __getitem__(self, idx):
        if self._train is False:  # Testing
            idx += (2 * 365 - self._time_window_size - 142 - 2) * len(self._store_codes)

        # store_code_onehot is of shape (5, 20)
        store_code_onehot = torch.zeros(len(self._store_codes), self._time_window_size)
        store_code_onehot[idx % len(self._store_codes)] = 1
        time_idx = int(idx/len(self._store_codes))
        store_idx = idx%len(self._store_codes)


        x_idx = time_idx + torch.tensor([i for i in range(self._time_window_size)])
        x = torch.cat((self._sales_data[x_idx, store_idx].transpose(0,1), store_code_onehot), 0)

        y1 = self._sales_data[time_idx + self._time_window_size + 1, store_idx]

        y2 = self._sales_data[time_idx + self._time_window_size + 2, store_idx]
        return x, y1, y2

class FamilyMartDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, train=True,
                 validation_split_ratio=0.0, num_workers=1):

        dataset = FamilyMartDataset(Path(data_dir), train)

        super().__init__(dataset, batch_size, shuffle, validation_split_ratio,
                         num_workers)
