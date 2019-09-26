import pickle
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
import sys

from base import BaseDataLoader

'''
MISSING:
test data loader
'''

class FamilyMartDataset(Dataset):
    def __init__(self, data_dir, train):
        # Window size is the number of consecutive days before the target
        # day to choose for the model to predict the sales of the target
        # day. i.e., the target day is not included in window size.
        self._time_window_size = 20

        '''
        train_data [num_valid_commodity, 365*2-140]
        feature_data [num_valid_commodity, NUM_STORE + NUM_COMMODITY + NUM_PING + NUM_CHING (one-hot encoding)]
        '''
        self._train = train

        if self._train:
            with open(data_dir / 'train_data.pkl', 'rb') as file:
                self._sales_data = pickle.load(file)
                
        else:
            with open(data_dir / 'test_data.pkl', 'rb') as file:
                self._sales_data = pickle.load(file)

        with open(data_dir / 'feature_data.pkl', 'rb') as file:
                self._feature_data = pickle.load(file)

    def __len__(self):
        
        '''
        every window_size + 1 days as 1 item -> a commidity has its length-20 items
        '''

        return (self._sales_data.shape[1] - self._time_window_size ) * self._sales_data.shape[0]

    def __getitem__(self, idx):

        day_axis, commidity_axis = divmod(idx, self._sales_data.shape[0])

        days_window = self._sales_data[commidity_axis, day_axis:day_axis + self._time_window_size]
        features = self._feature_data[commidity_axis]
        day_in_year_onehot = torch.zeros(365)
        day_in_week_onehot = torch.zeros(7)
        
        if self._train:
            day_in_year_onehot[(day_axis+self._time_window_size) % 365] = 1
            day_in_week_onehot[(day_axis+self._time_window_size) % 7] = 1
        
        else:
            day_in_year_onehot[(day_axis+self._time_window_size + (2*365 - 140 - 20)) % 365] = 1
            day_in_week_onehot[(day_axis+self._time_window_size + (2*365 - 140 - 20)) % 7] = 1

        x = torch.cat((features, day_in_year_onehot, day_in_week_onehot, days_window), dim=0)

        y1 = self._sales_data[commidity_axis, day_axis + self._time_window_size] # need to make sure the index ????
        y1 = torch.unsqueeze(y1, 0)
        #y2 = self._sales_data[time_idx + self._time_window_size + 2, store_idx]
        return x, y1

class FamilyMartDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, train=True,
                 validation_split_ratio=0.0, num_workers=1):

        dataset = FamilyMartDataset(Path(data_dir), train)

        super().__init__(dataset, batch_size, shuffle, validation_split_ratio,
                         num_workers)
