import torch
import torch.nn as nn
from base import BaseModel


class FCModel(BaseModel):
    """
    Fully-connected Model.
    """
    def __init__(self):
        super().__init__()

        # self._n_commodities = 759
        # pinfan
        self._n_commodities = 12
        # qunfan
        # self._n_commodities = 37
        # holiday
        self._n_stores = 5

        self.sales_embedding_net = nn.Sequential(
            # nn.Linear(self._n_commodities * 20, 256),
            # holiday
            # nn.Linear(self._n_commodities * 40, 256),
            # holiday one
            nn.Linear(self._n_commodities * 21, 256),

            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )

        self.store_embedding_net = nn.Sequential(
            nn.Linear(self._n_stores, 1),
            nn.BatchNorm1d(1),
            nn.ReLU(),
        )

        self.fc_net = nn.ModuleList([nn.Sequential(
            # nn.Linear(20 + 1 + 16, 64),
            # holiday
            # nn.Linear(40 + 1 + 16, 64),
            # holiday
            nn.Linear(21 + 1 + 16, 64),

            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )] * self._n_commodities)

    def forward(self, x):
        a = x
        
        b = self.store_embedding_net(x[:, -self._n_stores:, 0])
        # print(x.shape)
        # print(x[:, -self._n_stores:, 0].shape)
        # print(b.shape)
        # print(x[:, :-self._n_stores].view(x.size(0), -1).shape)
        # print(x[:, :-self._n_stores].shape)
        # exit()
        c = self.sales_embedding_net(x[:, :-self._n_stores].view(x.size(0), -1))
        
        y = torch.cat([
            self.fc_net[i](torch.cat([a[:, i], b, c], 1))
            for i in range(self._n_commodities)
        ], 1)

        return y
