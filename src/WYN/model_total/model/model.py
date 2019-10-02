import torch
import torch.nn as nn
from base import BaseModel

STORE_DIM = [5, 2]
COMMIDITY_DIM = [759, 4]
PING_DIM = [12, 2]
CHING_DIM = [37, 4]
DAY_IN_YEAR_DIM = [365, 2]
DAY_IN_WEEK_DIM = [7, 4]
WINDOW_SIZE = 20


class FCModel(BaseModel):
    """
    Fully-connected Model.
    """
    def __init__(self):
        super().__init__()

        self._n_commodities = 759
        self._n_stores = 5

        '''
        NUM_STORE(5) + NUM_COMMODITY(759) + NUM_PING(12) + NUM_CHING(37) + day_in_year_onehot(365) + day_in_week(7) + sales_data(20)
        '''

        self.all_net = nn.Sequential(

            nn.Linear( STORE_DIM[1] + COMMIDITY_DIM[1] + PING_DIM[1] + CHING_DIM[1] + DAY_IN_YEAR_DIM[1] + DAY_IN_WEEK_DIM[1] + WINDOW_SIZE, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 1),
        )

        self.store_net = nn.Sequential(
            nn.Linear(STORE_DIM[0], STORE_DIM[1]),
            nn.ReLU()
        )

        self.commodity_net = nn.Sequential(
            nn.Linear(COMMIDITY_DIM[0], COMMIDITY_DIM[1]),
            nn.ReLU()
        )

        self.ping_net = nn.Sequential(
            nn.Linear(PING_DIM[0], PING_DIM[1]),
            nn.ReLU()
        )

        self.ching_net = nn.Sequential(
            nn.Linear(CHING_DIM[0], CHING_DIM[1]),
            nn.ReLU()
        )

        self.day_year_net = nn.Sequential(
            nn.Linear(DAY_IN_YEAR_DIM[0], DAY_IN_YEAR_DIM[1]),
            nn.ReLU()
        )

        self.day_week_net = nn.Sequential(
            nn.Linear(DAY_IN_WEEK_DIM[0], DAY_IN_WEEK_DIM[1]),
            nn.ReLU()
        )
        

    def forward(self, x):

        '''
        x = batch_size * (NUM_STORE(5) + NUM_COMMODITY(759) + NUM_PING(12) + NUM_CHING(37) + day_in_year_onehot(365) + day_in_week(7) + sales_data(20))
        '''

        sn = self.store_net(x[:, :STORE_DIM[0]])
        cn = self.commodity_net(x[:, STORE_DIM[0]:STORE_DIM[0] + COMMIDITY_DIM[0]])
        pin = self.ping_net(x[:, STORE_DIM[0] + COMMIDITY_DIM[0]:STORE_DIM[0] + COMMIDITY_DIM[0] + PING_DIM[0]])
        cin = self.ching_net(x[:, STORE_DIM[0] + COMMIDITY_DIM[0] + PING_DIM[0]:STORE_DIM[0] + COMMIDITY_DIM[0] + PING_DIM[0] + CHING_DIM[0]])
        dyn = self.day_year_net(x[:, STORE_DIM[0] + COMMIDITY_DIM[0] + PING_DIM[0] + CHING_DIM[0]:STORE_DIM[0] + COMMIDITY_DIM[0] + PING_DIM[0] + CHING_DIM[0] + DAY_IN_YEAR_DIM[0]])
        dwn = self.day_week_net(x[:, STORE_DIM[0] + COMMIDITY_DIM[0] + PING_DIM[0] + CHING_DIM[0] + DAY_IN_YEAR_DIM[0]:STORE_DIM[0] + COMMIDITY_DIM[0] + PING_DIM[0] + CHING_DIM[0] + DAY_IN_YEAR_DIM[0] + DAY_IN_WEEK_DIM[0]])
        
        y = self.all_net( torch.cat((sn, cn, pin, cin, dyn, dwn, x[:, -WINDOW_SIZE:]), dim=1) )
        return y
