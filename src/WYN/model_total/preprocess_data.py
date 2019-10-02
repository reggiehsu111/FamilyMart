import pickle
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import numpy as np

NUM_STORE = 5
NUM_COMMODITY = 759
NUM_PING = 12
NUM_CHING = 37
two_years = 365*2

class preprocessor:
    def __init__(self):

        self.data_dir = Path('../data')

        sales_dataframes = []
        commodity_dataframe = pd.read_csv(self.data_dir / '商品主檔.txt', sep='\t')
        
        self.output_dir = Path("data/")
        self.store_codes = (commodity_dataframe.loc[:, '原始店號'].sort_values().unique())
        self.indexed_frame = commodity_dataframe.loc[:, ["商品代號", "原始店號", "品番", "群番"]].set_index(["商品代號", "原始店號"])

        for year in [2017, 2018]:
            sales_dataframes.append(
                pd.read_csv(
                    self.data_dir / '銷售數量{}.txt'.format(year), sep='\t'
                ).loc[:, ['原始店號', '日期', '商品代號', '銷售數量']]
            )

        self.sales_data = pd.concat(sales_dataframes, axis=0)
        self.commodity_codes = self.__get_valid_commodity_codes__(commodity_dataframe)

        self.sales_data = self.sales_data.groupby(['日期', '原始店號', '商品代號']).agg(["sum"])

    def generate(self):

        '''
        train_data [num_valid_commodity, 365*2-140]
        test_data [num_valid_commodity, 20+140]
        feature_data [num_valid_commodity, NUM_STORE + NUM_COMMODITY + NUM_PING + NUM_CHING (one-hot encoding)]
        '''

        axis_0 = len(self.store_codes) * len(self.commodity_codes)
        axis_1 = two_years + NUM_STORE + NUM_COMMODITY + NUM_PING + NUM_CHING

        self.processed_sales_data = np.zeros((axis_0, axis_1))
        processed_order_data = []

        self.__append_features__()
        self.__get_daily_data__()
        self.__remove_zero_row__()
        self.__sort_feature__()

        train_data = torch.tensor(self.processed_sales_data[:, :two_years-140], dtype=torch.float)
        test_data = torch.tensor(self.processed_sales_data[:, two_years-140-20:two_years], dtype=torch.float)
        feature_data = torch.tensor(self.processed_sales_data[:, two_years:], dtype=torch.float)

        print(torch.mean(train_data))
        train_mean, train_std, train_data = self.__normalize__(train_data)
        test_mean, test_std, test_data = self.__normalize__(test_data)
        print(torch.mean(train_data))

        with open(self.output_dir / 'train_data.pkl', 'wb') as file:
            pickle.dump(train_data, file)
        with open(self.output_dir / 'train_mean.pkl', 'wb') as file:
            pickle.dump(train_mean, file)
        with open(self.output_dir / 'train_std.pkl', 'wb') as file:
            pickle.dump(train_std, file)

        with open(self.output_dir / 'test_data.pkl', 'wb') as file:
            pickle.dump(test_data, file)
        with open(self.output_dir / 'test_mean.pkl', 'wb') as file:
            pickle.dump(test_mean, file)
        with open(self.output_dir / 'test_std.pkl', 'wb') as file:
            pickle.dump(test_std, file)

        with open(self.output_dir / 'feature_data.pkl', 'wb') as file:
            pickle.dump(feature_data, file)
        
    def aggregate_Ping(self):

        with open(self.output_dir / 'train_data.pkl', 'rb') as file:
            train_data = pickle.load(file)
        with open(self.output_dir / 'train_mean.pkl', 'rb') as file:
            train_mean = pickle.load(file)
        with open(self.output_dir / 'train_std.pkl', 'rb') as file:
            train_std =  pickle.load(file)

        with open(self.output_dir / 'test_data.pkl', 'rb') as file:
            test_data = pickle.load(file)
        with open(self.output_dir / 'test_mean.pkl', 'rb') as file:
            test_mean = pickle.load(file)
        with open(self.output_dir / 'test_std.pkl', 'rb') as file:
            test_std = pickle.load(file)

        with open(self.output_dir / 'feature_data.pkl', 'rb') as file:
            feature_data = pickle.load(file)        

        train_data = train_data * train_std + train_mean
        test_data = test_data * test_std + test_mean

        new_train_data = list()
        new_test_data = list()
        new_feature = list()
        tmp_train_data = torch.zeros_like(train_data[0])
        tmp_test_data = torch.zeros_like(test_data[0])
        tmp_feature = torch.zeros_like(feature_data[0])
        prev_row_comparer = torch.zeros(NUM_STORE+NUM_PING)
        for i, row in enumerate(feature_data):
            if i == 0:
                tmp_train_data += train_data[i]
                tmp_test_data += test_data[i]
                tmp_feature += row
                prev_row_comparer = torch.cat((row[:NUM_STORE], row[NUM_STORE+NUM_COMMODITY:NUM_STORE+NUM_COMMODITY+NUM_PING]), dim=0)
                continue
            row_comparer = torch.cat((row[:NUM_STORE], row[NUM_STORE+NUM_COMMODITY:NUM_STORE+NUM_COMMODITY+NUM_PING]), dim=0)
            the_same = torch.eq(prev_row_comparer, row_comparer).all()

            if not the_same:
                new_train_data.append(tmp_train_data)
                new_test_data.append(tmp_test_data)
                new_feature.append(torch.clamp(tmp_feature, min=0, max=1))
                tmp_train_data = torch.zeros_like(train_data[0])
                tmp_test_data = torch.zeros_like(test_data[0])
                tmp_feature = torch.zeros_like(feature_data[0])

            tmp_train_data += train_data[i]
            tmp_test_data += test_data[i]
            tmp_feature += row
            prev_row_comparer = row_comparer

        new_train_data.append(tmp_train_data)
        new_test_data.append(tmp_test_data)
        new_feature.append(torch.clamp(tmp_feature, min=0, max=1))
        
        train_data = torch.stack(new_train_data)
        test_data = torch.stack(new_test_data)
        feature_data = torch.stack(new_feature)

        train_mean, train_std, train_data = self.__normalize__(train_data)
        test_mean, test_std, test_data = self.__normalize__(test_data)  

        with open(self.output_dir / 'Ping/train_data.pkl', 'wb') as file:
            pickle.dump(train_data, file)
        with open(self.output_dir / 'Ping/train_mean.pkl', 'wb') as file:
            pickle.dump(train_mean, file)
        with open(self.output_dir / 'Ping/train_std.pkl', 'wb') as file:
            pickle.dump(train_std, file)

        with open(self.output_dir / 'Ping/test_data.pkl', 'wb') as file:
            pickle.dump(test_data, file)
        with open(self.output_dir / 'Ping/test_mean.pkl', 'wb') as file:
            pickle.dump(test_mean, file)
        with open(self.output_dir / 'Ping/test_std.pkl', 'wb') as file:
            pickle.dump(test_std, file)

        with open(self.output_dir / 'Ping/feature_data.pkl', 'wb') as file:
            pickle.dump(feature_data, file)

    def __get_valid_commodity_codes__(self, commodity_dataframe):

        summed_data = self.sales_data.groupby('商品代號').agg('sum')
        sales_data_lt_0 = summed_data.loc[summed_data['銷售數量'] > 20]
        commodity_codes = sales_data_lt_0.index
        expirations = commodity_dataframe['有效期限'].str.split('', n=2, expand=True)
        condition = (
            ((expirations[1] == 'D') & (pd.to_numeric(expirations[2]) < 6))
            | ((expirations[1] == 'H') & (pd.to_numeric(expirations[2]) < 6 * 24))
        )
        expiration_st_6 = commodity_dataframe.loc[condition]

        return set(commodity_codes).intersection(expiration_st_6['商品代號'].unique())

    def __get_formatted_time__(self, idx):

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

    def __get_daily_data__(self):

        for day_i in tqdm(range(365*2)):

            axis0_counter = 0
            time = self.__get_formatted_time__(day_i)

            for sc in self.store_codes:
                for cc in self.commodity_codes:
                    try:
                        self.processed_sales_data[axis0_counter, day_i] = self.sales_data.loc[time,  sc,  cc]['銷售數量', "sum"]
                    except KeyError:
                        pass

                    axis0_counter += 1

    def __append_features__(self):
        
        OFFSET = 365*2
        store_dict = dict()
        commodity_dict = dict()
        ping_dict = dict()
        ching_dict = dict()

        axis0_counter = -1

        for sc in self.store_codes:
            for cc in self.commodity_codes:
                axis0_counter += 1
                try:
                    ping = self.indexed_frame.loc[cc, sc]["品番"]
                    ching = self.indexed_frame.loc[cc, sc]["群番"]

                except KeyError as e:
                    continue

                store_one_hot = OFFSET + self.__get_one_hot__(store_dict, sc)
                commidity_one_hot = OFFSET + NUM_STORE + self.__get_one_hot__(commodity_dict, cc)
                ping_one_hot = OFFSET + NUM_STORE + NUM_COMMODITY + self.__get_one_hot__(ping_dict, ping)
                ching_one_hot = OFFSET + NUM_STORE + NUM_COMMODITY + NUM_PING + self.__get_one_hot__(ching_dict, ching)


                self.processed_sales_data[axis0_counter, [store_one_hot, commidity_one_hot, ping_one_hot, ching_one_hot]] = 1
        
        print(len(commodity_dict), len(ping_dict), len(ching_dict))

    def __remove_zero_row__(self):

        del_idx = list()
        for idx, row in enumerate(self.processed_sales_data):
            if np.sum(row[:365*2]):
                del_idx.append(idx)

        np.delete(self.processed_sales_data, del_idx, 0)

    def __sort_feature__(self):

        def cmp_to_key(mycmp):
            'Convert a cmp= function into a key= function'
            class K:
                def __init__(self, obj, *args):
                    self.obj = obj
                def __lt__(self, other):
                    return mycmp(self.obj, other.obj) < 0
                def __gt__(self, other):
                    return mycmp(self.obj, other.obj) > 0
                def __eq__(self, other):
                    return mycmp(self.obj, other.obj) == 0
                def __le__(self, other):
                    return mycmp(self.obj, other.obj) <= 0
                def __ge__(self, other):
                    return mycmp(self.obj, other.obj) >= 0
                def __ne__(self, other):
                    return mycmp(self.obj, other.obj) != 0
            return K

        def feature_cmp(x, y):
            for i in range(two_years, two_years+NUM_STORE):
                if x[i] != y[i]:
                    return y[i] - x[i]
            for i in range(two_years+NUM_STORE+NUM_COMMODITY, two_years+NUM_STORE+NUM_COMMODITY+NUM_PING+NUM_CHING):
                if x[i] != y[i]:
                    return y[i] - x[i]   
            for i in range(two_years+NUM_STORE, two_years+NUM_STORE+NUM_COMMODITY):
                if x[i] != y[i]:
                    return y[i] - x[i]
            return 0     
        print("sorting...")
        self.processed_sales_data = np.array(sorted(self.processed_sales_data, key=cmp_to_key(feature_cmp)))

    def __get_one_hot__(self, codebook, target):

        if target not in codebook:
            codebook[target] = len(codebook)
        return codebook[target]

    def __normalize__(self, tensor):
        mean = torch.mean(tensor)
        std = torch.std(tensor)
        tensor = (tensor - mean) / std
        return mean, std, tensor
        

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

def get_valid_commodity_codes(commodity_dataframe, sales_data):
    summed_data = sales_data.groupby('商品代號').agg('sum')

    sales_data_lt_0 = summed_data.loc[summed_data['銷售數量'] > 20]

    commodity_codes = sales_data_lt_0.index

    expirations = commodity_dataframe['有效期限'].str.split('', n=2, expand=True)

    condition = (
        ((expirations[1] == 'D') & (pd.to_numeric(expirations[2]) < 6))
        | ((expirations[1] == 'H') & (pd.to_numeric(expirations[2]) < 6 * 24))
    )
    expiration_st_6 = commodity_dataframe.loc[condition]

    return set(commodity_codes).intersection(expiration_st_6['商品代號'].unique())

if __name__ == '__main__':
    pc = preprocessor()
    # pc.generate()
    pc.aggregate_Ping()