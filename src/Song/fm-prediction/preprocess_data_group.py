import pickle
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import numpy as np


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
    data_dir = Path('../data')

    commodity_dataframe = pd.read_csv(
        data_dir / '商品主檔.txt', sep='\t')

    store_codes = (commodity_dataframe.loc[:, '原始店號']
                   .sort_values().unique())
    commodity_codes = (commodity_dataframe.loc[:, '商品代號']
                       .sort_values().unique())

    group_data = commodity_dataframe[['商品代號', '品番', '群番']].values
    group_data = np.unique(group_data, axis = 0)

    commodity_codes2p = {}
    commodity_codes2c = {}
    for group in group_data:
        commodity_codes2p[group[0]] = group[1]
        commodity_codes2c[group[0]] = group[2]


    sales_dataframes = []
    for year in [2017, 2018]:
        sales_dataframes.append(
            pd.read_csv(
                data_dir / '銷售數量{}.txt'.format(year), sep='\t'
            ).loc[:, ['原始店號', '日期', '商品代號', '銷售數量']]
        )

    sales_data = pd.concat(sales_dataframes, axis=0)

    del sales_dataframes

    commodity_codes = get_valid_commodity_codes(commodity_dataframe, sales_data)
    print(len(commodity_codes))
    # for i, cc in enumerate(commodity_codes):
    #     if cc == 610010:
    #         print(i)

    del commodity_dataframe

    sales_data = sales_data.groupby(['日期', '原始店號', '商品代號'])

    processed_sales_data, processed_order_data = [], []
    for day_i in tqdm(range(2 * 365)):
        time = get_formatted_time(day_i)
        # print(time, end=' ', flush=True)

        for sc in store_codes:
            for cc in commodity_codes:
                # print(cc)
                try:
                    # print(sales_data.get_group((time, sc, cc))['銷售數量'].agg('sum'))
                    processed_sales_data.append(
                        sales_data.get_group((time, sc, cc))['銷售數量'].agg('sum'))
                except KeyError:
                    processed_sales_data.append(0)
    
    processed_sales_data = torch.tensor(processed_sales_data, dtype=torch.float)
    processed_sales_data = processed_sales_data.view(730, 5, -1)

    processed_sales_data_p_dict = {}
    processed_sales_data_c_dict = {}
    commodity_codes_list = list(commodity_codes)
    for i in range(len(commodity_codes_list)):
        if commodity_codes2p[commodity_codes_list[i]] not in processed_sales_data_p_dict:
            processed_sales_data_p_dict[commodity_codes2p[commodity_codes_list[i]]] = np.array(processed_sales_data[:, :, i].tolist()).reshape((730, 5))
        else:
            processed_sales_data_p_dict[commodity_codes2p[commodity_codes_list[i]]] += np.array(processed_sales_data[:, :, i].tolist()).reshape((730, 5))
        if commodity_codes2c[commodity_codes_list[i]] not in processed_sales_data_c_dict:
            processed_sales_data_c_dict[commodity_codes2c[commodity_codes_list[i]]] = np.array(processed_sales_data[:, :, i].tolist()).reshape((730, 5))
        else:
            processed_sales_data_c_dict[commodity_codes2c[commodity_codes_list[i]]] += np.array(processed_sales_data[:, :, i].tolist()).reshape((730, 5))
    
    pinfan_code = [k for k in processed_sales_data_p_dict]
    qunfan_code = [k for k in processed_sales_data_c_dict]

    sales_data_pinfan = np.dstack(processed_sales_data_p_dict.values())
    sales_data_pinfan = torch.tensor(sales_data_pinfan, dtype=torch.float)
    sales_mean_pinfan = torch.mean(sales_data_pinfan)
    sales_std_pinfan = torch.std(sales_data_pinfan)
    sales_data_pinfan = (sales_data_pinfan - sales_mean_pinfan) / sales_std_pinfan

    sales_data_qunfan = np.dstack(processed_sales_data_c_dict.values())
    sales_data_qunfan = torch.tensor(sales_data_qunfan, dtype=torch.float)
    sales_mean_qunfan = torch.mean(sales_data_qunfan)
    sales_std_qunfan = torch.std(sales_data_qunfan)
    sales_data_qunfan = (sales_data_qunfan - sales_mean_qunfan) / sales_std_qunfan

    sales_mean = torch.mean(processed_sales_data)
    sales_std = torch.std(processed_sales_data)

    processed_sales_data = (processed_sales_data - sales_mean) / sales_std
    print(processed_sales_data.shape)
    with open(data_dir / 'sales_data.pkl', 'wb') as file:
        pickle.dump(processed_sales_data, file)
    with open(data_dir / 'sales_mean.pkl', 'wb') as file:
        pickle.dump(sales_mean, file)
    with open(data_dir / 'sales_std.pkl', 'wb') as file:
        pickle.dump(sales_std, file)
    with open(data_dir / 'commodity_codes.pkl', 'wb') as file:
        pickle.dump(commodity_codes, file)
    with open(data_dir / 'store_codes.pkl', 'wb') as file:
        pickle.dump(store_codes, file)
    with open(data_dir / 'sales_data_pinfan.pkl', 'wb') as file:
        pickle.dump(sales_data_pinfan, file)
    with open(data_dir / 'sales_mean_pinfan.pkl', 'wb') as file:
        pickle.dump(sales_mean_pinfan, file)
    with open(data_dir / 'sales_std_pinfan.pkl', 'wb') as file:
        pickle.dump(sales_std_pinfan, file)
    with open(data_dir / 'sales_data_qunfan.pkl', 'wb') as file:
        pickle.dump(sales_data_qunfan, file)
    with open(data_dir / 'sales_mean_qunfan.pkl', 'wb') as file:
        pickle.dump(sales_mean_qunfan, file)
    with open(data_dir / 'sales_std_qunfan.pkl', 'wb') as file:
        pickle.dump(sales_std_qunfan, file)