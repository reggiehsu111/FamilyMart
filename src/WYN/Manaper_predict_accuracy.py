import pandas as pd
import numpy as np

years = ["2017", "2018"]
shop_code = [1205, 3047, 5638, 7651, 12236]
delay = 4

for year in years:
    with open("data/訂貨進貨" + year + "_transformed.csv", "r", encoding='utf-8') as f:
        order_df = pd.read_csv(f, sep=',')
    with open("data/銷售數量" + year + "_transformed.csv", "r", encoding='utf-8') as f:
        sales_df = pd.read_csv(f, sep=',')
    for shop in shop_code:
        shop_order_df = order_df[order_df["原始店號"] == shop]
        shop_sales_df = sales_df[sales_df["原始店號"] == shop]
        sales_code = set(shop_sales_df["商品代號"].values)

        total_sales = 0
        total_diff = 0

        for comm_code in shop_order_df["商品代號"]:
            if comm_code not in sales_code:
                continue
            order_list = shop_order_df[shop_order_df["商品代號"] == comm_code].values.reshape((367,))
            sales_list = shop_sales_df[shop_sales_df["商品代號"] == comm_code].values.reshape((367,))
            diff = abs(sales_list[2+delay:] - order_list[2:-delay])
            total_diff += np.sum(diff)
            total_sales += np.sum(sales_list[2+delay:])
            # print(order_list.shape)
            # print(sales_list.shape)
            # print(diff)
            # print(total_diff, total_sales)
            # input()

        print(year + ", {}:".format(shop))
        print("accuracy: ", total_diff/total_sales, " ({}/{})".format(total_diff, total_sales))

