import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class transoformer():
    def __init__(self, file_name):

        self.file_name = file_name
        self.type = self.file_name[-12:-10]
        self.df = pd.read_csv(self.file_name, sep='\t')

    def transform(self):

        group_metrics = ["原始店號", "商品代號", "日期"]
        if self.type == "訂貨" or self.type == "庫存":
            group_metrics[2] ="交易日期"
        df_agged = self.df.groupby(group_metrics, as_index=False).agg(["sum"])
        indexes = df_agged.index
        df_agged = df_agged.values
        self.transformed_table = list()

        prev_shop_code = -1
        prev_product_code = -1
        base_date_obj = datetime.strptime(self.file_name[-8:-4]+"0101", "%Y%m%d") # file must end with 201X.txt

        for i, (shop_code, product_code, date) in enumerate(indexes):

            if shop_code != prev_shop_code or product_code != prev_product_code:
                self.transformed_table.append([0 for i in range(365+2)]) # no leap year in 2017~2018
                self.transformed_table[-1][0] = shop_code
                self.transformed_table[-1][1] = product_code
                prev_shop_code = shop_code
                prev_product_code = product_code


            date_obj = datetime.strptime(str(date), "%Y%m%d")
            delta = date_obj - base_date_obj
            idx = int(delta.days) + 2

            if self.type == "廢棄" or self.type == "銷售":
                self.transformed_table[-1][idx] = df_agged[i, 1]
            elif self.type == "訂貨":
                self.transformed_table[-1][idx] = str(df_agged[i, 0]) + "/" + str(df_agged[i, 1])
            else:
                self.transformed_table[-1][idx] = df_agged[i, 0]                

    def write(self):

        base_date_obj = datetime.strptime(self.file_name[-8:-4]+"0101", "%Y%m%d") # file must end with 201X.txt
        columns = ["原始店號", "商品代號"] + [ datetime.strftime(base_date_obj+timedelta(days=i), "%Y%m%d") for i in range(365) ]

        out_df = pd.DataFrame(np.array(self.transformed_table))
        out_df.to_csv(self.file_name[:-4]+"_transformed.csv",index=False, header=columns)

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='transform tables')
    parser.add_argument("--input", "-i", required=True, help='input file')

    args = parser.parse_args()
    tf = transoformer(args.input)
    tf.transform()
    tf.write()