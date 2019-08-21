import numpy as np
import pandas as pd

class observer:
    pass

df = pd.read_csv("../data/銷售數量2017_transformed.csv")

grouped_df = df.groupby("原始店號")

for key, item in grouped_df:
    print(key)
    groupby_store_data = np.array(grouped_df.get_group(key))
    
    pre_label = -1
    same_label_list = []
    
    for row in groupby_store_data:
        label = row[1] // 10000
        print(label)
        if label != pre_label:
            if same_label_list:
                for cor_row in np.corrcoef(same_label_list):
                    print(list(cor_row))
                input()
            pre_label = label
            same_label_list.clear()
        
        same_label_list.append(row[2:])