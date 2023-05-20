import torch
import pandas as pd
import os

from torch.utils.data import Dataset as DS


class Sales_Dataset(DS):
    def z_series(self, df: pd.Series):
        """Normalize dataframe series while eliminating the effect of zeros"""
        df_tmp = df[df != 0.0]
        df = df.clip(
            lower=df_tmp.mean() - 2 * df_tmp.std(),
            upper=df_tmp.mean() + 2 * df_tmp.std(),
        )
        return (df - df.mean()) / df.std()

    def parse_nominal(self, df: pd.Series):
        """Transform nominal data into numerical data"""
        mapping = {elem: i for i, elem in enumerate(set(df))}
        return df.map(mapping)

    def __init__(self, dir_pth):
        self.H = pd.read_csv(f"{dir_pth}/holidays_events.csv", index_col=False)
        self.O = pd.read_csv(f"{dir_pth}/oil.csv", index_col=False)
        self.S = pd.read_csv(f"{dir_pth}/stores.csv", index_col=False)
        self.TR = pd.read_csv(f"{dir_pth}/train.csv", index_col=False)
        self.TS = pd.read_csv(f"{dir_pth}/transactions.csv", index_col=False)

        # construct the primary key
        ids = set()
        self.TR.apply(lambda row: ids.add((row["date"], row["store_nbr"])), axis=1)
        self.ids = sorted(list(ids))
        self.TR.sort_values(["date", "store_nbr"], inplace=True)

        # preprocess data
        self.TR.family = self.z_series(self.parse_nominal(self.TR.family))
        self.TR.sales = self.z_series(self.TR.sales)
        self.TR.onpromotion = self.z_series(self.TR.onpromotion)

        self.TS.transactions = self.z_series(self.TS.transactions)
        self.S.city = self.z_series(self.parse_nominal(self.S.city))
        self.S.cluster = self.z_series(self.S.cluster)
        self.S.type = self.z_series(self.parse_nominal(self.S.type))
        self.S = self.S[["city", "cluster", "type"]]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        date, store_nbr = self.ids[idx]
        tr = self.TR[(self.TR.date == date) & (self.TR.store_nbr == store_nbr)]
        ts = self.TS[(self.TS.date == date) & (self.TR.store_nbr == store_nbr)]
        o = self.O[self.O.date == date]
        
        return (
            tr, ts, o
        )
