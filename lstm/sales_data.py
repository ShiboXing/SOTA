import pandas as pd
import os
import numpy as np
import torch
import pickle

from ipdb import set_trace
from torch.utils.data import Dataset as DS


class Sales_Dataset(DS):
    
    def get_log_ret(self, df: pd.DataFrame, y_col: str, sort_keys=[]):
        """Calculate in-place the log returns of y_col"""
        df.sort_values(by=sort_keys, inplace=True)
        rets = np.log(df[y_col] / df[y_col].shift(1))
        rets = rets.replace({np.nan: 0, -np.inf: 0, np.inf: 1})
        
        return rets

    def z_series(self, df: pd.Series, clip=False):
        """Normalize dataframe series while eliminating the effect of zeros"""
        # df_tmp = df[df != 0.0]
        if clip:
            df = df.clip(
                lower=df.mean() - 2 * df.std(),
                upper=df.mean() + 2 * df.std(),
            )
        return (df - df.mean()) / df.std()


    def get_nominal_dict(self, df: pd.Series):
        d = {elem: i for i, elem in enumerate(set(df))}
        vals = np.array(list(d.values()))
        z_vals = self.z_series(vals)
        d.update(zip(d.keys(), z_vals))

        return d


    def __init__(self, dir_pth, is_train = True):
        self.H = pd.read_csv(f"{dir_pth}/holidays_events.csv", index_col=False)
        self.O = pd.read_csv(f"{dir_pth}/oil.csv", index_col=False)
        self.S = pd.read_csv(f"{dir_pth}/stores.csv", index_col=False)
        self.TR = pd.read_csv(f"{dir_pth}/train.csv", index_col=False)
        self.TS = pd.read_csv(f"{dir_pth}/transactions.csv", index_col=False)

        # construct the primary key
        ids = set()
        self.TR.apply(lambda row: ids.add((row["date"], row["store_nbr"])), axis=1)
        self.ids = sorted(list(ids))

        # preprocess nominal data
        if is_train:
            family_encoding = self.get_nominal_dict(self.TR.family)
            city_encoding = self.get_nominal_dict(self.S.city)
            type_encoding = self.get_nominal_dict(self.S.type)
            cluster_encoding = self.get_nominal_dict(self.S.cluster)

            with open(f"{dir_pth}/family_encode.pkl", "wb") as f: pickle.dump(family_encoding, f)
            with open(f"{dir_pth}/city_encode.pkl", "wb") as f: pickle.dump(city_encoding, f)
            with open(f"{dir_pth}/type_encode.pkl", "wb") as f: pickle.dump(type_encoding, f)
            with open(f"{dir_pth}/cluster_encode.pkl", "wb") as f: pickle.dump(cluster_encoding, f)

        self.TR.family = self.TR.family.map(family_encoding)
        self.S.city = self.S.city.map(city_encoding)
        self.S.type = self.S.type.map(type_encoding)
        self.S.cluster = self.S.cluster.map(cluster_encoding)
        self.S = self.S[["city", "cluster", "type"]]

        # preprocess return data
        self.TR.sales = self.get_log_ret(
            self.TR, "sales", ["store_nbr", "family", "date"]
        )
        # self.TR.onpromotion = self.z_series(self.TR.onpromotion)
        # self.promo_std, self.promo_mean = (
        #     self.TR.onpromotion.std(),
        #     self.TR.onpromotion.mean(),
        # )
        self.TR.onpromotion = self.get_log_ret(self.TR, "onpromotion", ["store_nbr", "family", "date"])
        self.TS.transactions = self.get_log_ret(
            self.TS, "transactions", ["store_nbr", "date"]
        )
        self.O.dcoilwtico = self.get_log_ret(self.O, "dcoilwtico", ["date"])

        # order the rows
        self.TR.sort_values(["date", "store_nbr", "family"], inplace=True)
        self.TS.sort_values(["date", "store_nbr"], inplace=True)
        self.O.sort_values(["date"], inplace=True)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        date, store_nbr = self.ids[idx]
        tr = self.TR[(self.TR.date == date) & (self.TR.store_nbr == store_nbr)]
        ts = self.TS[(self.TS.date == date) & (self.TR.store_nbr == store_nbr)]
        o = self.O[self.O.date == date]

        return (tr, ts, o)
