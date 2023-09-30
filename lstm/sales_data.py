import pandas as pd
import os
import numpy as np
import torch
import pickle
from ipdb import set_trace
from torch.utils.data import Dataset as DS

from common_utils import join


class Sales_Dataset(DS):
    def get_log_ret(self, df: pd.DataFrame, y_col: str, sort_keys=[]):
        """Calculate in-place the log returns of y_col"""
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

    def __init__(self, dir_pth, is_train=True):
        self.H = pd.read_csv(join(dir_pth, "holidays_events.csv"), index_col=False)
        self.O = pd.read_csv(join(dir_pth, "oil.csv"), index_col=False)
        self.S = pd.read_csv(join(dir_pth, "stores.csv"), index_col=False)
        self.TR = pd.read_csv(join(dir_pth, "train.csv"), index_col=False)
        self.TS = pd.read_csv(join(dir_pth, "transactions.csv"), index_col=False)

        # preprocess nominal data
        if is_train:
            len_dict = {
                "family": len(set(self.TR["family"])),
            }
            family_encoding = self.get_nominal_dict(self.TR.family)
            city_encoding = self.get_nominal_dict(self.S.city)
            type_encoding = self.get_nominal_dict(self.S.type)
            cluster_encoding = self.get_nominal_dict(self.S.cluster)

            with open(join(dir_pth, "len_dict.pkl"), "wb") as f:
                pickle.dump(len_dict, f)
            with open(join(dir_pth, "family_encode.pkl"), "wb") as f:
                pickle.dump(family_encoding, f)
            with open(join(dir_pth, "city_encode.pkl"), "wb") as f:
                pickle.dump(city_encoding, f)
            with open(join(dir_pth, "type_encode.pkl"), "wb") as f:
                pickle.dump(type_encoding, f)
            with open(join(dir_pth, "cluster_encode.pkl"), "wb") as f:
                pickle.dump(cluster_encoding, f)

        self.family_len = len_dict["family"]

        self.TR.family = self.TR.family.map(family_encoding)
        self.S.city = self.S.city.map(city_encoding)
        self.S.type = self.S.type.map(type_encoding)
        self.S.cluster = self.S.cluster.map(cluster_encoding)
        self.S = self.S[["store_nbr", "city", "cluster", "type"]]

        # order the rows
        self.TR.sort_values(["store_nbr", "date", "family"], inplace=True)
        self.TR.set_index(["store_nbr", "date"], inplace=True)
        self.TR.drop("id")
        self.TS.sort_values(["store_nbr", "date"], inplace=True)
        self.TS.set_index(["store_nbr", "date"], inplace=True)
        self.O.date = pd.to_datetime(self.O.date)
        self.O.sort_values(["date"], inplace=True)
        self.O.set_index(["date"], inplace=True)
        self.S = self.S.sort_values(["store_nbr"]).set_index(["store_nbr"])

        # preprocess return data
        self.TR.sales = self.get_log_ret(
            self.TR, "sales", ["store_nbr", "family", "date"]
        )
        self.TR.onpromotion = self.get_log_ret(
            self.TR, "onpromotion", ["store_nbr", "family", "date"]
        )
        self.TS.transactions = self.get_log_ret(
            self.TS, "transactions", ["store_nbr", "date"]
        )

        self.O = self.O.asfreq("D")
        self.O = self.O.interpolate()
        self.O.dcoilwtico = self.get_log_ret(self.O, "dcoilwtico", ["date"])

        # construct the primary key
        self.ids = sorted(list(set(self.TR.index)))[1:]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """Sample dimension: per (date, store_nbr):
        (family(N)) * (sale + onpromotion + oil + transaction + city + cluster + type)
        """

        store_nbr, date = self.ids[idx]
        sample = torch.zeros((self.family_len, 7), dtype=torch.float32)
        sale_data = self.TR.loc[(store_nbr, date)]
        oil_data = self.O.loc[(date)].to_numpy()
        if (store_nbr, date) not in self.TS.index:
            trans_data = np.zeros((sample.shape[0]))
        else:
            trans_data = self.TS.loc[(store_nbr, date)].to_numpy()
        s_data = self.S.loc[(store_nbr)].to_numpy()

        sample[:, :2] = torch.tensor(sale_data[["sales", "onpromotion"]].to_numpy())
        sample[:, 2] = torch.tensor(oil_data)
        sample[:, 3] = torch.tensor(trans_data)
        sample[:, 4:] = torch.tensor(s_data)

        return sample
