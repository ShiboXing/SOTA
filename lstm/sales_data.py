import pandas as pd
import os
import numpy as np
import torch
import pickle
from ipdb import set_trace
from torch.utils.data import Dataset as DS

from common_utils import join


class Sales_Dataset(DS):
    @staticmethod
    def log_ret_2_sales(rets, base_price):
        sales = np.array([base_price])
        for r in rets:
            sales = np.append(sales, 10**r * sales[-1])
        return sales[1:]

    def get_log_ret(self, df: pd.DataFrame, y_col: str):
        """Calculate in-place the log returns of y_col"""
        np.seterr(divide="ignore")  # suppress divide by zero
        rets = np.log10(df[y_col] / df[y_col].shift(1))
        np.seterr(divide="warn")

        return Sales_Dataset.df_fix_float(rets)

    def z_series(self, df: pd.Series, clip=False):
        """Normalize dataframe series while eliminating the effect of zeros"""
        # df_tmp = df[df != 0.0]
        if clip:
            df = df.clip(lower=df.mean() - 2 * df.std(), upper=df.mean() + 2 * df.std())
        return (df - df.mean()) / df.std()

    def get_nominal_dict(self, df: pd.Series):
        d = {elem: i for i, elem in enumerate(set(df))}
        vals = np.array(list(d.values()))
        z_vals = self.z_series(vals)
        d.update(zip(d.keys(), z_vals))

        return d

    def df_fix_float(df):
        return df.replace({np.nan: 0, -np.inf: 0, np.inf: 1})

    @staticmethod
    def df_adjust_date(df: pd.DataFrame, date_a, date_b):
        """
        fix the range of dataframe as desired and interpolate the data in the missing dates
        """
        # expand
        if df.index[0] > date_a:
            df = pd.concat([pd.DataFrame(index=[date_a]), df])
        if df.index[-1] < date_b:
            df = pd.concat([df, pd.DataFrame(index=[date_b])])

        # clamp
        df = df[(df.index >= date_a) & (df.index <= date_b)]

        # interpolate
        df = df.asfreq("D").interpolate()
        return Sales_Dataset.df_fix_float(df)

    def __init__(self, dir_pth, seq_len=500, is_train=True):
        self.H = pd.read_csv(join(dir_pth, "holidays_events.csv"), index_col=False)
        self.O = pd.read_csv(join(dir_pth, "oil.csv"), index_col=False)
        self.S = pd.read_csv(join(dir_pth, "stores.csv"), index_col=False)
        self.TR = pd.read_csv(join(dir_pth, "train.csv"), index_col=False)
        self.TT = pd.read_csv(join(dir_pth, "test.csv"), index_col=False)
        self.TS = pd.read_csv(join(dir_pth, "transactions.csv"), index_col=False)
        self.is_train = is_train

        # preprocess nominal data
        self.store_nbrs = set(self.TR["store_nbr"])
        self.family_set = set(self.TR["family"])
        # self.family_encoding = self.get_nominal_dict(self.TR.family)
        city_encoding = self.get_nominal_dict(self.S.city)
        type_encoding = self.get_nominal_dict(self.S.type)
        cluster_encoding = self.get_nominal_dict(self.S.cluster)

        self.S.city = self.S.city.map(city_encoding)
        self.S.type = self.S.type.map(type_encoding)
        self.S.cluster = self.S.cluster.map(cluster_encoding)
        self.S = self.S[["store_nbr", "city", "cluster", "type"]]

        # order the rows
        self.TS.sort_values(["store_nbr", "date"], inplace=True)
        self.TS.set_index(["store_nbr"], inplace=True)
        self.TR.drop("id", axis=1, inplace=True)
        self.TR.sort_values(["store_nbr", "family", "date"], inplace=True)
        self.TR.set_index(["store_nbr"], inplace=True)
        self.TT.drop("id", axis=1, inplace=True)
        self.TT.sort_values(["store_nbr", "family", "date"], inplace=True)
        self.TT.set_index(["store_nbr"], inplace=True)
        self.O.sort_values(["date"], inplace=True)
        self.O.set_index(["date"], inplace=True)
        self.O.index = pd.to_datetime(self.O.index)
        self.S = self.S.sort_values(["store_nbr"]).set_index(["store_nbr"])

        # get statistics
        self.sample_seq_len = seq_len
        min_date, max_date = min(self.TR.date), max(self.TR.date)
        self.num_days = len(pd.date_range(start=min_date, end=max_date))
        if self.is_train:
            self.num_store_samples = self.num_days - self.sample_seq_len # predict T+1
        else:
            self.num_store_samples = 16 

        # store the base sales for inference purpose
        self.base_sales = self.TR[self.TR.date == "2017-08-15"].reset_index()

        # preprocess return data
        self.TR.sales = self.get_log_ret(self.TR, "sales")
        self.TR.onpromotion = self.get_log_ret(self.TR, "onpromotion")
        self.TS.transactions = self.get_log_ret(self.TS, "transactions")
        self.O = self.O.asfreq("D").interpolate()
        self.O.dcoilwtico = self.get_log_ret(self.O, "dcoilwtico")

        # construct the primary key
        self.ids = sorted(list(set(self.S.index)))


    def __len__(self):
        return len(self.ids) * self.num_store_samples
    
    
    def _get_sample(self, store_id, local_id, is_train):
        
        """Sample dimension: per (date, store_nbr):
        sequence_length * (family(N) + oil(1) + transaction(1) + city(1) + cluster(1) + type(1))

        L * (33*2 + 2 + 3)
        """            
        store_nbr = self.ids[store_id]
        if is_train:
            sale_data = self.TR.loc[store_nbr].set_index("date")
        else:
            sale_data = self.TT.loc[store_nbr].set_index("date")
        sale_data.index = pd.to_datetime(sale_data.index)
        start_date, end_date = (
            pd.to_datetime(sale_data.index[0]),
            pd.to_datetime(sale_data.index[-1]),
        )

        # transform the sale data
        sale_df = pd.DataFrame(
            index=pd.to_datetime(
                sale_data[sale_data.family == sale_data.iloc[0].family].index
            )
        )
        # make a column for each family of product
        for d in sorted(list(set(sale_data.family))):
            if is_train:
                sale_df = pd.concat(
                    [
                        sale_df,
                        pd.DataFrame(
                            {f"{d}_sales": sale_data[sale_data.family == d].sales}
                        ),
                    ],
                    axis=1,
                )
            sale_df = pd.concat(
                [
                    sale_df,
                    pd.DataFrame(
                        {
                            f"{d}_onpromotion": sale_data[
                                sale_data.family == d
                            ].onpromotion
                        }
                    ),
                ],
                axis=1,
            )

        # fix the data on the missing dates
        sale_df = self.df_adjust_date(sale_df, start_date, end_date)
        trans_data = self.TS.loc[store_nbr].set_index("date")
        trans_data.index = pd.to_datetime(trans_data.index)
        trans_data = self.df_adjust_date(trans_data, start_date, end_date)
        oil_data = self.df_adjust_date(self.O, start_date, end_date)

        # append other features
        sale_df = pd.concat([sale_df, oil_data], axis=1)
        if is_train:
            sale_df = pd.concat([sale_df, trans_data], axis=1)
        s_data = self.S.loc[(store_nbr)].to_numpy()

        # combine the features into batch
        sample = torch.zeros(
            sale_df.shape[0], sale_df.shape[1] + 3, dtype=torch.float32
        ).cuda()
        sample[:, :sale_df.shape[1]] = torch.tensor(
            sale_df.to_numpy(), dtype=torch.float32
        )
        sample[:, -3:] = torch.tensor(s_data, dtype=torch.float32)

        start_t, end_t = local_id, local_id + self.sample_seq_len if is_train else self.num_store_samples-1
        cols = sale_df.filter(like="sales").columns.tolist() + ["transactions"]
        data = sample[start_t:end_t]
        if is_train:
            return data, torch.tensor(
                sale_df[cols].to_numpy(), dtype=torch.float32
            )[start_t+1 : min(self.num_days, end_t+1)]
        else:
            return data, None


    def __getitem__(self, idx):
        # prepare to build batch
        store_id, local_id = idx // self.num_store_samples, idx % self.num_store_samples
        if self.is_train:
            return self._get_sample(store_id, local_id, self.is_train)
        else:
            test_part, _ = self._get_sample(store_id, local_id, False)
            train_part, _ = self._get_sample(store_id, self.num_store_samples, True)
            return 