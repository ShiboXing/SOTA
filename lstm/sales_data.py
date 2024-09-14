import pandas as pd
import os
from os.path import join
import numpy as np
import torch
import pickle
from datetime import timedelta
from ipdb import set_trace
from torch.utils.data import Dataset as DS


class Sales_Dataset(DS):

    @staticmethod
    def ret_2_sale(data: torch.Tensor, base_sales: torch.Tensor):
        # (torch.exp(log_returns) - 1 + 1）* base_sales
        return torch.exp(data) * base_sales

    def get_log_ret(self, df: pd.DataFrame, y_col: str):
        """Calculate in-place the log returns of y_col"""
        np.seterr(divide="ignore")  # suppress divide by zero
        rets = np.log10(df[y_col] / df[y_col].shift(1))
        np.seterr(divide="warn")

        return Sales_Dataset.df_fix_float(rets)

    def get_log_ret_v2(self, A: torch.tensor, B: torch.tensor):
        """Calculate in-place the log returns of y_col"""
        X = B / A - 1
        rets = torch.log1p(X)
        return torch.nan_to_num(rets, nan=0.0, posinf=1, neginf=0.0)

    def z_series(self, df: pd.DataFrame, clip=False):
        """
        Normalize dataframe series while eliminating the effect of zeros
        """
        # df_tmp = df[df != 0.0]
        stddev = df.std()
        if type(stddev) != np.float64 and type(stddev) != np.float32:
            stddev[stddev == 0.0] = 1.0  # prevent div by 0
        if clip:
            df = df.clip(lower=df.mean() - 2 * stddev, upper=df.mean() + 2 * stddev)
        return (df - df.mean()) / stddev

    def get_nominal_dict(self, df: pd.Series):
        # ensure that replicability of nominal encoding
        sorted_keys = sorted(set(df))

        d = {elem: i for i, elem in enumerate(sorted_keys)}
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

    def set_log_rets(self, label: torch.Tensor, store_id, date):
        offset = 0

        # assign sales data
        for col in self.families:
            store_data = self.TR.loc[store_id]
            store_data[(store_data.family == col) & (store_data.date == date)][
                "sales"
            ] = label[offset].item()

            offset += 1

        # assign transaction data
        self.TS.loc[
            (self.TS.index == store_id) & (self.TS["date"] == date), "transactions"
        ] = label[-1].item()

    def __apply_holidays__(self):
        self.TR = pd.merge(self.TR, self.S, on="store_nbr")
        self.TR["hol"] = 0.0
        self.TR["hol"] = self.TR["hol"].astype("float32")
        for _, row in self.H.iterrows():
            if row.transferred == True:
                continue
            if row.locale == "Local":
                self.TR.loc[
                    (self.TR.date == row.date) & (self.TR.city == row.locale_name),
                    "hol",
                ] = 0.2
            elif row.locale == "Regional":
                self.TR.loc[
                    (self.TR.date == row.date) & (self.TR.state == row.locale_name),
                    "hol",
                ] = 0.35
            else:
                self.TR.loc[(self.TR.date == row.date), "hol"] = 0.5

        self.TR = self.TR.drop(["city", "state", "type", "cluster"], axis=1)

    def __init__(self, dir_pth, seq_len=500, is_train=True, device="cpu"):
        self.device = device
        self.H = pd.read_csv(join(dir_pth, "holidays_events.csv"), index_col=False)
        self.O = pd.read_csv(join(dir_pth, "oil.csv"), index_col=False)
        self.S = pd.read_csv(join(dir_pth, "stores.csv"), index_col=False)
        self.TR = pd.read_csv(join(dir_pth, "train.csv"), index_col=False)
        self.TS = pd.read_csv(join(dir_pth, "transactions.csv"), index_col=False)
        self.TT = pd.read_csv(join(dir_pth, "test.csv"), index_col=False)
        self.is_train = is_train

        # apply holiday column to TR
        _TR_hol_cache_ = "TR_hol_cache.csv"
        if os.path.exists(_TR_hol_cache_):
            self.TR = pd.read_pickle(_TR_hol_cache_)
        else:
            self.__apply_holidays__()
            self.TR.to_pickle("TR_hol_cache.csv")

        # preprocess nominal data
        self.store_nbrs = set(self.TR.index)
        self.families = sorted(list(set(self.TR["family"])))
        city_encoding = self.get_nominal_dict(self.S.city)
        type_encoding = self.get_nominal_dict(self.S.type)
        cluster_encoding = self.get_nominal_dict(self.S.cluster)

        self.S.city = self.S.city.map(city_encoding)
        self.S.type = self.S.type.map(type_encoding)
        self.S.cluster = self.S.cluster.map(cluster_encoding)
        self.S = self.S[["store_nbr", "city", "cluster", "type"]]
        self.S = self.S.sort_values(["store_nbr"]).set_index(["store_nbr"])

        # standardize the date column
        self.TR.date = pd.to_datetime(self.TR.date)
        self.TT.date = pd.to_datetime(self.TT.date)
        self.TS.date = pd.to_datetime(self.TS.date)
        self.O.date = pd.to_datetime(self.O.date)

        # concat test and train data together
        self.TR = pd.concat([self.TR, self.TT], axis=0).fillna(0)

        # order the rows
        self.TS.sort_values(["store_nbr", "date"], inplace=True)
        self.TS.set_index(["store_nbr"], inplace=True)
        self.TR.drop("id", axis=1, inplace=True)
        self.TR.sort_values(["store_nbr", "family", "date"], inplace=True)
        self.TR.set_index(["store_nbr"], inplace=True)
        self.O.sort_values(["date"], inplace=True)
        self.O.set_index(["date"], inplace=True)
        self.O.index = pd.to_datetime(self.O.index)

        # get statistics
        self.sample_seq_len = seq_len
        min_date, self.train_max_date, self.total_max_date = (
            min(self.TS.date),
            max(self.TS.date),
            max(self.TR.date),
        )
        self.num_days = len(pd.date_range(start=min_date, end=self.train_max_date))
        self.num_store_samples = (
            self.num_days - self.sample_seq_len
        )  # num of samples per store

        # store the base sales for inference purpose
        self.base_sales = self.TR[self.TR.date == "2017-08-15"].reset_index()
        self.base_sales = self.base_sales[["store_nbr", "date", "sales", "family"]]

        # preprocess return data
        self.TR_OG = self.TR.copy()  # for building labels
        self.TR.sales = self.get_log_ret(self.TR, "sales")
        self.TR.onpromotion = self.get_log_ret(self.TR, "onpromotion")
        self.TS.transactions = self.get_log_ret(self.TS, "transactions")
        self.O = self.O.asfreq("D").interpolate()
        self.O.dcoilwtico = self.get_log_ret(self.O, "dcoilwtico")

        # extend TS to max date
        self.ids = sorted(list(set(self.S.index)))

        # re-sort TS by (store # and date)
        self.TS = self.TS.reset_index().sort_values(["store_nbr", "date"])
        self.TS.set_index(["store_nbr"], inplace=True)

        self.INFER_DAYS = len(set(self.TR[self.TR.date > self.train_max_date].date))
        self.TR.sales += 0.001
        self.TR.onpromotion += 0.001
        if not self.is_train:
            # remove all rows unused in inference
            self.TR = self.TR[
                self.TR.date
                > self.total_max_date - timedelta(days=seq_len + self.INFER_DAYS)
            ]

    def __len__(self):
        if self.is_train:
            return len(self.ids) * self.num_store_samples
        else:
            return len(self.ids)

    def __getitem__(self, idx):
        """Sample dimension: per (date, store_nbr):
        sequence_length * (family(N)*2 + oil(1) + transaction(1) + city(1) + cluster(1) + type(1) + holiday(1))
        MEAN: 0
        MIN: -1
        MAX: 1
        L * (33*2 + 6)
        """
        if self.is_train:
            store_id, local_id = (
                idx // self.num_store_samples,
                idx % self.num_store_samples,
            )
        else:
            store_id, local_id = idx, 0
        store_nbr = self.ids[store_id]
        sale_data = self.TR.loc[store_nbr].set_index("date")
        sale_data.index = pd.to_datetime(sale_data.index)
        start_date, end_date = (
            pd.to_datetime(sale_data.index[0]),
            pd.to_datetime(sale_data.index[-1]),
        )

        # nominal sales data
        og_sale_data = self.TR_OG.loc[store_nbr].set_index("date")
        og_sale_data.index = pd.to_datetime(og_sale_data.index)

        # transform the sale data
        sale_df = sale_data[sale_data.family == sale_data.iloc[0].family][["hol"]]
        sale_og_df = pd.DataFrame()
        # make a column for each family of product
        for d in self.families:
            # make column for sales, promo returns
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
            # make column for nominal sales to build labels
            sale_og_df = pd.concat(
                [
                    sale_og_df,
                    pd.DataFrame(
                        {f"{d}_sales": og_sale_data[og_sale_data.family == d].sales}
                    ),
                ],
                axis=1,
            )

        # fix the data on the missing dates
        sale_df = self.df_adjust_date(sale_df, start_date, end_date)
        sale_og_df = self.df_adjust_date(sale_og_df, start_date, end_date)
        trans_data = self.TS.loc[store_nbr].set_index("date")
        trans_data = self.df_adjust_date(trans_data, start_date, end_date)
        oil_data = self.df_adjust_date(self.O, start_date, end_date)

        # append other features (oil, transaction)
        sale_df = pd.concat([sale_df, oil_data], axis=1)
        sale_df = pd.concat([sale_df, trans_data], axis=1)
        # s_info = self.S.loc[(store_nbr)]
        # sale_df["city"], sale_df["cluster"], sale_df["type"] = (
        #     s_info.city,
        #     s_info.cluster,
        #     s_info.type,
        # )
        sale_df = self.z_series(sale_df)
        # combine the features into batch
        sample = torch.tensor(sale_df.to_numpy(), dtype=torch.float32).to(self.device)
        label_sample = torch.tensor(sale_og_df.to_numpy(), dtype=torch.float32).to(
            self.device
        )
        # slice the samples in the date range
        start_t, end_t = local_id, local_id + self.sample_seq_len

        # output the training data and label
        base_data = sample[start_t:end_t]
        t_sale_rets = sample[start_t:end_t][:, :66:2]
        base_data = torch.concat(
            (
                base_data,  # T sales, promo
                sample[start_t:end_t][:, 66:68],  # T oil, trans
                sample[start_t + self.INFER_DAYS : end_t + self.INFER_DAYS][
                    :, 1:66:2
                ],  # T+n promo
                sample[start_t + self.INFER_DAYS : end_t + self.INFER_DAYS][
                    :, 66:68
                ],  # T+n oil, trans
            ),
            axis=1,
        )

        label_t0 = label_sample[start_t:end_t].to(self.device)
        label = torch.zeros(self.sample_seq_len, 33 * self.INFER_DAYS).to(self.device)
        tgt_data = torch.zeros(self.sample_seq_len, 33 * self.INFER_DAYS).to(
            self.device
        )
        for i in range(
            1, self.INFER_DAYS + 1
        ):  # for every time step T, predict all family class from T+1 to T+16 (dim = 16*33)
            label_ti = label_sample[start_t + i : end_t + i].to(self.device)
            label_rets = self.get_log_ret_v2(label_t0, label_ti).to(
                self.device
            )  # sales ret columns
            label[:, (i - 1) * 33 : i * 33] = label_rets
            tgt_data[:, (i - 1) * 33 : i * 33] = t_sale_rets

        if self.is_train:
            return base_data, tgt_data, label
        else:
            return base_data, tgt_data, label_t0, store_nbr
