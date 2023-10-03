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
        np.seterr(divide="ignore")  # suppress divide by zero
        rets = np.log10(df[y_col] / df[y_col].shift(1))
        np.seterr(divide="warn")

        return Sales_Dataset.df_fix_float(rets)

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
        self.TR.set_index("store_nbr", inplace=True)
        self.TR.drop("id", axis=1, inplace=True)
        self.TS.sort_values(["store_nbr", "date"], inplace=True)
        self.TS.set_index(["store_nbr"], inplace=True)
        self.O.sort_values(["date"], inplace=True)
        self.O.set_index(["date"], inplace=True)
        self.O.index = pd.to_datetime(self.O.index)
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
        self.O = self.O.asfreq("D").interpolate()
        self.O.dcoilwtico = self.get_log_ret(self.O, "dcoilwtico", ["date"])

        # construct the primary key
        self.ids = sorted(list(set(self.S.index)))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """Sample dimension: per (date, store_nbr):
        sequence_length * (family(N) + oil(1) + transaction(1) + city(1) + cluster(1) + type(1))

        L * (33*2 + 2 + 3)
        """

        # prepare to build batch
        store_nbr = self.ids[idx]
        sale_data = self.TR.loc[store_nbr].set_index("date")
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
        for d in set(sale_data.family):
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
        sale_df = pd.concat(
            [sale_df, oil_data],
            axis=1,
        )
        sale_df = pd.concat(
            [sale_df, trans_data],
            axis=1,
        )

        # combine the features into batch
        sample = torch.zeros(
            sale_df.shape[0], sale_df.shape[1] + 3, dtype=torch.float32
        ).cuda()
        s_data = self.S.loc[(store_nbr)].to_numpy()
        sample[:, : sale_df.shape[1]] = torch.tensor(
            sale_df.to_numpy(), dtype=torch.float32
        )
        sample[:, -3:] = torch.tensor(s_data, dtype=torch.float32)

        return (
            sample[:-1],
            torch.tensor(sale_df.filter(like="sales").to_numpy(), dtype=torch.float32)[
                1:
            ],
        )

    # sample[:, :2] = torch.tensor(sale_data[["sales", "onpromotion"]].to_numpy())
    # sample[:, 2] = torch.tensor(oil_data)
    # sample[:, 3] = torch.tensor(trans_data)
    # sample[:, 4:] = torch.tensor(s_data)

    # return sample.reshape(1, -1)
