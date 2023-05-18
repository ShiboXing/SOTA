import torch
import pandas as pd
import os

from torch.utils.data import Dataset as DS

class Sales_Dataset(DS):
    def __init__(self, dir_pth):
        self.H = pd.read_csv(f"{dir_pth}/holidays_events.csv", index_col=False)
        self.O = pd.read_csv(f"{dir_pth}/oil.csv", index_col=False)
        self.S = pd.read_csv(f"{dir_pth}/stores.csv", index_col=False)
        self.TR = pd.read_csv(f"{dir_pth}/train.csv", index_col=False)
        self.TS = pd.read_csv(f"{dir_pth}/transactions.csv", index_col=False)
        self.dates = list(set(self.TR.date)) # use dates as index
    def __len__(self):
        return len(self.dates)
    
    def __getitem__(self, idx):
        date = self.dates[idx]
        return self.TR[self.TR.date == date], \
            self.TS[self.TS.date == date], \
            self.H[self.H.date == date], \
            self.O[self.O.date == date]