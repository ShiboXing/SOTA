import os
import torch
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import Dataset as DS
from PIL import Image


class OID_Dataset(DS):
    def __init__(self, root, label_file, classes, dtype="TRAIN"):
        super(DS, self).__init__()
        self.__root__ = root
        self.fnames = os.listdir(f"{root}/IMAGES/{dtype}/DATA/")
        self.labels = pd.read_csv(f"{root}/ANNOT/{label_file}")
        self.labels = self.labels[self.labels.LabelName.isin(classes)]
        self.data_type = dtype

        self.trans_func = transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        # assign an integer to each class name
        self.classes = dict([(b, a) for a, b in enumerate(classes)])
        self.labels.LabelName = self.labels.LabelName.replace(self.classes)

    def __getitem__(self, ind):
        _id = self.fnames[ind].split(".")[0]
        img_pth = f"{self.__root__}/IMAGES/{self.data_type}/DATA/{_id}.jpg"
        img = Image.open(img_pth).convert("RGB")
        img = self.trans_func(img)
        label = self.labels[self.labels.ImageID == _id].iloc[:, 2:8]
        return img, torch.tensor(label.values)

    def __len__(self):
        return len(self.fnames)
