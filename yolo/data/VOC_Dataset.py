import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from os.path import join
from subprocess import run
from PIL import Image
from numpy import array, transpose
from torch.utils.data import Dataset as DS, DataLoader as DL


def get_bboxes(xml_pth):
    """
    get the coordinates from the xml in xml_pth

    return Value:
    dims -- xmin, ymin, xmax, ymax

    """
    tree = ET.parse(xml_pth)
    objs = tree.findall(".//object")
    dims = []

    for o in objs:
        bbox = o.findall(".//bndbox/*")
        dims.append([int(float(n.text)) for n in bbox])
    return dims


def get_pct_coords(bbox, img_dims):
    """Keyword arguments:

    bbox -- (xmin, ymin, xmax, ymax)
    img_dims -- (xsize, ysize)
    """
    return (
        float(bbox[0]) / img_dims[0],
        float(bbox[1]) / img_dims[1],
        float(bbox[2]) / img_dims[0],
        float(bbox[3]) / img_dims[1],
    )


class VOC_Dataset(DS):
    def __init__(self, voc_root: str):
        self.img_root = join(voc_root, "JPEGImages")
        self.ant_root = join(voc_root, "Annotations")

        res = run(f"ls {self.img_root}", shell=True, check=True, capture_output=True)
        self.imgs = res.stdout.decode("utf-8").strip().split("\n")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, ind):
        """
        Return the resized image and its resized percentage coordinates
        """
        img_file = self.imgs[ind]
        img_id = img_file.split(".")[0]
        label_file = f"{img_id}.xml"
        img_pth, ant_pth = join(self.img_root, img_file), join(
            self.ant_root, label_file
        )
        
        img = Image.open(img_pth)
        coords = get_bboxes(ant_pth)
        pct_coords = []
        for coord in coords:
            pct_coords.append(get_pct_coords(coord, img.size))
        res_img = img.resize((448, 448))

        return transpose(array(res_img, copy=True), (2, 0, 1)), pct_coords
