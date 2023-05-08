import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from os import listdir
from os.path import join
from ipdb import set_trace
from PIL import Image
from numpy import array, transpose
from torch.utils.data import Dataset as DS


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


def get_classes(xml_pth):
    "Get class name of a xml"
    tree = ET.parse(xml_pth)
    names = tree.findall(".//object/name")

    return tuple(n.text for n in names)


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
        self.imgs = listdir(self.img_root)

        total_C = set()
        for f in listdir(self.ant_root):
            classes = get_classes(join(self.ant_root, f))
            total_C |= set(classes)
        if len(total_C) != 20:
            raise Exception("number of classes must be 20")
        # one-hot encode classes
        self.class_dict = dict((c, i) for i, c in enumerate(total_C))
        print("class dict: ", self.class_dict)

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
        classes = get_classes(ant_pth)
        pct_coords, obj_classes = [], []
        for i, coord in enumerate(coords):
            pct_coords.append(get_pct_coords(coord, img.size))
            # class encoding
            obj_classes.append(self.class_dict[classes[i]])
        img_arr = array(img.resize((448, 448)), copy=True) / 255.0  # normalize for RGB
        
        return transpose(img_arr, (2, 0, 1)), pct_coords, obj_classes
