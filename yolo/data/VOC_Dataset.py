import xml.etree.ElementTree as ET

from os.path import join
from subprocess import run
from torch.utils.data import Dataset as DS

def get_bboxes(xml_pth):
    tree = ET.parse(xml_pth)
    objs = tree.findall(".//object")
    dims = []
    for o in objs:
        bbox = o.findall(".//bndbox/*")
        dims.append([int(n.text) for n in bbox])
    
    return dims

class VOC_Dataset(DS):
    def __init__(self, voc_root: str):
        self.img_root = join(voc_root, "JPEGImages")
        self.ant_root = join(voc_root, "Annotations")

        res = run(f"ls {self.img_root}", shell=True, check=True, capture_output=True)
        self.imgs = res.stdout.decode("utf-8").split("\n")
        
    def __getitem__(self, ind):
        img_file = self.imgs[ind]
        img_id = img_file.split(".")[0]
        label_file = f"{img_id}.xml"
        
        return (
            join(self.img_root, img_file),
            join(self.ant_root, label_file)
        )