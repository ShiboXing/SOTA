import xml.etree.ElementTree as ET

def get_bboxes(xml_pth):
    
    tree = ET.parse(xml_pth)
    objs = tree.findall(".//object")
    dims = []
    for o in objs:
        bbox = o.findall(".//bndbox/*")
        dims.append([int(n.text) for n in bbox])
    
    return dims


