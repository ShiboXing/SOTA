from models import YOLOv1

import torch

test = torch.randn(3, 448, 448)
yolo = YOLOv1()
res = yolo(test)