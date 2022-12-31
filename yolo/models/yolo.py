from torch import nn

class YOLOv1(nn.Module):
    def __init__(self, c=64):
        super(YOLO, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, stride=2, padding=1),
            nn.LeakyReLU(0.1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.LeakyReLU(0.1),
        )
        
    def forward(self, img):
        # conv layers
        h = self.conv1(img)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        # output through FC
        h = h.reshape(h.shape[0], -1)
        h = self.conn(h)
        h = h.reshape(h.shape[0], 30, 7, 7)
        
        return h
