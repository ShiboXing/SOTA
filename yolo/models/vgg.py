from torch import nn

class VGG(nn.Module):
    def __init__(self, c=64):
        super(VGG, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, c, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(c, c, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=0),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c*2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(c*2, c*2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=0),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(c*2, c*4, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(c*4, c*4, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(c*4, c*4, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=0),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(c*4, c*8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(c*8, c*8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(c*8, c*8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=0),
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(c*8, c*8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(c*8, c*8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(c*8, c*8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=0),
        )
        
        self.conn = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(),
            nn.Linear(4096, 1470),
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
    
class RPN(nn.Module):
    def __init__(self, c=512, ratio=[1.0, 2.0, 0.5]):
        super(RPN, self).__init__()
#         self.feat = nn.Conv2d(c, c/2, 3, stride=1, padding=1)
#         self.objness = nn.Conv2d()
    def forward(self, feat):
        pass