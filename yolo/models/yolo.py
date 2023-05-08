from torch import nn
from ipdb import set_trace


class YOLOv1(nn.Module):
    def __add_relu__(self, conv: list):
        res = []
        for l in conv:
            res.append(l)
            res.append(nn.LeakyReLU(negative_slope=0.1))

        return res

    def __init__(self, c=64):
        super(YOLOv1, self).__init__()

        conv1 = [
            nn.Conv2d(3, 64, 7, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2, stride=2, padding=1),
        ]

        conv2 = [
            nn.Conv2d(64, 192, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2, stride=2, padding=0),
        ]

        conv3 = [
            nn.Conv2d(192, 128, 1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 256, 1, padding=0),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 512, 3, padding=0),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2, stride=2),
        ]

        conv4 = [
            nn.Conv2d(512, 256, 1, padding=1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 256, 1, padding=1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 256, 1, padding=1),
            nn.Conv2d(256, 512, 3, padding=0),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 256, 1, padding=0),
            nn.Conv2d(256, 512, 3, padding=0),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 512, 1, padding=0),
            nn.Conv2d(512, 1024, 3, padding=0),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2, stride=2),
        ]

        conv5 = [
            nn.Conv2d(1024, 512, 1, padding=1),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(1024, 512, 1, padding=1),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(1024, 1024, 3),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(1024, 1024, 3, stride=2),
            nn.LeakyReLU(negative_slope=0.1),
        ]

        conv6 = [
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
        ]

        fc1 = [
            nn.Linear(50176, 4096),
            nn.Dropout(),
            nn.LeakyReLU(negative_slope=0.1),
        ]

        fc2 = [nn.Linear(4096, 1470)]

        self.conv1 = nn.Sequential(*conv1)  # nn.Sequential(*self.__add_relu__(conv1))
        self.conv2 = nn.Sequential(*conv2)  # nn.Sequential(*self.__add_relu__(conv2))
        self.conv3 = nn.Sequential(*conv3)  # nn.Sequential(*self.__add_relu__(conv3))
        self.conv4 = nn.Sequential(*conv4)  # nn.Sequential(*self.__add_relu__(conv4))
        self.conv5 = nn.Sequential(*conv5)  # nn.Sequential(*self.__add_relu__(conv5))
        self.conv6 = nn.Sequential(*conv6)  # nn.Sequential(*self.__add_relu__(conv6))
        self.fc1 = nn.Sequential(*fc1)  # nn.Sequential(*self.__add_relu__(fc1))
        self.fc2 = nn.Sequential(*fc2)  # nn.Sequential(*self.__add_relu__(fc2))

    def forward(self, img):
        # conv layers
        h1 = self.conv1(img)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h4 = self.conv4(h3)
        h5 = self.conv5(h4)
        h6 = self.conv6(h5)
        h7 = self.fc1(h6.reshape(h6.shape[0], 1, -1))
        h8 = self.fc2(h7.reshape(h7.shape[0], 1, -1))

        return h8.reshape(h8.shape[0], 30, 7, 7)
