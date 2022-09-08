import torch.nn as nn


class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5, self).__init__()
        # 32, 32, 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        # 14, 14, 6
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        # 5, 5, 16
        self.fc1 = nn.Sequential(
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(84, 10)

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.fc1(x.view(x.shape[0], -1))
        x = self.fc2(x)
        logits = self.classifier(x)

        return logits


class NormalCNN(nn.Module):
    def __init__(self):
        super(NormalCNN, self).__init__()
        # (bs, 32, 32, 1)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # (bs, 28, 28, 32)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )
        # (bs, 12, 12, 32)
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # (bs, 10, 10, 64)
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )
        # bs, 4, 4, 64 -> bs, 1024
        self.layer5 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU()
        )
        self.classifier = nn.Linear(256, 10)

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.shape[0], -1)
        x = self.layer5(x)
        logits = self.classifier(x)

        return logits


class NormalCNN2(nn.Module):
    def __init__(self):
        super(NormalCNN2, self).__init__()
        # (bs, 32, 32, 1)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # (bs, 28, 28, 32)
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # nn.Dropout(0.2)
        )
        # (bs, 12, 12, 32)
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # (bs, 10, 10, 64)
        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Dropout(0.2)
        # )
        # bs, 4, 4, 64 -> bs, 1024
        # self.layer5 = nn.Sequential(
        #     nn.Linear(1024, 256),
        #     nn.ReLU()
        # )
        self.classifier = nn.Linear(4096, 10)

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        x = x.view(x.shape[0], -1)
        # x = self.layer5(x)
        logits = self.classifier(x)

        return logits
