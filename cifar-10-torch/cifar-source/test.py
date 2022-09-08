import torch
from torch.utils.data import DataLoader
from models import NormalCNN, Lenet5
from torchvision.models import resnet152
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
import torch.nn as nn

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def evaluate(model, data_loader):
    model.eval()
    pred_tags = []
    true_tags = []
    with torch.no_grad():
        for batch in data_loader:
            batch_data = batch['img'].to(device)
            batch_label = batch['label'].to(device)

            logits = model(batch_data)

            pred = torch.argmax(logits, dim=1).cpu().numpy()
            tags = batch_label.cpu().numpy()

            pred_tags.extend(pred)
            true_tags.extend(tags)

    assert len(pred_tags) == len(true_tags)
    correct_num = sum(int(x == y) for (x, y) in zip(pred_tags, true_tags))
    accuracy = correct_num / len(pred_tags)

    return accuracy


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.4734), (0.2507))
    ])
    test_dataloader = DataLoader(CIFAR10(root='./cifar-10-python/', train=False, transform=transform),
                                 batch_size=128)
    model = Lenet5()
    # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model.fc = nn.Linear(2048, 10)
    model.to(device)
    model.load_state_dict(torch.load('./res/distilled_lenet5_best.pt'))

    model.eval()
    pred_tags = []
    true_tags = []
    with torch.no_grad():
        for batch in test_dataloader:
            batch_data = batch[0].to(device)
            batch_label = batch[1].to(device)

            logits = model(batch_data)

            pred = torch.argmax(logits, dim=1).cpu().numpy()
            tags = batch_label.cpu().numpy()

            pred_tags.extend(pred)
            true_tags.extend(tags)

    assert len(pred_tags) == len(true_tags)
    correct_num = sum(int(x == y) for (x, y) in zip(pred_tags, true_tags))
    accuracy = correct_num / len(pred_tags)

    print(accuracy * 100)
