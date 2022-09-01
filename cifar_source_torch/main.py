import torch
import torch.nn as nn
import numpy as np
from dataprocess import preprocess, preprocess2, CIFAR_Dataset
from torch.utils.data import DataLoader
from models import Lenet5, NormalCNN
from torchvision import transforms
from torchvision.models import vgg19_bn, resnet152
from torchvision.datasets import CIFAR10
from torch.optim import AdamW, Adam, SGD
import os

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def train(EPOCHS, lr, batch_size):
    transform_train = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.449), (0.226))
    ])
    transform_test = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.449), (0.226))
    ])

    train_dataloader = DataLoader(CIFAR10(root='./cifar-10-python/', train=True, transform=transform_train), shuffle=True, batch_size=batch_size, num_workers=4)
    test_dataloader = DataLoader(CIFAR10(root='./cifar-10-python/', train=False, transform=transform_test), batch_size=batch_size, num_workers=4)

    model = resnet152(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(2048, 10)
    # model = Lenet5()
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    max_test_acc = 0.
    for epoch in range(EPOCHS):
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch_train = batch[0].to(device)
            batch_label = batch[1].to(device)

            model.zero_grad()
            logits = model(batch_train)
            loss = loss_fn(logits, batch_label)

            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
            optimizer.step()

            if step % (len(train_dataloader) // 9) == 0:
                print("epoch: {} step: {}/{}".format(epoch, step, len(train_dataloader)))

        torch.save(model.state_dict(), './res/lenet5_last_1c.pt')

        # train_acc = evaluate(model, train_dataloader)
        test_acc = evaluate(model, test_dataloader)
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            torch.save(model.state_dict(), './res/lenet5_best_1c.pt')
            print("Best model saved!")

        print("epoch: {}  test_acc: {:.2f}%".format(epoch, test_acc * 100))


def evaluate(model, data_loader):
    model.eval()
    pred_tags = []
    true_tags = []
    with torch.no_grad():
        for batch in data_loader:
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

    return accuracy


if __name__ == '__main__':
    if not os.path.exists('./res'):
        os.mkdir('./res')
    train(100, 0.001, 128)
