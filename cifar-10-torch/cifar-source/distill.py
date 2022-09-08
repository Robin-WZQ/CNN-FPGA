import numpy as np
from models import Lenet5
from torchvision.models import resnet152
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def run(EPOCHS, lr, batch_size, alpha, T, resume=True):
    teacher_model = resnet152()
    teacher_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    teacher_model.fc = nn.Linear(2048, 10)
    teacher_model.to(device)
    teacher_model.load_state_dict(torch.load('./res/resnet152_best_1c.pt'))
    student_model = Lenet5().to(device)
    if resume:
        student_model.load_state_dict(torch.load('./res/distilled_lenet5_best.pt'))

    transform_teacher = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.449), (0.226)),
    ])
    transform_student = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4734), (0.2507)),
    ])
    transform_test = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.4734), (0.2507))
    ])
    torch.manual_seed(0)
    g = torch.Generator()
    train_dataloader_teacher = DataLoader(CIFAR10(root='./cifar-10-python/', train=True, transform=transform_teacher),
                                          shuffle=True, generator=g, batch_size=batch_size, num_workers=4)
    torch.manual_seed(0)
    g = torch.Generator()
    train_dataloader_student = DataLoader(CIFAR10(root='./cifar-10-python/', train=True, transform=transform_student),
                                          shuffle=True, generator=g, batch_size=batch_size, num_workers=4)
    test_dataloader = DataLoader(CIFAR10(root='./cifar-10-python/', train=False, transform=transform_test),
                                 batch_size=batch_size, num_workers=4)

    optimizer = Adam(student_model.parameters(), lr=lr)
    student_loss = nn.CrossEntropyLoss()
    distill_loss = nn.KLDivLoss(reduction='batchmean')

    teacher_model.eval()
    max_test_acc = 0.
    for epoch in range(EPOCHS):
        student_model.train()
        for step, (batch_tea, batch_stu) in enumerate(zip(train_dataloader_teacher, train_dataloader_student)):
            tea_data = batch_tea[0].to(device)
            stu_data = batch_stu[0].to(device)
            stu_label = batch_stu[1].to(device)

            with torch.no_grad():
                teacher_logits = teacher_model(tea_data)

            student_model.zero_grad()
            logits = student_model(stu_data)
            loss = alpha * student_loss(logits, stu_label) + (1 - alpha) * distill_loss(
                F.log_softmax(logits / T, dim=1), F.softmax(teacher_logits / T, dim=1))

            loss.backward()
            optimizer.step()

            if step % (len(train_dataloader_student) // 9) == 0:
                print("epoch: {} step: {}/{}".format(epoch, step, len(train_dataloader_student)))

        torch.save(student_model.state_dict(), './res/distilled_lenet5_last.pt')

        test_acc = evaluate(student_model, test_dataloader)
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            torch.save(student_model.state_dict(), './res/distilled_lenet5_best.pt')
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
    run(EPOCHS=100, lr=0.001, batch_size=128, alpha=0.3, T=7, resume=True)
