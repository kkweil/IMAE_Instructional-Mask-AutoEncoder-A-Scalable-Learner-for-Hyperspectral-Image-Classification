import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.nn import init
# utils
import math
import os
import datetime
import numpy as np
# from sklearn.externals
import joblib
from tqdm import tqdm

from common.datautils import *
from torch.utils.tensorboard import SummaryWriter


class Baseline(nn.Module):
    """
    Baseline network
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, dropout=False):
        super(Baseline, self).__init__()
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(input_channels, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, n_classes)

        self.apply(self.weight_init)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc3(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc4(x)
        return x


class HuEtAl(nn.Module):
    """
    Deep Convolutional Neural Networks for Hyperspectral Image Classification
    Wei Hu, Yangyu Huang, Li Wei, Fan Zhang and Hengchao Li
    Journal of Sensors, Volume 2015 (2015)
    https://www.hindawi.com/journals/js/2015/258619/
    """

    @staticmethod
    def weight_init(m):
        # [All the trainable parameters in our CNN should be initialized to
        # be a random value between −0.05 and 0.05.]
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            init.uniform_(m.weight, -0.05, 0.05)
            init.zeros_(m.bias)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels)
            x = self.pool(self.conv(x))
        return x.numel()

    def __init__(self, input_channels, n_classes, kernel_size=None, pool_size=None):
        super(HuEtAl, self).__init__()
        if kernel_size is None:
            # [In our experiments, k1 is better to be [ceil](n1/9)]
            kernel_size = math.ceil(input_channels / 9)
        if pool_size is None:
            # The authors recommand that k2's value is chosen so that the pooled features have 30~40 values
            # ceil(kernel_size/5) gives the same values as in the paper so let's assume it's okay
            pool_size = math.ceil(kernel_size / 5)
        self.input_channels = input_channels

        # [The first hidden convolution layer C1 filters the n1 x 1 input data with 20 kernels of size k1 x 1]
        self.conv = nn.Conv1d(1, 20, kernel_size)
        self.pool = nn.MaxPool1d(pool_size)
        self.features_size = self._get_final_flattened_size()
        # [n4 is set to be 100]
        self.fc1 = nn.Linear(self.features_size, 100)
        self.fc2 = nn.Linear(100, n_classes)
        self.apply(self.weight_init)

    def forward(self, x):
        # [In our design architecture, we choose the hyperbolic tangent function tanh(u)]
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = torch.tanh(self.pool(x))
        x = x.view(-1, self.features_size)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


def train_epoch(model, dataloader, optimzier, loss_fn):
    if torch.cuda.is_available():
        model.cuda()
    size = len(dataloader.dataset)
    correct = 0
    loss_ = []
    for batch, (X, y) in enumerate(dataloader):
        model.train()
        preds = model(X.cuda())
        loss = loss_fn(preds, y.type(torch.long).cuda())
        loss_.append(loss.item())
        optimzier.zero_grad()
        loss.backward()
        optimzier.step()
        model.eval()
        correct = correct + (model(X.cuda()).argmax(1) == y.cuda()).type(torch.float).sum().item()

        # print(f"loss: {loss.item():>7f}  [{batch * len(X):>5d}/{size:>5d}]")
    accuracy = (correct / size) * 100.0
    print(f"Avg Accuracy：{round(accuracy, 3)}%")
    torch.cuda.empty_cache()
    return loss_, accuracy


def test_epoch(model, dataloader, loss_fn):
    if torch.cuda.is_available():
        model.cuda()
    size = len(dataloader.dataset)
    correct = 0
    loss_ = []
    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            preds = model(X.cuda())
            loss =  loss_fn(preds, y.type(torch.long).cuda())
            loss_.append(loss.item())
            correct = correct + (model(X.cuda()).argmax(1) == y.cuda()).type(torch.float).sum().item()
            # if batch % 100 == 0:
            #     print(f"loss: {loss.item():>7f}  [{batch * len(X):>5d}/{size:>5d}]")
    accuracy = (correct / size) * 100.0
    print(f"Avg Accuracy：{round(accuracy, 3)}%")
    torch.cuda.empty_cache()
    return loss_, accuracy


if __name__ == '__main__':
    # inputs = torch.randn((20, 204, 15, 15))
    class_num = 16
    learning_rate = 1e-3
    epochs = 200
    batch_size = 32
    pretrain = True
    # load hsi data
    # imp = r'../data/Salinas_corrected.mat'
    # gtp = r'../data/Salinas_gt.mat'
    imp = r'../data/Indian_pines_corrected.mat'
    gtp = r'../data/Indian_pines_gt.mat'
    # imp = r'../data/PaviaU.mat'
    # gtp = r'../data/PaviaU_gt.mat'

    dataset = HSIloader(img_path=imp, gt_path=gtp, patch_size=15, sample_mode='ratio', train_ratio=0.01,
                        sample_points=30, merge=None, rmbg=True)
    dataset(spectral=True)

    train_dataset = HSIDataset(dataset.x_train_spectral, dataset.gt, dataset.coordinate_train)
    test_dataset = HSIDataset(dataset.x_test_spectral, dataset.gt, dataset.coordinate_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=3000, shuffle=True, pin_memory=True)
    # next(iter(train_dataloader))

    model = Baseline(input_channels=200, n_classes=class_num)
    if torch.cuda.is_available():
        model.cuda()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model_archive = f'clf_record/IN_nn_1'
    writer = SummaryWriter(model_archive)
    best_acc = 0
    history = {'train_loss': [], 'train_accuracy': [], 'test_loss': [], 'test_accuracy': []}
    for i in range(epochs):
        print(f'Epoch{i + 1}:-------------------------------')
        train_loss, train_accuracy = train_epoch(model, train_dataloader, optimizer, loss_fn)
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        writer.add_scalar('Loss/train', sum(train_loss) / len(train_loss), i)
        writer.add_scalar('Acc/train', train_accuracy, i)
        if (i + 1) % 20 == 0:
            test_loss, test_accuracy = test_epoch(model, test_dataloader, loss_fn)
            history['test_loss'].append(test_loss)
            history['test_accuracy'].append(test_accuracy)
            writer.add_scalar('Loss/valid', sum(test_loss) / len(test_loss), i)
            writer.add_scalar('Acc/valid', test_accuracy, i)
            if test_accuracy > best_acc:
                best_acc = test_accuracy
                torch.save(model.state_dict(), os.path.join(model_archive, 'best.pth'))

        torch.save(model.state_dict(), os.path.join(model_archive, 'last.pth'))
    a = 0
