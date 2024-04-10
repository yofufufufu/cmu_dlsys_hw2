import sys

import numpy

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim)
            )
        ),
        nn.ReLU()
    )
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    modules = [nn.Linear(dim, hidden_dim), nn.ReLU()]
    for i in range(num_blocks):
        # 注意ResidualBlock是dim in, dim out的
        modules.append(ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob))
    modules.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*modules)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_func = nn.SoftmaxLoss()
    total_loss = 0
    error_num = 0
    if opt:
        model.train()
        for index, batch_data in enumerate(dataloader):
            opt.reset_grad()
            logits = model(batch_data[0])
            loss = loss_func(logits, batch_data[1])
            total_loss += loss.numpy()
            error_num += (logits.numpy().argmax(axis=1) != batch_data[1].numpy()).sum()
            loss.backward()
            opt.step()
    else:
        model.eval()
        for index, batch_data in enumerate(dataloader):
            logits = model(batch_data[0])
            loss = loss_func(logits, batch_data[1])
            total_loss += loss.numpy()
            error_num += (logits.numpy().argmax(axis=1) != batch_data[1].numpy()).sum()
    return error_num / len(dataloader.dataset), total_loss / (index + 1)
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(data_dir + "/train-images-idx3-ubyte.gz", data_dir + "/train-labels-idx1-ubyte.gz")
    test_dataset = ndl.data.MNISTDataset(data_dir + "/t10k-images-idx3-ubyte.gz", data_dir + "/t10k-labels-idx1-ubyte.gz")
    training_dataloader = ndl.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = ndl.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    input_dim = train_dataset[0][0].shape[0]
    model = MLPResNet(dim=input_dim, hidden_dim=hidden_dim)
    opt = optimizer(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        training_avg_error_rate, training_avg_loss = epoch(training_dataloader, model=model, opt=opt)
    test_avg_error_rate, test_avg_loss = epoch(test_dataloader, model=model, opt=None)
    # notebook说返回acc, 实际上应该是返回错误的概率，测试才能通过
    return training_avg_error_rate, training_avg_loss, test_avg_error_rate, test_avg_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
