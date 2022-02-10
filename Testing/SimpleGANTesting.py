from src.GANTrainer.SimpleGANTrainer import SimpleGANTrainer
from src.GANTrainer.ToTrain import TwoFiveRule
import torch
import torch.nn as nn
import math
import numpy as np


def lat_space(batch_size, dev):
    return torch.randint(0, 2, size=(batch_size, 7), device=dev).float()


def list_from_num(num):
    return [int(x) for x in list(bin(num))[2:]]


def batch_from_data(batch_size, dev):
    max_int = 128
    # Get the number of binary places needed to represent the maximum number
    max_length = int(math.log(max_int, 2))

    # Sample batch_size number of integers in range 0-max_int
    sampled_integers = np.random.randint(0, int(max_int / 2), batch_size)

    # create a list of labels all ones because all numbers are even
    labels = [1] * batch_size

    # Generate a list of binary numbers for training.
    data = [list_from_num(int(x * 2)) for x in sampled_integers]
    data = [([0] * (max_length - len(x))) + x for x in data]

    return torch.tensor(data, device=dev).float()


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.dense_layer = nn.Linear(7, 7)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense_layer(x))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dense = nn.Linear(7, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense(x))


gen = Generator()
dis = Discriminator()

gen_opt = torch.optim.Adam(gen.parameters(), lr=0.001)
dis_opt = torch.optim.Adam(dis.parameters(), lr=0.001)

gen_loss = nn.BCELoss()
dis_loss = nn.BCELoss()

sw = TwoFiveRule()

device = input("Would you like to run this test on cpu or cuda? Type the one you wish to use: ")

gan = SimpleGANTrainer(gen, dis, lat_space, batch_from_data, gen_loss, dis_loss, gen_opt, dis_opt, device, sw)
gan.train(7000, 16)
print(gan.eval_generator(lat_space(16, device)))
gan.loss_by_epoch_d()

