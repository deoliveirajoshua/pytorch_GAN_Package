from src.GANTrainer.ConditionalGANTrainer import ConditionalGANTrainer
from src.GANTrainer.ToTrain import TwoFiveRule
import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import Data.Spotify.GetSpotifyData as GetSpotifyData


def lat_space(batch_size, dev="cpu"):
    data = torch.rand(batch_size, num_inputs, device=dev)
    labels = torch.randint(0, num_classes, size=(batch_size, 1), device=dev)
    labels = func.one_hot(labels, num_classes=num_classes)
    labels = labels.reshape(batch_size, num_classes)
    return torch.cat((data, labels), 1)


def batch_from_data(batch_size=16, dev="cpu", label=-1):
    # check for labels or not
    if label == -1:
        real_data = example_data
        real_labels = example_labels
    else:
        real_data = example_data[np.where(example_labels[:] == label)]
        real_labels = example_labels[np.where(example_labels[:] == label)]

    # Obtain some entries
    num_rows = real_data.shape[0]
    rand_indices = np.random.choice(num_rows, size=batch_size, replace=False)
    data = real_data[rand_indices, :]

    # Get related labels
    labels = real_labels[rand_indices]
    labels = to_one_hot(labels)

    # Combine data and labels
    data_labels = np.concatenate((data, labels), axis=1).astype(float)

    return torch.tensor(data_labels, device=dev)


def to_one_hot(labels):
    oneHot = np.zeros((labels.shape[0], num_classes)).astype(int)
    oneHot[np.arange(labels.size), labels.astype(int)] = 1
    return oneHot


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.embedding_label = nn.Embedding(num_classes, 1)
        self.dense_layer = nn.Linear(num_inputs + num_classes, num_inputs)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        data = x[:, 0:num_inputs]
        labels = x[:, num_inputs:].int()
        gan_batch_size = labels.size(dim=0)
        gen_input = torch.cat((data, self.embedding_label(labels).reshape(gan_batch_size, num_classes)), 1)
        new_data = self.activation(self.dense_layer(gen_input))
        return torch.cat((new_data, labels), 1)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.embedding_label = nn.Embedding(num_classes, 1)
        self.dense = nn.Linear(num_inputs + num_classes, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        data = x[:, 0:num_inputs]
        labels = x[:, num_inputs:].int()
        gan_batch_size = labels.size(dim=0)
        dis_input = torch.cat((data, self.embedding_label(labels).reshape(gan_batch_size, num_classes)), 1)
        return self.activation(self.dense(dis_input.float()))


def print_output(out):
    out_data = out[:, 0:num_inputs]
    out_labels = out[:, num_inputs:].int()
    out_labels = torch.argmax(out_labels, dim=1)
    print(out_data.detach().numpy() * maxDataValues)
    print(out_labels)


def get_gen_input(lableNum, batch_size, dev="cpu"):
    data = torch.rand(batch_size, num_inputs, device=dev)
    labels = torch.from_numpy(np.ones(batch_size).astype(int) * lableNum)
    labels = func.one_hot(labels.to(torch.int64), num_classes=num_classes)
    labels = labels.reshape(batch_size, num_classes)
    return torch.cat((data, labels), 1)


# Data imports
example_data, example_labels, maxDataValues = GetSpotifyData.spotify_data()
num_inputs = np.shape(example_data)[1]
num_classes = len(np.unique(example_labels))

gen = Generator()
dis = Discriminator()

gen_opt = torch.optim.Adam(gen.parameters(), lr=0.001)
dis_opt = torch.optim.Adam(dis.parameters(), lr=0.001)

gen_loss = nn.BCELoss()
dis_loss = nn.BCELoss()

sw = TwoFiveRule()
device = input("Would you like to run this test on cpu or cuda? Type the one you wish to use: ")

gan = ConditionalGANTrainer(gen, dis, lat_space, batch_from_data, gen_loss, dis_loss, gen_opt, dis_opt,
                            device, sw, num_input_variables=num_inputs, classes=num_classes, do_wass_viz=True)
epochs = 7
gan.train(epochs, 16)
output = gan.eval_generator(lat_space(16, device))
print_output(output.to("cpu"))

gan.divergence_by_epoch()

gan.loss_by_epoch_d()

# Split data into training and testing
# Do a scipy linear or logistic regression
# Train on real, test on real
# Train on fake, test on real
# Train on real+fake, test on real
