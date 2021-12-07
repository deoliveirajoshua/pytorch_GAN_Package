import SuperTrainer
import ToTrain
import torch
import math
import torch.optim as optim
import matplotlib.pyplot as plt


class WassersteinGANTrainer(SuperTrainer.SuperTrainer):
    def __init__(self, generator, discriminator, latent_space_function, random_from_dataset, g_loss, d_loss, g_lr, d_lr, device, tt=None, d_thresh=0.5):
        """Class to train a Wasserstein GAN.
        Generator and discriminator are torch model objects
        Latent_space_function(n) is a function which returns an array of n points from the latent space
        Random_from_dataset is a function which returns an array of n points from the real dataset"""
        g_opt = optim.RMSprop(generator.parameters(), g_lr)
        d_opt = optim.RMSprop(discriminator.parameters(), d_lr)

        if tt is None:
            self.totrain = ToTrain.TwoFiveRule()
        else:
            self.totrain = tt
        self.dataset = random_from_dataset
        self.latent_space = latent_space_function
        SuperTrainer.SuperTrainer.__init__(self, tt, models={"G": generator.to(device), "D": discriminator.to(device)},
                                           in_functions={"G": self.generator_input,
                                                         "D": self.discriminator_input},
                                           loss_functions={"G": g_loss, "D": d_loss},
                                           opts={"G": g_opt, "D": d_opt})
        self.stats["losses"] = {"G": [], "D": []}
        self.stats["epochs_trained"] = {"G": 0, "D": 0}

        self.stats["d_fpr"] = []
        self.stats["d_recall"] = []
        self.stats["d_precision"] = []

        self.d_thresh = d_thresh
        self.device = device

    def train(self, n_epochs, n_batch):
        all_dists = []
        for epoch in range(n_epochs):
            tt = self.totrain.next(self)  # Determine which model to train - sw will either be "D" or "G"

            # Both input functions return the tuple (dis_in, labels)
            # generator_in returns (gen_out, labels) - this data is passed through D and used to train G
            # discriminator_in returns (dis_in, labels) - this is used to train D directly
            # For other GAN types: input functions can return whatever makes the most sense for your specific type of GAN
            # (so controllable GAN, for instance, might want to return a classification vector as well)
            dis_in, y = self.in_functions[tt](n_batch)
            if tt == "G":  # If we're training the generator, we should temporarily put the discriminator in eval mode
                self.models["D"].eval()
            mod_pred = self.models["D"](dis_in)
            self.models["D"].train()
            mod_loss = self.loss_functions[tt](mod_pred, y)

            # Logging for visualizers
            self.stats["losses"][tt].append(mod_loss.item())
            self.stats["epochs_trained"][tt] += 1

            y_flat = y.cpu().numpy().flatten()  # Calculate fPr, recall, precision
            mod_pred_flat = mod_pred.cpu().detach().numpy().flatten()
            fP = 0
            fN = 0
            tP = 0
            tN = 0
            for i in range(len(y_flat)):
                if y_flat[i] == 0:
                    if mod_pred_flat[i] > self.d_thresh:
                        fP += 1
                    else:
                        tN += 1
                else:
                    if mod_pred_flat[i] > self.d_thresh:
                        tP += 1
                    else:
                        fN += 1

            if fP + tN > 0:
                self.stats["d_fpr"].append(fP / (fP + tN))
            if tP + fP > 0:
                self.stats["d_precision"].append(tP / (tP + fP))
            if tP + fN > 0:
                self.stats["d_recall"].append(tP / (tP + fN))

            # Pytorch training steps
            self.optimizers[tt].zero_grad()
            mod_loss.backward()
            self.optimizers[tt].step()

            if tt == "D":
                for p in self.models["D"].parameters():
                    p.data.clamp_(-0.01, 0.01)


            # TODO: ???
            w_dists = self.all_Wasserstein_dists(self.eval_generator(self.latent_space(256)), self.dataset(256))
            w_dist_mean = torch.mean(w_dists)
            all_dists.append(w_dist_mean)
            #print(w_dist_mean)
        print(len(all_dists))
        plt.title('Wasserstein GAN Training Over Time')
        plt.xlabel('Batches')
        plt.ylabel('Wasserstein Distance Mean')
        plt.plot(all_dists)
        plt.show()

    def eval_generator(self, in_dat):
        return self.eval("G", in_dat)

    def eval_discriminator(self, in_dat):
        return self.eval("D", in_dat)

    def get_g_loss_fn(self):
        return self.loss_functions["G"]

    def get_g_opt_fn(self):
        return self.optimizers["G"]

    def get_d_loss_fn(self):
        return self.loss_functions["D"]

    def get_d_opt_fn(self):
        return self.optimizers["D"]

    def loss_by_epoch_g(self):
        self.loss_by_epoch("G")

    def loss_by_epoch_d(self):
        self.loss_by_epoch("D")

    def discriminator_input(self, n_batch):
        gen_in = self.latent_space(math.ceil(n_batch / 2), self.device)
        self.models["G"].eval()
        gen_out = self.models["G"](gen_in)
        self.models["G"].train()
        dis_in = torch.cat((gen_out, self.dataset(int(n_batch / 2), self.device)))
        y = torch.tensor([[0] for n in range(math.ceil(n_batch / 2))] + [[1] for n in range(int(n_batch / 2))], device=self.device).float()  # TODO: used .float() here because the model I'm using to test uses floats. Find a way to automatically find the correct data type
        return dis_in, y

    def generator_input(self, n_batch):
        gen_in = self.latent_space(n_batch, self.device)
        gen_out = self.models["G"](gen_in)
        y = torch.tensor([[1] for n in range(n_batch)], device=self.device).float()
        return gen_out, y
