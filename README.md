## About SimpleGANTrainer.py

SimpleGANTrainer is an object designed to train, store, and evaluate simple GANs. That is to say, any GAN with one generator and one discriminator where the generator is directly trained from the output from the discriminator.
The Trainer object is designed to be able to train any GAN which meets this description, and automatically keep track of selected metrics during training so as to easily review the training process.
This tutorial will show how to create and train a simple GAN using the Trainer object, and will go over some ways to use the Trainer object’s functionality in more advanced ways.

## Setup

The Trainer object requires some setup before it can be created. In particular, it requires:

* Two pytorch models (the generator and the discriminator)
* The optimizer and loss function for each model
* A function to draw from the latent space
* A function to draw from the real data
* The device on which to train the models
* Optionally, the Trainer object can also take:
* A ToTrain object
* The desired positive result threshold for the discriminator, used for calculating certain metrics

### Environment Setup
The package requires pandas, numpy, matplotlib, and pytorch. To install pytorch, follow the directions at: https://pytorch.org/get-started/locally/

Numpy, pandas, and matplotlib can be installed via pip, but will also be installed automatically on installing the package. The package can be installed via pip by running the command: `pip install git+https://github.com/deoliveirajoshua/pytorch_GAN_package.git`
### Imports
First, we need to import the packages we plan on using.
```python
from GANTrainer.SimpleGANTrainer import SimpleGANTrainer
from GANTrainer.ToTrain import TwoFiveRule
import torch
import torch.nn as nn
import numpy as np
```


### Designing the GAN

Before we can train a GAN, we need to know what we want the GAN to do. For this tutorial, we will create an extremely simple GAN - the generator will generate 7-bit even binary numbers, and the discriminator will distinguish between even and odd 7-bit binary numbers.

### Models

The simplest possible generator which meets our requirements is a single layer consisting of 7 inputs and outputs and with a Sigmoid activation. This model is defined as follows:
```python
# Generator
class Generator(nn.Module):
		
 def __init__(self):
	 super(Generator, self).__init__()
	 self.dense_layer = nn.Linear(7, 7)
	 self.activation = nn.Sigmoid()
		
 def forward(self, x):
 	return self.activation(self.dense_layer(x))
```

Our discriminator is similarly simple. It consists of a single layer with 7 inputs and 1 output, again with a Sigmoid activation. It is defined as follows:

```python
# Discriminator
class Discriminator(nn.Module):
    	
 def __init__(self):
  super(Discriminator, self).__init__()
 	self.dense = nn.Linear(7, 1)
 	self.activation = nn.Sigmoid()

 def forward(self, x):
 	return self.activation(self.dense(x))
```

Finally, we create the model objects:

```python
# Model objects
gen = Generator()
dis = Discriminator()
```

The Trainer object stores the models in its models dictionary. Specifically, the models dictionary is of the form: `{“G”:generator, “D”:discriminator}`. The models are kept in training mode normally, though the discriminator is set to eval mode while training the generator, and the `eval(model, in_dat)` function sets the specified model to eval mode before evaluating, and returns it to train mode afterwards.

### Optimizers and Loss Functions

For our GAN we simply use built-in optimizers and loss functions:
```python
# built-in optimizers and loss functions:
gen_opt = torch.optim.Adam(gen.parameters(), lr=0.001)
dis_opt = torch.optim.Adam(dis.parameters(), lr=0.001)

gen_loss = nn.BCELoss()
dis_loss = nn.BCELoss()
```
However, any optimizers and loss functions which work in a pytorch training loop, including custom objects, work perfectly fine with the Trainer object.

### Latent Space and Dataset Functions

We now need functions to draw from the latent space and dataset. The latent space is the parameter space from which the generator draws. For our GAN, this is just a simple random tensor of size 7:
```python
# random tensor of size 7
def lat_space(batch_size, dev):
	return torch.randint(0, 2, size=(batch_size, 7), device=dev).float()
```
The dataset function is a function designed to return a batch of real data. Real data for us is just an even number, so it’s easier to generate data than retrieve it from a database.
```python
# dataset function
def batch_from_data(batch_size, dev):
	max_int = 128
	# Get the number of binary places needed to represent the maximum number
	max_length = int(math.log(max_int, 2))

	# Sample batch_size number of integers in range 0-max_int
	sampled_integers = np.random.randint(0, int(max_int / 2), batch_size)

 # create a list of labels - all ones because all numbers are even
	labels = [1] * batch_size

	# Generate a list of binary numbers for training.
	data = [list_from_num(int(x * 2)) for x in sampled_integers]
	data = [([0] * (max_length - len(x))) + x for x in data]

	return torch.tensor(data, device=dev).float()
 
 def list_from_num(num):
  return [int(x) for x in list(bin(num))[2:]]
```
Both the latent space and dataset functions take the parameters `(batch_size, device)`. The `batch_size` parameter determines the size of the batch, and the device parameter is the device on which the tensors are created.
The functions must output a tensor of the shape `(batch_size, input_size)` - the outputs of the latent space and dataset functions are passed directly into the generator and discriminator respectively.

### Device

The Trainer object supports training on any device visible to PyTorch. We want to train on the GPU, so we use:
```python
# train on GPU
GAN_DEVICE = "cuda"
```
If we do not have a GPU, we would use:
```python
# train without GPU
GAN_DEVICE = "cpu"
  ```
### ToTrain Object

ToTrain objects are objects designed to determine which model to train during the training process. The package comes with a number of built-in ToTrain objects, and they are designed to be as easy as possible to build your own custom ToTrain object.
Our GAN just uses the Two-Five Rule ToTrain object, which trains the generator for two epochs then trains the discriminator for five epochs.

### Discriminator Positive Threshold

The Trainer object allows the user to specify the threshold above which output from the discriminator is considered to be positive. This only impacts calculation of certain metrics (precision, recall, and false positive rate specifically), and does not affect training.
By default, this parameter is set to 0.5 if not specified. This is fine for our purposes, and so we do not set this parameter.

### Creating the Trainer Object

All that is left to do is to create the trainer object. This is done by:
```python
# trainer object creation
gan = SimpleGANTrainer(gen, dis, lat_space, batch_from_data, gen_loss, dis_loss, gen_opt, dis_opt, device, sw)
```
## Training the GAN

With our Trainer object created, we can now train it at will. To train a GAN, call the `.train(epochs, batch_size)` function:

```python
# call to train GAN
gan.train(7000, 16)
```
This will train the generator and discriminator according to the ToTrain object we specified. With the Two-Five Rule ToTrain object, this trains the generator for a total of 2,000 epochs and the discriminator for a total of 5,000 epochs.
Trainer objects can train for any length of time, across any number of different invocations of the `.train()` function. The function is blocking, though, so if we want to see output in the middle of training we must call the `.train()` function multiple times:

```python
gan.train(2000, 16)
gan.loss_by_epoch(“G”)  # Graphs the generator’s loss for the first 2000 epochs
gan.train(5000, 16)
```
The state of the ToTrain object is preserved across multiple calls to .train(), so 
```python
gan.train(2, 16)
gan.train(5, 16)
```
is equivalent to
```python
gan.train(7, 16)
```

## Evaluating the Models

The model objects can be directly accessed through the models dictionary. The Trainer object also has the `.eval(model, in_dat)` function, or the `.eval_generator(in_dat)` and `.eval_discriminator(in_dat)` functions (which just call `self.eval(“G”, in_dat)` and `self.eval(“D”, in_dat)` respectively).
To see output from the trained generator:

```python
# output from trained generator
print(gan.eval_generator(lat_space(16, device)))

# Equivalent to:
print(gan.eval("G", lat_space(16, device)))

# Or:
gan.models["G"].eval()
print(gan.models["G"](lat_space(16, device)))
gan.models["G"].train()
```

### Evaluating on a Different Device

The Trainer object supports moving the models to different devices, so it’s possible to use Trainer objects to train and evaluate models on different devices. Use the `.models_to(new_device)` function to send all models to the specified device.
To train the models on the GPU and evaluate on the CPU, for instance, we would:
```python
# evaluate on different device
GAN_DEVICE = "cuda"
gan = SimpleGANTrainer(gen, dis, lat_space, batch_from_data, gen_loss, dis_loss, gen_opt, dis_opt, device, sw)
gan.train(7000, 16)
print(gan.eval_generator(lat_space(16, device)))
gan.loss_by_epoch_d()
```

## Visualizing Training

### Loss by Epoch 

Shows a graph of the loss by epoch for the specified model. Called with `.loss_by_epoch(model_name) # “D” or “G”`

### Divergence by Epoch

Shows a graph of the Wasserstein distance of the generator per epoch. Called with `.divergence_by_epoch()`

### Epochs Trained

Returns the total number of epochs which the specified model was trained. Called with `.epochs_trained(model)`

## Saving and Loading
Trainer objects can save and load checkpoints by using the `soft_save(path)` and `soft_load(path)` functions respectively. These functions use pickle - **only attempt to load checkpoints from a trusted source.**

### Loading a Checkpoint
Trainer objects save model and optimizer states using the built-in pytorch functions for saving state dictionaries. This means that it is only possible to load the models and optimizers if their class definitions are already known to the Trainer object.
This is easily done if the Trainer object is the same object which saved the checkpoint:
```python
gan.soft_save(PATH)
...
gan.soft_load(PATH)
```
If loading a checkpoint as a new object, it is necessary to define the Trainer object with the correct models and optimizers.
```python
gan1.soft_save(PATH)
...
gen = Generator()
dis = Discriminator()
g_opt = torch.optim.Adam(gen.parameters(), lr=0.001)
d_opt = torch.optim.Adam(dis.parameters(), lr=0.001)
gan2 = SimpleGANTrainer(gen, dis, lat_space, batch_from_data, None, None, g_opt, d_opt, “cuda”, None)
```
