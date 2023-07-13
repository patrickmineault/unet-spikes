# Unet-spikes

Tutorial repository for the Cajal school in machine learning for neuroscience. 

The core of the repo showcases a UNet that infers a latent space on spikes. The idea is to train a UNet to perform masked language modeling (MLM) on binned spike trains. This is conceptually similar to the Neural Data Transformer (NDT) from Ye and Pandarinath (2021), itself taking its cues from BERT. This replaces the transformer with the more stable training inherent in a UNet. 

The UNet performs a combination of local smoothing and mixing data from different channels (i.e. nonlinear PCA) to smooth spike trains. It learns the statistics of ensembles of spike trains towards this end.

## Getting started

* Fork this repository
* Clone your fork
* Create a fresh conda environment with e.g. Python 3.9, `conda create --name unet-spikes python=3.9`
* `cd` into the directory and `pip install -e .`
* `pip install -r requirements.txt`

### Copy the data

Use an automated tool like `wget`:

```
wget -P data/h5 https://cajal-data-740441.s3.eu-west-3.amazonaws.com/lfads_lorenz.h5
wget -P data/h5 https://cajal-data-740441.s3.eu-west-3.amazonaws.com/chaotic_rnn_no_inputs_dataset_N50_S50.h5
```

Note that if wget is not installed on your system, you can install it via:

```
conda install -c anaconda wget
```

Alternatively, download manually [1](https://cajal-data-740441.s3.eu-west-3.amazonaws.com/chaotic_rnn_no_inputs_dataset_N50_S50.h5
) and [2](https://cajal-data-740441.s3.eu-west-3.amazonaws.com/lfads_lorenz.h5) and put it into the `data/h5` folder.

## Training a model

`cd` into scripts and use `python train.py` to train the model.

## Exercises

### Debugging the model

0. `cd` into the scripts folder and train the model using `python train.py`. Note: you can exit the run early using Ctrl+C.
1. Load up tensorboard and visualize a training run. Tensorboard can be pulled up via:

```
tensorboard --logdir=runs
```

How does it look? Based on the graphs, do you think that the network is learning something meaningul?

2. Add visualization for the model outputs and the model targets. The model is trained on spike data, but because this is simulation data, we have access to the underlying rates. Log the following information to Tensorboard: `preds`, `target`, `the_mask` and `rates`. It's sufficient to log the last examplar from the last training batch. Use functions such as `logger.add_image` or `logger.add_figure` to log images and figures to Tensorboard in `train.py`. How do they look? Hint: a good place to write this code between the train loop and the validate loop in `train.py`. 

<details>
  <summary>Hint if you're stuck</summary>
  Calling tensorboard's `log_image` function allows you to write an image. Try adding this line after the train loop:

  ```
  logger.add_image('debug/preds', preds[-1], total_epoch, dataformats='HW')
  ```

  dataformats='HW' is necessary because each prediction has the shape of an image that is Height x Width, and there is only one such prediction, hence there is no "channel" dimension. 

  Do the same for `target`, `the_mask` and `rates`.
  
</details>


3. One big issue with the model is that it can give negative rates. This is a problem because rates are inherently positive. Add a nonlinearity to the model to ensure that the rates are positive. Look inside the CNN module (`src/cnn.py`) and add a nonlinearity to the output of the last layer. You can use `torch.nn.ReLU` or `torch.nn.Softplus` for this. How does this change the training? How does this change the predictions? How does this change the validation R2?


### Parameter sweep

1. Let's tweak the learning rate. Try a grid of different learning rates, say from 1e-4 to 1e1 in logarithmic space. Look at the loss function and the validation R2. What is a good learning rate? Is the model training very sensitive to the learning rate?

### Visualizing the middle layers

1. How does the network do its magic? Let's visualize the weights of the model. Create a jupyter notebook. Load up a trained checkpoint, which should be saved in 'scripts/runs'. For example:

```
import torch
from src.cnn import CNN

cnn = CNN(29, 10)
cnn.load_state_dict(torch.load("../scripts/runs/Jul12_03-25-21_AP-T-020.local/model.pt"))
cnn.eval()
```

Visualize the embedding layer. For example:

```
import matplotlib.pyplot as plt
plt.imshow(cnn.embedding.weight.data.squeeze())
```

Do the same with the unembedding layer. Can you see any structure in the weights? What do you think is going on?

2. Visualize the weights of the first convolutional layer. You are going to need to look at slices of the weights, as they actually contain three different indices. For example, you might start by looking at `cnn.smoothing.weight.data[0, :, :].squeeze()`. Can you see anything in there?

3. (optional) Sometimes intermediate layers can be difficult to visualize even when their output is meaningful. Load up a trained checkpoint and visualize the output of the first convolutional layer. You will need to load up some data and run it through the model. For example:

```
from src.dataset import SpikesDataset
dataset = SpikesDataset("../data/config/lorenz.yaml")

# First examplar
spikes, rates, _, _ = dataset[0]

# Run the spikes through part of the network
X = spikes.unsqueeze(0).to(torch.float32)
X = cnn.embedding(X)
X = cnn.smoothing(X)
plt.imshow(X.squeeze().detach().numpy())
```

What does this tell you about how the network is doing its job?

### Improving the model

1. Perhaps the model is too low capacity and it needs more layers to learn some interesting features. Add one more set of layers to the model. You can do this by adding `smoothing1`, `bn1` and `relu1` layers in the initialization of `cnn.py`, and forwarding your inputs through them in the `forward` function. How does this change the training? How does this change the predictions? How does this change the validation R2? Note: you need to create separate layers and not reuse the old ones, because otherwise the weights will be tied in the first convolutional layer and the second one.
2. We made higher capacity, but we now we have a bit of a mess on our hands! It would be better if the model was more modular. Time to refactor! Add a `Smoothing` layer class inside of `cnn.py` that wraps `smoothing`, `bn` and `relu` operations. It should be a subclass of `nn.Module`. Run the training again to make sure you didn't break anything! Stretch goal: test the CNN from the test module in `tests/test_cnn.py`. You can run this from the command line via `pytest test_cnn.py` or via the `test`. Write another test that checks that the `Smoothing` class works.
3. (optional) Now that we have a modular CNN, we can add more layers to it. Try the network with up to 4 layers. You will need to make the filters shorter, as the number of weights are starting to add up. Does it help?
4. (optional) Now let's swap out the CNN for a UNet. The UNet is conceptually similar to the CNN, but it uses a series of downsampling layers and upsampling layers, allowing the filters to response to more of the signal despite small kernels. You can find the UNet in `unet.py`. Swap out the CNN for the UNet in `train.py`. How does this change the training? How does this change the predictions? How does this change the validation R2?

### New features

Consider different kinds of modifications of the network that might improve it. You don't have to do these in order: they're just to get you thinking about how to improve the model and exercise your PyTorch hacking skills.

1. Replace the masked tokens, which are currently assigned zeros, with something else. You could use, e.g.:
    * the average spike rate outside of the mask for each neuron
    * replace the spike rates with the nearest neighbor on either side of the mask
    * a linear interpolation of the spike trains on either side of the mask
    * give the mask to the input and output layers and allow it to be mixed in
2. Try to increase the masking ratio up to 0.6 and see if the model can still learn. How about using masking over *neurons* rather than masking over timesteps? Does that help?
3. Add dropout layers. Look up the function `torch.nn.Dropout1d`. Does dropout do anything here?
4. Add in [stochastic weight averaging](https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html) (SWA).
5. Try the same model on the chaotic rnn dataset. Does it perform better or worse? Why?
6. (advanced) Using the mean squared error is generally appropriate for continuous data. However, we know that for spikes, the Poisson distribution better captures the statistics of the data. Try to use the Poisson loss function instead of the MSE loss function. Look up the `torch.nn.PoissonNLLLoss` function. [You can use this paper as a reference](https://www.biorxiv.org/content/10.1101/463422v2).
