# Unet-spikes

Tutorial repository for the Cajal school in machine learning for neuroscience. 

The core of the repo showcases a UNet that infers a latent space on spikes. The idea is to train a UNet to perform masked language modeling (MLM) on binned spike trains. This is conceptually similar to the Neural Data Transformer (NDT) from Ye and Pandarinath (2021), itself taking its cues from BERT. This replaces the transformer with the more stable training inherent in a UNet. 

The UNet performs a combination of local smoothing and mixing data from different channels (i.e. nonlinear PCA) to smooth spike trains. It learns the statistics of ensembles of spike trains towards this end.

## Getting started

* Clone this repository (or a fork of it if you want to be able to modify it)
* Create a fresh conda environment with e.g. Python 3.9, `conda create --name unet-spikes python=3.9`
* `cd` into the directory and `pip install -e .`
* `pip install -r requirements.txt`

Then, clone the data necessary to continue. Use an automated tool like `wget`:

```
wget -P data https://cajal-data-740441.s3.eu-west-3.amazonaws.com/lfads_lorenz.h5
wget -P data https://cajal-data-740441.s3.eu-west-3.amazonaws.com/chaotic_rnn_no_inputs_dataset_N50_S50.h5
```

Or download manually [1](https://cajal-data-740441.s3.eu-west-3.amazonaws.com/chaotic_rnn_no_inputs_dataset_N50_S50.h5
) and [2](https://cajal-data-740441.s3.eu-west-3.amazonaws.com/lfads_lorenz.h5) and put it into the data folder.

## Training a model

`cd` into scripts and use `python train.py` to train the model.

## Exercises

### Debugging the model

0. Train the model using `python train.py`.
1. Load up tensorboard and visualize a training run. How does it look? Based on the graphs, do you think that the network is learning something meaningul?

*Hint*: in `train.py`, use `logger.add_image` to

2. Add visualization for the model outputs and the model targets. The model is trained on spike data, but because this is simulation data, we have access to the underlying rates. Log the following information to Tensorboard: `preds`, `target`, `the_mask` and `rates`. It's sufficient to log the last training batch. Use functions such `logger.add_image` or `logger.add_figure` to log images and figures to Tensorboard. How do they look?

1. One big issue with the model is that it can give negative rates. This is a problem because rates are inherently positive. Add a nonlinearity to the model to ensure that the rates are positive. Look inside the CNN function and add a nonlinearity to the output of the last layer. You can use `torch.nn.ReLU` or `torch.nn.Softplus` for this. How does this change the training? How does this change the predictions? How does this change the validation R2?


### Parameter sweep

1. Let's tweak the learning rate. Try a grid of different parameters. Look at the loss function and the validation R2. What is a good learning rate? Is the model training very sensitive to the learning rate?

### Improving the model

1. Perhaps the model is too low capacity and it needs more layers to learn some interesting features. Add one more set of layers to the model. You can do this by adding `smoothing1`, `bn1` and `relu1` layers in `cnn.py`, and also adding them in the `forward` function of `UNet`. How does this change the training? How does this change the predictions? How does this change the validation R2? Note: you need to create separate layers and not reuse the old ones, because otherwise the weights will be tied in the first convolutional layer and the second one.
2. Now we have a bit of a mess on our hands! It would be better if the model was more modular. Time to refactor! Add a `Smoothing` layer class inside of `cnn.py` that wraps `smoothing`, `bn` and `relu` operations. It should be a subclass of `nn.Module`. Run the training again to make sure you didn't break anything! Stretch goal: test the CNN from the test module in tests/test_cnn.py. You can run this from the command line via `pytest test_cnn.py`. Write another test that checks that the Smoothing class works.
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
