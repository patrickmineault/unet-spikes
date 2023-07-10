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
wget -P data https://cajal-data-740441.s3.eu-west-3.amazonaws.com/chaotic_rnn_no_inputs_dataset_N50_S50.h5
wget -P data https://cajal-data-740441.s3.eu-west-3.amazonaws.com/lfads_lorenz.h5
```

Or download manually [1](https://cajal-data-740441.s3.eu-west-3.amazonaws.com/chaotic_rnn_no_inputs_dataset_N50_S50.h5
) and [2](https://cajal-data-740441.s3.eu-west-3.amazonaws.com/lfads_lorenz.h5) and put it into the data folder.

## Training a model

`cd` into scripts and use `python train.py` to train the model.

## Exercises

A core learning objective of this repository is to help you become productive in deep learning with larger codebases, and so the exercises are less structured than conventional lab-in-a-notebook setups. Here are some exercises that you can focus on:

### Testing

1. Write some tests for the UNet. You can put them in `tests/test_unet.py`. Start by writing a test `test_shape` that verifies that the shape of inputs is the same as the shape of outputs when you run them through the `UNet.forward` function. Make sure that this works regardless of the size of the input. Verify your tests run via `pytest`. To challenge yourself, use the `broken-unet` branch, which contains a broken version of the UNet. As you uncover more issues with the UNet, add more tests.
2. Write some tests for the data loader. Make sure that the code works equally when you use "../data/configs/lorenz.yaml" or "../data/configs/chaotic.yaml".  

Potential solutions are in the `sample-tests` branch.

### Visualization

1. Load up tensorboard and visualize a training run. How does it look? Do you have a clear view of what the network is learning? Add more visualization to verify that the middle layers are learning something helpful.
2. Add support for weights and biases (wandb). Sign up for a [wandb](https://wandb.com/) account, `pip install wandb`, go through `wandb init`, and add a couple of lines to the training loop (`train.py`) so that it supports both local and remote logging. Do a training run and visualize it online. 

### Parameter sweeps

1. Convince yourself that the inner layers in the UNet are doing something. Do a parameter sweep from 0 to 4 inner layers (first parameter of `UNet`). Visualize the predictions and accuracy in Tensorboard. What does the validation R2 look like? What do predictions look like? Can you summarize in a few sentences what differs when you add in extra layers? Hint: look at the spatial smoothness of the predictiohns.
2. Look at what happens as you sweep across latent dimensionality from `4` to `64` in steps of powers of 2. Does the network keep improving with higher latent dimensionality? How does the dimensionality of the network compare to the size of the data? Bonus: look back to the auto-encoder exercise you did yesterday. Does it react the same with latent dimensionality? Speculate as to why this may or may not be the case.

### New features

Currently, the network performs about as well as the Neural Data Transformer on the Lorenz benchmark but significantly underperforms on the chaotic RNN (R2 of 0.5 vs. 85 reported in the paper). Why is that? Consider different kinds of modifications of the network that might "fix it".

1. Replace the masked tokens, which are currently assigned zeros, with something else. You could use, e.g.:
    * the average spike rate outside of the mask for each neuron
    * replace the spike rates with the nearest neighbor on either side of the mask
    * a linear interpolation of the spike trains on either side of the mask
    * give the mask to the input and output layers and allow it to be mixed in
2. Add dropout layers. NDT reported that dropout needed to be > .2 for their network to work well. Does dropout do anything here?
3. Replace the output nonlinearity of the network from an exponential to something less extreme. The output nonlinearity is implicit, we use a Poisson loss and give it the log rates, and internally it exponentiates them. This is consistent with the exponential being the canonical nonlinearity for the Poisson GLM. Replace it with something less extreme, like log(1+a*exp(x)). Be careful to implement this in a [numerically stable way](https://github.com/pytorch/pytorch/issues/39242). Write tests to verify your nonlinearity works well and doesn't result in NaNs with large negative and large positive inputs.
4. Add back in a transformer-like mechanism to allow long-range interactions. On the highest level layer, add in two transformers, one that works across space, another that works across time.
5. Do an automated sweep over the learning rate using [PyTorch Lightning's learning rate finder](https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html).
6. Add in [stochastic weight averaging](https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html) (SWA).
7. Try to implement some of the proposed changes in the [ConvNextV2 paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.pdf).

## New datasets

Though the network is trained on synthetic datasets, it can be modified to work with arbitrary datasets. To apply to some of the datasets used in the neural latents benchmark, install the dandi tool (`pip install dandi`) and download these datasets to a local folder:

```
dandi download DANDI:000127
dandi download DANDI:000128
dandi download DANDI:000129
dandi download DANDI:000130
dandi download DANDI:000138
dandi download DANDI:000139
dandi download DANDI:000140
```

You can then preprocess the data using `scripts/prep_nlb.py`. Specify the input directory using the `--data-root` argument. Once the data is preprocessed, it will be put in data/h5. You may then use it to train the network. Read up more on the datasets in the [Neural Latents Benchmark paper](https://arxiv.org/abs/2109.04463). 