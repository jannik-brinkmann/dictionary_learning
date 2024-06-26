This is a repository for doing dictionary learning via sparse autoencoders on neural network activations. It was developed by Samuel Marks and Aaron Mueller. 

For accessing, saving, and intervening on NN activations, we use the [`nnsight`](http://nnsight.net/) package; as of March 2024, `nnsight` is under active development and may undergo breaking changes. That said, `nnsight` is easy to use and quick to learn; if you plan to modify this repo, then we recommend going through the main `nnsight` demo [here](http://nnsight.net/notebooks/walkthrough/).

Some dictionaries trained using this repository (and asociated training checkpoints) can be accessed at [https://baulab.us/u/smarks/autoencoders/](https://baulab.us/u/smarks/autoencoders/). See below for more information about these dictionaries.

# Set-up

Navigate to the to the location where you would like to clone this repo, clone and enter the repo, and install the requirements.
```bash
git clone https://github.com/saprmarks/dictionary_learning
cd dictionary_learning
pip install -r requirements.txt
```

To use `dictionary_learning`, include it as a subdirectory in some project's directory and import it; see the examples below.

# Using trained dictionaries

To use a dictionary, just import the dictionary class (currently only autoencoders are supported), initialize an `AutoEncoder`, and load a saved state_dict (see a description for downloading our pretrained dictionaries below).
```python
from dictionary_learning import AutoEncoder
import torch

activation_dim = 512 # dimension of the NN's activations to be autoencoded
dictionary_size = 16 * activation_dim # number of features in the dictionary
ae = AutoEncoder(activation_dim, dictionary_size)
ae.load_state_dict(torch.load("path/to/dictionary/weights"))

# get NN activations using your preferred method: hooks, transformer_lens, nnsight, etc. ...
# for now we'll just use random activations
activations = torch.randn(64, activation_dim)
features = ae.encode(activations) # get features from activations
reconstructed_activations = ae.decode(features)

# if you want to use both the features and the reconstruction, you can get both at once
reconstructed_activations, features = ae(activations, output_features=True)
```

Dictionaries have `encode`, `decode`, and `forward` methods -- see `dictionary.py`.

# Training your own dictionaries

To train your own dictionaries, you'll need to understand a bit about our infrastructure. (See below for downloading our dictionaries.)

One key object is the `ActivationBuffer`, defined in `buffer.py`. Following [Neel Nanda's appraoch](https://www.lesswrong.com/posts/fKuugaxt2XLTkASkk/open-source-replication-and-commentary-on-anthropic-s), `ActivationBuffer`s maintain a buffer of NN activations, which it outputs in batches.

An `ActivationBuffer` is initialized from an `nnsight` `LanguageModel` object, a submodule (e.g. an MLP), and a generator which yields strings (the text data). It processes a large number of strings, up to some capacity, and saves the submodule's activations. You sample batches from it, and when it is half-depleted, it refreshes itself with new text data.

Here's an example for training a dictionary; in it we load a language model as an `nnsight` `LanguageModel` (this will work for any Huggingface model), specify a submodule, create an `ActivationBuffer`, and then train an autoencoder with `trainSAE`.
```python
from nnsight import LanguageModel
from dictionary_learning import ActivationBuffer
from dictionary_learning.training import trainSAE

model = LanguageModel(
    'EleutherAI/pythia-70m-deduped', # this can be any Huggingface model
    device_map = 'cuda:0'
)
submodule = model.gpt_neox.layers[1].mlp # layer 1 MLP
activation_dim = 512 # output dimension of the MLP
dictionary_size = 16 * activation_dim

# data much be an iterator that outputs strings
data = iter([
    'This is some example data',
    'In real life, for training a dictionary',
    'you would need much more data than this'
])
buffer = ActivationBuffer(
    data,
    model,
    submodule,
    submodule_output_dim=activation_dim, # output dimension of the model component
    n_ctxs=3e4, # you can set this higher or lower dependong on your available memory
    device='cuda:0' # doesn't have to be the same device that you train your autoencoder on
) # buffer will return batches of tensors of dimension = submodule's output dimension

# train the sparse autoencoder (SAE)
ae = trainSAE(
    buffer,
    activation_dim,
    dictionary_size,
    lr=3e-4,
    sparsity_penalty=1e-3,
    device='cuda:0'
)
```
Some technical notes our training infrastructure and supported features:
* Training uses the `ConstrainedAdam` optimizer defined in `training.py`. This is a variant of Adam which supports constraining the `AutoEncoder`'s decoder weights to be norm 1.
* Neuron resampling: if a `resample_steps` argument is passed to `trainSAE`, then dead neurons will periodically be resampled according to the procedure specified [here](https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-resampling).
* Learning rate warmup: if a `warmup_steps` argument is passed to `trainSAE`, then a linear LR warmup is used at the start of training and, if doing neuron resampling, also after every time neurons are resampled.

If `submodule` is a model component where the activations are tuples (e.g. this is common when working with residual stream activations), then the buffer yields the first coordinate of the tuple.

# Downloading our open-source dictionaries

To download our pretrained dictionaries automatically, run:

```bash
./pretrained_dictionary_downloader.sh
```
Per default, this will download dictionaries of all submodules (~2.5 GB). The `--layers "embed,0,1,2,3,4,5"` flag allows you to select specific layers, e.g append `--layers "embed,2"` to download dictionaries for the outputs of embedding, as well as attention, MLP, and residual stream submodules in layer 2. Optionally, you can download checkpoints for the selected layers with `--checkpoints` (note that checkpoints take up ~10 GB per layer).

Currently, the main thing to look for is the dictionaries in our `10_32768` set; this set has dictionaries for MLP outputs, attention outputs, and residual streams (including embeddings) in all layers of EleutherAI's Pythia-70m-deduped model. These dictionaries were trained on 2B tokens from The Pile.

Let's explain the directory structure by example. After using the script above, you'll have a `dictionaries/pythia-70m-deduped/mlp_out_layer1/10_32768` directory corresponding to the layer 1 MLP dictionary from the `10_32768` set. This directory contains:
* `ae.pt`: the `state_dict` of the fully trained dictionary
* `config.json`: a json file which specifies the hyperparameters used to train the dictionary
* `checkpoints/`: a directory containing training checkpoints of the form `ae_step.pt` (only if you used the `--checkpoints` flag)

We've also previously released other dictionaries which can be found and downloaded [here](https://baulab.us/u/smarks/autoencoders/). 

## Statistics for our dictionaries

We'll report the following statistics for our `10_32768` dictionaries. These were measured using the code in `evaluation.py`.
* **MSE loss**: average squared L2 distance between an activation and the autoencoder's reconstruction of it
* **L1 loss**: a measure of the autoencoder's sparsity
* **L0**: average number of features active above a random token
* **Percentage of neurons alive**: fraction of the dictionary features which are active on at least one token out of 8192 random tokens
* **CE diff**: difference between the usual cross-entropy loss of the model for next token prediction and the cross entropy when replacing activations with our dictionary's reconstruction
* **Percentage of CE loss recovered**: when replacing the activation with the dictionary's reconstruction, the percentage of the model's cross-entropy loss on next token prediction that is recovered (relative to the baseline of zero ablating the activation)

### Attention output dictionaries

| Layer | Variance Explained (%) | L1 | L0  | % Alive | CE Diff | % CE Recovered |
|-------|------------------------|----|-----|---------|---------|----------------|
| 0     | 92                     | 8  | 128 | 17      | 0.02    | 99             |
| 1     | 87                     | 9  | 127 | 17      | 0.03    | 94             |
| 2     | 90                     | 19 | 215 | 12      | 0.05    | 93             |
| 3     | 89                     | 12 | 169 | 13      | 0.03    | 93             |
| 4     | 83                     | 8  | 132 | 14      | 0.01    | 95             |
| 5     | 89                     | 11 | 144 | 20      | 0.02    | 93             |


### MLP output dictionaries

| Layer  | Variance Explained (%) | L1 | L0  | % Alive | CE Diff | % CE Recovered |
|--------|------------------------|----|-----|---------|---------|----------------|
|     0  | 97                     | 5  | 5   | 40      | 0.10    | 99             |
|     1  | 85                     | 8  | 69  | 44      | 0.06    | 95             |
|     2  | 99                     | 12 | 88  | 31      | 0.11    | 88             |
|     3  | 88                     | 20 | 160 | 25      | 0.12    | 94             |
|     4  | 92                     | 20 | 100 | 29      | 0.14    | 90             |
|     5  | 96                     | 31 | 102 | 35      | 0.15    | 97             |


### Residual stream dictionaries
NOTE: these are indexed so that the resid_i dictionary is the *output* of the ith layer. Thus embeddings go first, then layer 0, etc.

| Layer   | Variance Explained (%) | L1 | L0  | % Alive | CE Diff | % CE Recovered |
|---------|------------------------|----|-----|---------|---------|----------------|
|    embed| 96                     |  1 |  3  | 36      | 0.17    | 98             |
|       0 | 92                     | 11 | 59  | 41      | 0.24    | 97             |
|       1 | 85                     | 13 | 54  | 38      | 0.45    | 95             |
|       2 | 96                     | 24 | 108 | 27      | 0.55    | 94             |
|       3 | 96                     | 23 | 68  | 22      | 0.58    | 95             |
|       4 | 88                     | 23 | 61  | 27      | 0.48    | 95             |
|       5 | 90                     | 35 | 72  | 45      | 0.55    | 92             |




# Extra functionality supported by this repo

**Note:** these features are likely to be depricated in future releases.

We've included support for some experimental features. We briefly investigated them as an alternative approaches to training dictionaries.

* **MLP stretchers.** Based on the perspective that one may be able to identify features with "[neurons in a sufficiently large model](https://transformer-circuits.pub/2022/toy_model/index.html)," we experimented with training "autoencoders" to, given as input an MLP *input* activation $x$, output not $x$ but $MLP(x)$ (the same output as the MLP). For instance, given an MLP which maps a 512-dimensional input $x$ to a 1024-dimensional hidden state $h$ and then a 512-dimensional output $y$, we train a dictionary $A$ with hidden dimension 16384 = 16 x 1024 so that $A(x)$ is close to $y$ (and, as usual, so that the hidden state of the dictionary is sparse).
    * The resulting dictionaries seemed decent, but we decided not to pursue the idea further.
    * To use this functionality, set the `io` parameter of an activaiton buffer to `'in_to_out'` (default is `'out'`).
    * h/t to Max Li for this suggestion.
* **Replacing L1 loss with entropy**. Based on the ideas in this [post](https://transformer-circuits.pub/2023/may-update/index.html#simple-factorization), we experimented with using entropy to regularize a dictionary's hidden state instead of L1 loss. This seemed to cause the features to split into dead features (which never fired) and very high-frequency features which fired on nearly every input, which was not the desired behavior. But plausibly there is a way to make this work better.
* **Ghost grads**, as described [here](https://transformer-circuits.pub/2024/jan-update/index.html). 


