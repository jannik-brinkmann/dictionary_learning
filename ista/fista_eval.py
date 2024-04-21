
# This is an adaptation of the training script of https://github.com/saprmarks/dictionary_learning

import torch
from datasets import load_dataset
from nnsight import LanguageModel
from tqdm import tqdm

from dictionary_learning import ActivationBuffer
from dictionary_learning.training import trainSAE
from dictionary_learning.evaluation import evaluate

from fista_utils import FISTADict


device = 'cuda:0'
model = LanguageModel(
    'gpt2',
    device_map = device
)
submodule = model.transformer.h[1].mlp
activation_dim = 768 # output dimension of the MLP

# data much be an iterator that outputs strings
data = iter(load_dataset("jbrinkma/pile-300k")["train"]["text"])
buffer = ActivationBuffer(
    data,
    model,
    submodule,
    submodule_output_dim=activation_dim, # output dimension of the model component
    n_ctxs=1e4, # you can set this higher or lower depending on your available memory
    return_act_batch_size=256,
    device='cpu' # doesn't have to be the same device that you train your autoencoder on
) # buffer will return batches of tensor

# initialise dictionary
sparsity_coefficient = 0.8
dictionary = FISTADict(
    dict_path = f"FISTA_dict_{sparsity_coefficient}.pt",
    sparsity_coefficient=sparsity_coefficient
)

out = evaluate(
    model,
    submodule,  # a submodule of model
    dictionary,  # a dictionary
    buffer,  # an ActivationBuffer
    max_len=128,  # max context length for loss recovered
    batch_size=6,  # batch size for loss recovered
    entropy=False,  # whether to use entropy regularization
    hist_save_path="hist.png",  # path for saving histograms
    hist_title="FISTA",  # title for histograms
    io="out",  # can be 'in', 'out', or 'in_to_out'
    device="cuda",
)