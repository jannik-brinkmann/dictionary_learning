
# This is an adaptation of the training script of https://github.com/saprmarks/dictionary_learning

import torch
from datasets import load_dataset
from nnsight import LanguageModel
from tqdm import tqdm

from dictionary_learning import ActivationBuffer
from dictionary_learning.training import trainSAE

from fista_utils import train_fista


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
    return_act_batch_size=1024,
    device='cpu' # doesn't have to be the same device that you train your autoencoder on
) # buffer will return batches of tensor

# training loop
sparsity_coefficient = 0.8
dictionary = train_fista(
    activation_buffer=buffer,
    activation_dim=activation_dim, 
    expansion_factor=16,
    sparsity_coefficient=sparsity_coefficient,
    steps=1000,
)
torch.save(dictionary, f"FISTA_dict_{sparsity_coefficient}.pt")
