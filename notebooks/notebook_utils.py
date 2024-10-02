from tqdm import tqdm
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
# tinymodel
from tinymodel import TinyModel, tokenizer
import nnsight
import einops
from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
import torch.nn.functional as F

def load_tinymodel(convert_to_nnsight=True):
    lm = TinyModel(n_layers=2, from_pretrained='tiny_model_2L_3E')
    if convert_to_nnsight:
        nnsight_lm = nnsight.NNsight(lm)
        return nnsight_lm
    return lm


class AutoEncoderTopK(nn.Module):
    """
    The top-k autoencoder architecture and initialization used in https://arxiv.org/abs/2406.04093
    NOTE: (From Adam Karvonen) There is an unmaintained implementation using Triton kernels in the topk-triton-implementation branch.
    We abandoned it as we didn't notice a significant speedup and it added complications, which are noted
    in the AutoEncoderTopK class docstring in that branch.

    With some additional effort, you can traisn a Top-K SAE with the Triton kernels and modify the state dict for compatibility with this class.
    Notably, the Triton kernels currently have the decoder to be stored in nn.Parameter, not nn.Linear, and the decoder weights must also
    be stored in the same shape as the encoder.
    """

    def __init__(self, activation_dim: int, dict_size: int, k: int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.k = k

        self.encoder = nn.Linear(activation_dim, dict_size)
        self.encoder.bias.data.zero_()

        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.decoder.weight.data = self.encoder.weight.data.clone().T
        self.set_decoder_norm_to_unit_norm()

        self.b_dec = nn.Parameter(torch.zeros(activation_dim))

    def encode(self, x: torch.Tensor, return_topk: bool = False):
        post_relu_feat_acts_BF = nn.functional.relu(self.encoder(x - self.b_dec))
        post_topk = post_relu_feat_acts_BF.topk(self.k, sorted=False, dim=-1)

        # We can't split immediately due to nnsight
        tops_acts_BK = post_topk.values
        top_indices_BK = post_topk.indices

        buffer_BF = torch.zeros_like(post_relu_feat_acts_BF)
        encoded_acts_BF = buffer_BF.scatter_(dim=-1, index=top_indices_BK, src=tops_acts_BK)

        if return_topk:
            return encoded_acts_BF, tops_acts_BK, top_indices_BK
        else:
            return encoded_acts_BF

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x) + self.b_dec

    def forward(self, x: torch.Tensor, output_features: bool = False):
        encoded_acts_BF = self.encode(x)
        x_hat_BD = self.decode(encoded_acts_BF)
        if not output_features:
            return x_hat_BD
        else:
            return x_hat_BD, encoded_acts_BF

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        eps = torch.finfo(self.decoder.weight.dtype).eps
        norm = torch.norm(self.decoder.weight.data, dim=0, keepdim=True)
        self.decoder.weight.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.decoder.weight.grad is not None  # keep pyright happy

        parallel_component = einops.einsum(
            self.decoder.weight.grad,
            self.decoder.weight.data,
            "d_in d_sae, d_in d_sae -> d_sae",
        )
        self.decoder.weight.grad -= einops.einsum(
            parallel_component,
            self.decoder.weight.data,
            "d_sae, d_in d_sae -> d_in d_sae",
        )

    def from_pretrained(path, k: int, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = torch.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        autoencoder = AutoEncoderTopK(activation_dim, dict_size, k)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder
    
class FeatureAutoEncoderTopK(AutoEncoderTopK):
    def __init__(self, activation_dim: int, dict_size: int, k: int, setting="normal"):
        super().__init__(activation_dim, dict_size, k)
        self.setting = setting

    def define_feature_by_feature_matrix(self):
        # turn off gradients to the decoder
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.feature_by_feature = nn.Linear(self.dict_size, self.dict_size, bias=False)

    def initialize_feature_by_feature_matrix(self, decoder_weight):
        self.feature_by_feature.weight.data = self.encoder.weight.clone() @ decoder_weight.weight
    
    def feature_by_feature_forward(self, x: torch.Tensor):
        x = self.encode(x)
        return self.feature_by_feature(x)
        
    def from_pretrained(path, k: int, device=None, setting="normal"):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = torch.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        autoencoder = FeatureAutoEncoderTopK(activation_dim, dict_size, k, setting=setting)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
            
        if(setting == "feature_by_feature"):
            autoencoder.define_feature_by_feature_matrix()
            autoencoder.feature_by_feature.to(device)
            autoencoder.decoder.to("cpu")
                
        return autoencoder
    

def download_dataset(dataset_name, tokenizer, max_length=256, num_datapoints=None):
    if(num_datapoints):
        split_text = f"train[:{num_datapoints}]"
    else:
        split_text = "train"
    dataset = load_dataset(dataset_name, split=split_text)
    total_failed_tokens = 0
    all_tokens = []
    for text in tqdm(dataset["text"]):
        try:
            tokens = [9996] + tokenizer.encode(text)[:max_length]
        except:
            total_failed_tokens += 1
            continue
        # only include if it's at least max_length
        if len(tokens) == max_length+1:
            all_tokens.append(tokens)
    print(f"Failed to tokenize {total_failed_tokens} tokens")
    # convert into a dataset class
    return  Dataset.from_dict({"input_ids": all_tokens})

def load_tinydataset(batch_size, max_seq_length, num_datapoints, verbose=True):
    dataset_name = "noanabeshima/TinyStoriesV2"
    dataset = download_dataset(dataset_name, tokenizer=tokenizer, max_length=max_seq_length, num_datapoints=num_datapoints)
    true_num_datapoints = len(dataset)
    # added BOS
    max_seq_length +=1
    total_tokens = max_seq_length * true_num_datapoints
    if verbose:
        print(f"Number of datapoints w/ {max_seq_length} tokens: {true_num_datapoints}")
        print(f"Total Tokens: {total_tokens / 1e6}M")
    with dataset.formatted_as("pt"):
        dl = DataLoader(dataset["input_ids"], batch_size=batch_size)
    return dl


def load_saes(sae_class=AutoEncoderTopK, k = 30, setting="normal"):
    model_id = "jbrinkma/tinystories_saes"
    sae_filenames = ["embed.pt"] + [f"torso_{layer}_{filename}" for layer in range(2) for filename in ["attn.pt","mlp_out_transcoder.pt",  "res_final.pt"]]
    saes = {}
    for filename in sae_filenames:
        download_loc = hf_hub_download(repo_id=model_id, filename=filename)
        if(sae_class==AutoEncoderTopK):
            sae = sae_class.from_pretrained(download_loc, k=k)
        else:
            sae = sae_class.from_pretrained(download_loc, k=k, setting=setting)
        name = filename.split(".")[0]
        saes[name] = sae
    return saes