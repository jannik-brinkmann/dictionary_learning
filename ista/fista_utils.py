import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from nnsight.models.LanguageModel import LanguageModelProxy

from dictionary_learning.dictionary import Dictionary


from nnsight import DEFAULT_PATCHER
from nnsight.tracing.Proxy import proxy_wrapper
from nnsight.patching import Patch


DEFAULT_PATCHER.add(Patch(torch, proxy_wrapper(torch.zeros), "zeros"))


def FISTA(I,basis,lambd,num_iter,eta=None, useMAGMA=True):
    # Copied and modified from Yun et al. 
    
    # This is a positive-only PyTorch-Ver FISTA solver
    batch_size=I.size(1)
    M = basis.size(1)
    if eta is None:
        L = torch.max(torch.linalg.eigh(torch.mm(basis, basis.t()))[0])
        eta = 1./L

    tk_n = 1.
    tk = 1.
    Res = torch.zeros_like(I).cuda()
    # Res = torch.cuda.FloatTensor(.size()).fill_(0)
    ahat = torch.zeros((M,batch_size)).cuda()
    ahat_y = torch.zeros((M,batch_size)).cuda()

    for t in range(num_iter):
        tk = tk_n
        tk_n = (1+np.sqrt(1+4*tk**2))/2
        ahat_pre = ahat
        Res = I - torch.mm(basis,ahat_y)
        ahat_y = ahat_y.add(eta * basis.t().mm(Res))
        ahat = ahat_y.sub(eta * lambd).clamp(min = 0.)
        ahat_y = ahat.add(ahat.sub(ahat_pre).mul((tk-1)/(tk_n)))
    Res = I - torch.mm(basis,ahat)
    return ahat, Res

def quadraticBasisUpdate(basis, Res, ahat, lowestActivation, HessianDiag, stepSize = 0.001,constraint = 'L2', Noneg = False):
    # Copied and modified from Yun et al.
    
    """
    This matrix update the basis function based on the Hessian matrix of the activation.
    It's very similar to Newton method. But since the Hessian matrix of the activation function is often ill-conditioned, we takes the pseudo inverse.

    Note: currently, we can just use the inverse of the activation energy.
    A better idea for this method should be caculating the local Lipschitz constant for each of the basis.
    The stepSize should be smaller than 1.0 * min(activation) to be stable.
    """
    dBasis = stepSize*torch.mm(Res, ahat.t())/ahat.size(1)
    dBasis = dBasis.div_(HessianDiag+lowestActivation)
    basis = basis.add_(dBasis)
    if Noneg:
        basis = basis.clamp(min = 0.)
    if constraint == 'L2':
        basis = basis.div_(basis.norm(2,0))
    return basis



def train_fista(
    activation_buffer, 
    activation_dim: int,
    expansion_factor: int, 
    steps: int,
    sparsity_coefficient: float,
    device: str = "cuda"
):
    
    n_features = expansion_factor * activation_dim
    
    # initialise the dict
    dict_size = [activation_dim, n_features]
    dictionary = torch.randn(dict_size).to(device)
    dictionary = dictionary.div_(dictionary.norm(2, 0))
    
    # initialise training parameters
    hessian_diag = torch.zeros(n_features).to(device)
    act_hist = 300
    
    # dict training loop
    for step, acts in enumerate(tqdm(activation_buffer, total=steps)):
        
        if steps is not None and step >= steps:
            break

        if isinstance(acts, torch.Tensor): # typical case
            acts = acts.to(device)
        elif isinstance(acts, tuple): # if autoencoder input and output are different
            acts = tuple(a.to(device) for a in acts)

        # compute sparse codes
        sparse_codes, residuals = FISTA(
            acts.T, 
            dictionary, 
            sparsity_coefficient, 
            500  # iterations
        )

        # update dict
        hessian_diag = hessian_diag.mul((act_hist - 1.0) / act_hist) 
        hessian_diag += torch.pow(sparse_codes, 2).mean(1) / act_hist
        dictionary = quadraticBasisUpdate(
            dictionary, 
            residuals, 
            sparse_codes, 
            0.001,  # small value to avoid dividing by zero
            hessian_diag, 
            0.005  # step size
        )

        if step % 2 == 0:
            MSE = ((acts.T - torch.mm(dictionary, sparse_codes)) ** 2).mean()
            L0 = torch.norm(sparse_codes, 0, dim=0).mean()
            print(f"MSE: {MSE}. L0: {L0}.")
            
    return dictionary


class FISTADict(Dictionary, nn.Module):
    
    def __init__(self, dict_path, sparsity_coefficient) -> None:
        super().__init__()
        self.dict = torch.load(dict_path).cuda()
        self.sparsity_coefficient = sparsity_coefficient
        
    def encode(self, x):
        features, _ = FISTA(
            x.T, 
            self.dict, 
            self.sparsity_coefficient, 
            500  # iterations
        )
        return features
    
    def decode(self, f):
        return torch.mm(self.dict, f).T
    
    def forward(self, x, output_features=False, ghost_mask=None):
        f = self.encode(x)
        x_hat = self.decode(f)
        if output_features:
            return x_hat, f
        else:
            return x_hat
