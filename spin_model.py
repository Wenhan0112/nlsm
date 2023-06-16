import torch
import numpy as np
from typing import Optional
import tqdm
import physics_model

cpu = torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else cpu

def uniform_sphere(d: int, size=1, device=device) -> torch.Tensor:
    """
    Uniformly generate points on a unit sphere.
    @params d (int): The dimension of the sphere.
        CONSTRAINT: d >= 1
    @params size (int or sequence of ints): The shape of the samples.
        The output array shape is np.append(size, d). 
    @return (torch.Tensor): The samples from the uniform sphere.
        CONSTRAINT: RETURN.shape = size + (3,)
    """
    if size == 1:
        size = (d,)
    else:
        size = np.append(size, d)
    n = torch.randn(*size, device=device, dtype=torch.double)
    return torch.nn.functional.normalize(n, dim=-1)

###### Old neighbor generation algorithm
###### Every lattice site is updated, not only a specific one.
# def neighbor_generation(n: torch.Tensor, sigma: float, device=device) \
#         -> torch.Tensor:
#     n = n + torch.randn(n.shape, device=device, dtype=torch.double) * sigma
#     return torch.nn.functional.normalize(n, dim=-1)

class Spin_Model(physics_model.Physics_Model):
    """
    A physics model of a spin field, i.e. a collections of vectors on 
    3-dimensional unit sphere. 

    The spin model provides a neighbor generation model, used for 
    Metropolis algorthm.
    
    """
    def __init__(self,
            batch_size: int = 1,
            gen_prob: Optional[float] = None,
            gen_sigma: Optional[float] = None,
            device: torch.device = device):
        """
        Constructor.
        @params batch_size (int): Number of states created.
            CONSTRAINT: batch_size > 0
        @params gen_prob (Optional[float], DEFAULT None): The new spin 
            generation probability in the neighbor generation model. If it is 
            greater than 1, then the spin generation probability is assumed to 
            be 1. 
            CONSTRAINT: gen_prob > 0
        @params gen_sigma (Optional[float], DEFAULT None): The mean offset in 
            the neighbor generation algorithm in the neighbor generation 
            model. 
            CONSTRAINT: gen_sigma > 0
        @params device (torch.device): The device where the states are stored. 
        """
        super().__init__(batch_size=batch_size, device=device)
        self.gen_sigma = gen_sigma
        self.gen_prob = gen_prob

    def neighbor_gen_activated(self):
        """
        OVERRIDE
        """
        return (self.gen_sigma is not None) and (self.gen_prob is not None)

    def neighbor_generation(self) -> torch.Tensor:
        n = self.n
        # num_sites = int(np.prod(n.shape[1:-1]))
        # mask = np.zeros(num_sites, dtype=bool)
        # num = min(num_sites, num)
        # mask_idx = np.random.default_rng().choice(num_sites, num, replace=False)
        # mask[mask_idx] = True
        # mask = np.stack([mask.reshape(n.shape[1:-1])] * n.shape[-1], axis=-1)
        # mask = torch.tensor(mask, device=device, dtype=torch.bool)
        mask = torch.rand(*n.shape[:-1], device=self.device) < self.gen_prob
        mask = mask.unsqueeze(-1)
        # mask = mask.repeat(*np.ones(n.dim()-1, dtype=int), n.shape[-1])
        n = n + torch.randn(n.shape, device=device, dtype=torch.double) \
            * self.gen_sigma * mask
        return torch.nn.functional.normalize(n, dim=-1)


