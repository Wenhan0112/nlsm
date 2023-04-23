import torch
import numpy as np
import physics_model
from typing import Optional
import matplotlib.pyplot as plt

cpu = torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else cpu

class Ising_Model_1D(physics_model.Physics_Model):
    def __init__(self, 
            num_sites: int, 
            gen_prob: Optional[float] = None,
            batch_size: int = 1, 
            device: torch.device = device,
            boundary: str = "periodic"
        ):
        self.num_sites = num_sites
        self.gen_prob = gen_prob
        if num_sites < 2:
            raise ValueError("At least 2 sites should be present!")
        self.batch_size = batch_size
        self.device = device
        assert boundary in ["open", "periodic"]
        self.boundary = boundary
    
    def initialize(self, n: Optional[torch.Tensor] = None):
        size = (self.batch_size, self.num_sites)
        if n is None:
            self.n = torch.randn(size, device=self.device) > 0
        else:
            self.n = torch.tensor(n, device=self.device, dtype=torch.bool).reshape(size)
        
    def hamiltonian(self, n: Optional[torch.Tensor] = None):
        if n is None:
            n = self.n
        if self.boundary == "open":
            nn = n[..., 1:] ^ n[..., :-1]
        elif self.boundary == "periodic":
            nn = n.roll(1, -1) ^ n
        else:
            raise ValueError()
        return torch.sum(nn.to(dtype=float) * 2 - 1, axis=-1)
    
    def neighbor_gen_activated(self):
        return self.gen_prob is not None
    
    def neighbor_generation(self, n: Optional[torch.Tensor] = None) -> torch.Tensor:
        if n is None:
            n = self.n
        mask = torch.rand(*n.shape, device=self.device) < self.gen_prob
        return n ^ mask
    
    def get_all_states(self):
        return gen_all_binary_vectors(self.num_sites)
    
    def get_spin(self):
        return self.n.float() * 2 - 1


        
def gen_all_binary_vectors(length: int) -> torch.Tensor:
    """
    Generate all bit arrays of specific length.
    @params length (int): The length of the bit array.
        CONSTRAINT: length > 0
    @RETURN (torch.Tensor): All the bit arrays with length LENGTH.
        CONSTRAINT: RETURN.dtype == torch.bool
        CONSTRAINT: tuple(RETURN.shape) == (2**length, length)
        RETURN[i, j]: The J-th element of bit array I. 
    """
    return ((
        torch.arange(2**length).unsqueeze(1) >> 
        torch.arange(length-1, -1, -1)
    ) & 1).bool()