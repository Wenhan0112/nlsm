import torch
import numpy
from typing import Optional
import spin_model

cpu = torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else cpu

class Spin_Field_Model(spin_model.Spin_Model):
    def __init__(self, magnetic_field,
            batch_size: Optional[int] = None,
            gen_prob: Optional[float] = None,
            gen_sigma: Optional[float] = None,
            device=device):
        super().__init__(batch_size=batch_size,
            gen_prob=gen_prob, gen_sigma=gen_sigma, device=device)
        self.b = torch.tensor(magnetic_field, device=device, dtype=torch.double)
        self.device = device

    def initialize(self, n: Optional[torch.Tensor] = None):
        size = (self.batch_size,)
        if n is None:
            self.n = spin_model.uniform_sphere(3, size=size, device=self.device)
        else:
            self.n = torch.tensor(n, device=self.device, dtype=torch.double)
            self.n = self.n.reshape(size + (3,))

    # def neighbor_gen_activated(self):
    #     return gen_num is not None

    # def neighbor_generation(self) -> torch.Tensor:
    #     return spin_model.uniform_sphere(3,
    #         size=(self.batch_size,), device=self.device)

    def _hamiltonian(self):
        return -(self.n @ self.b)
