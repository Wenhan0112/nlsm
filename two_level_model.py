import torch
import numpy
from typing import Optional
import physics_model

cpu = torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else cpu

class Two_Level_Simple_Model(physics_model.Physics_Model):
    def initialize(self, n=None):
        if n is not None:
            self.n = torch.tensor(n, dtype=torch.bool, device=self.device)
        else:
            self.n = self.neighbor_generation()

    def hamiltonian(self):
        return self.n.clone().to(dtype=torch.double)

    def neighbor_generation(self):
        return torch.randint(2, (self.batch_size,),
            dtype=torch.bool, device=self.device)

    def neighbor_gen_activated(self):
        return True
