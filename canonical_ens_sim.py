import numpy as np
import torch

cpu = torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else cpu

class Canonical_Ensemble_Simulator():
    def __init__(self, model, beta: float, device=device):
        """
        Perform canonical ensemble simulation using the Metropolis algorithm
        @params beta (float): The thermodynamic beta. 
            CONSTRAINT: beta > 0
        @params steps (int): The number of steps of the simulation.
            CONSTRAINT: steps > 0
        @params showbar (bool, DEFAULT: False): Iff true, a progress bar 
            showing the current iteration in the simulation is shown. 
        @return (torch.Tensor): The hamiltonian of the system throughout the 
            simulation. 
            CONSTRAINT: RETURN.ndim == 2 and RETURN.shape[1] == steps + 1
            RETURN[i, j]: Hamiltonian of state I at step j. 
        @error ValueError: Error if the neighbor generation is not actiavted. 
        """
        self.model = model
        if not self.model.neighbor_gen_activated():
            raise ValueError(
                "Neighbor generation is not activated in this model"
            )
        self.beta = beta
        self.device = device
    
    def initialize(self):
        self.hamiltonian = self.model.hamiltonian()
        self.it = 0        
    
    def step(self):
        next_n = self.model.neighbor_generation()
        next_hamiltonian = self.model.hamiltonian(n=next_n)
        alpha = torch.rand(self.model.batch_size, device=self.device,
            dtype=torch.double)
        delta_hamiltonian = self.hamiltonian - next_hamiltonian
        acceptance_rate = torch.exp(self.beta * delta_hamiltonian)
        accepted = alpha < acceptance_rate
        self.model.n[accepted] = next_n[accepted]
        # print(accepted, self.hamiltonian, next_hamiltonian)
        # print(torch.where(accepted,
        #     next_hamiltonian, self.hamiltonian))
        # print(self.model.hamiltonian())
        self.hamiltonian = torch.where(accepted,
            next_hamiltonian, self.hamiltonian)
        self.it += 1