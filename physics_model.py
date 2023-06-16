import torch
import numpy as np
from typing import Optional
import tqdm

cpu = torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else cpu

class Physics_Model():
    """
    A physics model including a state and relavent properties. 
    """
    def __init__(self, batch_size: int = 1, 
            device: torch.device = device):
        """
        Constructor.
        @params batch_size (int): Number of states created.
            CONSTRAINT: batch_size > 0
        @params device (torch.device): The device where the states are stored. 
        """
        self.batch_size = batch_size
        self.device = device

    def hamiltonian(self):
        """
        Hamiltonian of the states.
        @error NotImplementedError: True whenver this function is called but 
            not overriden. 
        """
        raise NotImplementedError("Hamiltonian not implemented!")

    def neighbor_gen_activated(self):
        """
        Get iff the neighbor generation algorithm is implemented.
        @return (bool): Iff the neighbor generation algorithm is implemented.
        """
        return False

    def canonical_ens_sim(self, beta: float, steps: int, 
            show_bar: bool = False, callback: Optional[callable] = None
            ) -> torch.Tensor:
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
        if not self.neighbor_gen_activated():
            raise ValueError(
                "Neighbor generation is not activated in this model"
            )
        if show_bar:
            iterator = tqdm.trange(steps,
                desc="Microcanonical Ensemble Simulation")
        else:
            iterator = range(steps)
        hamiltonian = torch.zeros((self.batch_size, steps + 1),
            device=self.device, dtype=torch.double)
        # dhs = torch.zeros((self.batch_size, steps),
        #     device=self.device, dtype=torch.double) ### Additional return value
        # all_ns = torch.zeros((self.batch_size, steps + 1, *self.n.shape[1:]),
        #     device=self.device, dtype=torch.double)
        # all_ns[:, 0] = self.n
        hamiltonian[:, 0] = self.hamiltonian()
        reject_count = 0
        for i in iterator:
            next_n = self.neighbor_generation()
            self.n, old_n = next_n, self.n
            next_hamiltonian = self.hamiltonian()
            alpha = torch.rand(self.batch_size, device=self.device,
                dtype=torch.double)
            delta_hamiltonian = hamiltonian[:, i] - next_hamiltonian
            acceptance_rate = torch.exp(beta * delta_hamiltonian)
            # dhs[:, i] = delta_hamiltonian ### Additional return value
            not_accepted = alpha > acceptance_rate
            reject_count += torch.mean(not_accepted.to(dtype=torch.double))
            self.n[not_accepted] = old_n[not_accepted]
            # if not_accepted.to(dtype=torch.float64).mean() < 1:
            #     last_accepted = self.n
            # print(not_accepted, hamiltonian[:, i], next_hamiltonian)
            # print(torch.where(not_accepted,
            #     hamiltonian[:, i], next_hamiltonian))
            # print(self.hamiltonian())

            hamiltonian[:, i+1] = torch.where(not_accepted,
                hamiltonian[:, i], next_hamiltonian)
            # hamiltonian[:, i+1] = self.hamiltonian()
            if callback:
                callback(self, i)
            # all_ns[:, i+1] = self.n
        # print("Number of samples rejected", reject_count)
        # print(f"Rejection ratio: {reject_count / steps * 100:.5f}")
        # if reject_count == steps:
        #     print("All generations have been rejected!")
        # return hamiltonian, dhs, all_ns
        return hamiltonian

    def get_state(self):
        """
        Get the state of the system
        """
        return self.n

    def test_generation_h_distribution(self, num_gen: int, show_bar=False):
        """
        Test the hamiltonian distribution of the neighbor generation 
        algorithm. 
        """
        assert self.neighbor_gen_activated()
        if show_bar:
            iterator = tqdm.trange(num_gen,
                desc="Neighbor generation Hamiltonian distribution")
        else:
            iterator = range(num_gen)
        original_h = self.hamiltonian()
        new_h = torch.zeros((self.batch_size, num_gen))
        for i in iterator:
            new_n = self.neighbor_generation()
            self.n, old_n = new_n, self.n
            new_h[:, i] = self.hamiltonian()
            self.n = old_n
        return original_h, new_h
