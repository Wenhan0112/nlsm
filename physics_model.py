import torch
import numpy as np
from typing import Optional
import tqdm

cpu = torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else cpu

class Physics_Model():
    def __init__(self,
            batch_size: Optional[int] = None,
            device=device):
        self.batch_size = batch_size if batch_size is not None else 1
        self.batched = batch_size is not None
        self.device = device

    def hamiltonian(self):
        hamiltonian = self._hamiltonian()
        assert hamiltonian.shape == (self.batch_size,), \
            "Hamiltonian shape does not match. "
        if not self.batched:
            return hamiltonian.item()
        return hamiltonian

    def _micro_ens_sim(self, beta, steps, show_bar=False):
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
        hamiltonian[:, 0] = self._hamiltonian()
        reject_count = 0
        for i in iterator:
            next_n = self.neighbor_generation()
            self.n, old_n = next_n, self.n
            next_hamiltonian = self._hamiltonian()
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
            # hamiltonian[:, i+1] = torch.where(not_accepted,
            #     hamiltonian[:, i], next_hamiltonian)
            hamiltonian[:, i+1] = self._hamiltonian()
            # all_ns[:, i+1] = self.n
        # print("Number of samples rejected", reject_count)
        print(f"Rejection ratio: {reject_count / steps * 100:.5f}")
        if reject_count == steps:
            print("All generations have been rejected!")
        # return hamiltonian, dhs, all_ns
        return hamiltonian

    def micro_ens_sim(self, beta, steps, show_bar=False):
        assert self.neighbor_gen_activated()
        hamiltonian = self._micro_ens_sim(
            beta, steps, show_bar=show_bar)
        if self.batched:
            return hamiltonian
        return hamiltonian[0]

    def gradient_descent(self, delta_t, steps, show_bar=False):
        hamiltonian = self._gd(delta_t, steps, show_bar=show_bar)
        if not self.batched:
            return hamiltonian[0]
        return hamiltonian

    def get_n(self):
        if self.batched:
            return self.n
        return self.n[0]

    def _test_generation_h_distribution(self, num_gen: int, show_bar=False):
        if show_bar:
            iterator = tqdm.trange(num_gen,
                desc="Neighbor generation Hamiltonian distribution")
        else:
            iterator = range(num_gen)
        original_h = self._hamiltonian()
        new_h = torch.zeros((self.batch_size, num_gen))
        for i in iterator:
            new_n = self.neighbor_generation()
            self.n, old_n = new_n, self.n
            new_h[:, i] = self._hamiltonian()
            self.n = old_n
        return original_h, new_h

    def test_generation_h_distribution(self, num_gen: int, show_bar=False):
        assert self.neighbor_gen_activated()
        original_h, new_h = self._test_generation_h_distribution(
            num_gen, sigma, num, show_bar=show_bar)
        if self.batched:
            return original_h, new_h
        return original_h[0], new_h[0]
