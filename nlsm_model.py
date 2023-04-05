import numpy as np
import torch
import torchgeometry
from typing import Optional
import physical_fn as pypf
import tqdm
import spin_model

cpu = torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else cpu

class NLSM_model(spin_model.Spin_Model):
    def __init__(self, 
            grid_size: float,
            num_sample: int,
            inter_layer_interaction,
            intra_layer_interaction: float = 1.,
            batch_size: int = 1,
            gen_prob: Optional[int] = None,
            gen_sigma: Optional[float] = None,
            metal_distance: float = 1.,
            num_bzone: int = 0,
            device=device,
            gpu_memory=0
        ):
        super().__init__(batch_size=batch_size,
            gen_sigma=gen_sigma, gen_prob=gen_prob, device=device)
        self.l = num_sample
        self.intra = intra_layer_interaction
        self.inter = inter_layer_interaction.clone().to(device=device, dtype=torch.double)
        self.metal_distance = metal_distance
        self.grid_size = grid_size
        self.num_bzone = num_bzone
        self.mesh_area = (self.grid_size / self.l)**2
        self.gpu_memory = gpu_memory
        

    def initialize(self, n: Optional[torch.Tensor] = None):
        self.set_electrostatic_potential_kernel()
        size = (self.batch_size, 2, self.l, self.l)
        if n is None:
            self.n = spin_model.uniform_sphere(3, size=size, device=self.device)
        else:
            self.n = n.clone().to(device=self.device, dtype=torch.double)
            self.n = self.n.reshape(size + (3,))

    def get_grad_n_h(self, grad=None, n=None):
        if grad is None:
            h, grad = self.hamiltonian(compute_grad=True, n=n)
            if n is None:
                return grad, self.n, h
            else:
                return grad, n, h
        else:
            return grad, n, None

    def get_grad(self, grad=None, n=None):
        if grad is None:
            _, grad = self.hamiltonian(compute_grad=True, n=n)
            return grad
        return grad


    def hamiltonian(self, compute_grad=False, n=None):
        if n is None:
            n = self.n
        if compute_grad:
            n.requires_grad_(True)
        x_grad = n.roll(-1, -3) - n
        y_grad = n.roll(-1, -2) - n
        intra_hamiltonian = torch.sum(x_grad ** 2, dim=(-4,-3,-2,-1)) \
            + torch.sum(y_grad ** 2, dim=(-4,-3,-2,-1))
        intra_hamiltonian *= self.intra
        inter_hamiltonian = torch.sum(self.inter * (n[..., 0, :, :, :] - n[..., 1, :, :, :])**2,
            dim=(-3,-2,-1)) * (self.mesh_area / 4 / np.pi**2)
        ee = self.electrostatic_energy(n=n)
        print(intra_hamiltonian.item(), inter_hamiltonian.item(), ee.item())
        hamiltonian = intra_hamiltonian + inter_hamiltonian + ee
        if compute_grad:
            loss = hamiltonian.sum()
            loss.backward()
            hamiltonian.detach_()
            hamiltonian.requires_grad_(False)
            n.requires_grad_(False)
            return hamiltonian, n.grad
        else:
            return hamiltonian


    def evolution_matrix(self, delta_t: float, grad=None, n=None):
        """
        Calculate the NLSM rotation matrix. Only consider nearest neighbor
            interactions.
        @params delta_t (float): The time step of the evolution.
            Constraints: delta_t > 0
        """
        grad = self.get_grad(grad=grad, n=n)
        rot = grad * delta_t
        rot[..., 1, :, :, :] = -rot[..., 1, :, :, :]
        rot = pypf.angle_axis_to_rot_mat(rot)
        return rot

    def evolution(self, delta_t, steps, show_bar=False):
        bs = self.n.shape[:-4]
        hamiltonian = torch.zeros(bs + (steps+1,), device=self.device,
            dtype=torch.double)
        if show_bar:
            iterator = tqdm.trange(steps, desc="NLSM Evolution")
        else:
            iterator = range(steps)
        for i in iterator:
            hamiltonian[:, i], g0 = self.hamiltonian(compute_grad=True, n=self.n)
            em0 = self.evolution_matrix(delta_t/2, grad=g0)
            n1 = torch.matmul(em0, self.n.unsqueeze(-1)).squeeze(-1)
            h1, g1 = self.hamiltonian(compute_grad=True, n=n1)
            em1 = self.evolution_matrix(delta_t, grad=g1)
            self.n = torch.matmul(em1, self.n.unsqueeze(-1)).squeeze(-1)
        hamiltonian[:, -1] = self.hamiltonian()
        return hamiltonian

    def gd_matrix(self, delta_t, grad=None, n=None):
        """
        Calculate the NLSM gradient descent rotation matrix. Only consider nearest
            neighbor interactions.
        @params delta_t (float): The time step of the evolution.
            Constraints: delta_t > 0
        @return (numpy.ndarray): The NLSM rotation matrix.
            Constraints: RETURN.shape == (2, n.shape[1], n.shape[1], 3, 3)
        """
        grad, n, _ = self.get_grad_n_h(grad=grad, n=n)        
        l = self.l
        rot = torch.cross(n, grad, dim=-1) * delta_t
        rot = pypf.angle_axis_to_rot_mat(rot)
        return rot


    def gd(self, delta_t, steps, show_bar=False):
        bs = self.n.shape[:-4]
        hamiltonian = torch.zeros(bs + (steps+1,), device=self.device,
            dtype=torch.double)
        if show_bar:
            iterator = tqdm.trange(steps, desc="NLSM Gradient Descent")
        else:
            iterator = range(steps)
        for i in iterator:
            hamiltonian[:, i], grad = self.hamiltonian(compute_grad=True)
            self.n = torch.matmul(
                self.gd_matrix(delta_t),
                self.n.unsqueeze(-1)
            ).squeeze(-1)
        hamiltonian[:, -1] = self.hamiltonian()
        return hamiltonian

    def skyrmion_density(self, n=None):
        if n is None:
            n = self.n
        n_x, n_y = n.roll(-1, -3), n.roll(-1, -2)
        density_1 = torch.sum(n * torch.cross(n_x, n_y, axis=-1), axis=-1)
        density_1 /= 1. + torch.sum(n * n_x, axis=-1) \
            + torch.sum(n_x * n_y, axis=-1) \
            + torch.sum(n_y * n, axis=-1)
        n_x, n_y = n.roll(1, -3), n.roll(1, -2)
        density_2 = torch.sum(n * torch.cross(n_x, n_y, axis=-1), axis=-1)
        density_2 /= 1. + torch.sum(n * n_x, axis=-1) \
            + torch.sum(n_x * n_y, axis=-1) \
            + torch.sum(n_y * n, axis=-1)
        density = (density_1.arctan() + density_2.arctan()) / (2 * np.pi * self.mesh_area) 
        density[..., 1, :, :] = -density[..., 1, :, :]
        return density

    def current_density(self, grad=None, n=None):
        if n is None:
            n = self.n
        if grad is None:
            _, grad = self.hamiltonian(n=n)
        curr_d = torch.zeros(n.shape[:-4] + (self.l, self.l, 2), device=self.device)
        nd = n.roll(-1, -3) - n
        j = torch.sum(nd * grad, axis=-1)
        curr_d[..., 0] = j[..., 1, :, :] - j[..., 0, :, :]
        nd = n.roll(-1, -2) - n
        j = torch.sum(nd * grad, axis=-1)
        curr_d[..., 1] = j[..., 0, :, :] - j[..., 1, :, :]
        return curr_d

        

    def set_electrostatic_potential_kernel(self):
        if self.gpu_memory > 0:
            recip = torch.arange(self.l, device=self.device) * (2*np.pi / self.grid_size)
            self.potential_kernel = torch.zeros((self.l, self.l), device=self.device)
            for i in range(-self.num_bzone, self.num_bzone):
                recip_x2 = (recip + (2*np.pi*self.l*i/self.grid_size))**2
                for j in range(-self.num_bzone, self.num_bzone):
                    recip_y2 = (recip + (2*np.pi*self.l*j/self.grid_size))**2
                    recip_dist = torch.sqrt(recip_x2[:, None] + recip_y2[None, :])
                    potential_kernel = self.metal_distance * pypf.tanhc(recip_dist * self.metal_distance) 
                    potential_kernel *= torch.exp(-recip_dist**2/2)
                    self.potential_kernel += potential_kernel
        else:
            recip = torch.arange(self.l, device=self.device) * (2*np.pi / self.grid_size)
            block_idx = torch.arange(-self.num_bzone, self.num_bzone, device=self.device) * (2*np.pi*self.l/self.grid_size)
            total_recip = recip[None, :] + block_idx[:, None]
            recip_dist2 = total_recip[:, None, :, None]**2 + total_recip[None, :, None, :]**2
            recip_dist = torch.sqrt(recip_dist2)
            self.potential_kernel = pypf.tanhc_var1(recip_dist, self.metal_distance)
            self.potential_kernel *= torch.exp(-recip_dist2/2)
            self.potential_kernel = self.potential_kernel.sum(axis=(0,1)) 
        self.potential_kernel *= self.mesh_area**2 / (4*np.pi*self.grid_size**2)

        # print(self.potential_kernel)


    def electrostatic_energy(self, n=None):
        if n is None:
            n = self.n
        density = self.skyrmion_density(n=n).sum(axis=-3)
        density_dft = torch.fft.fft2(density)
        energy = density_dft.abs()**2 * self.potential_kernel
        energy = energy.sum(axis=(-2, -1))
        return energy

    def skyrmion_count(self):
        density = self.skyrmion_density()
        count = torch.sum(density, axis=(-3,-2,-1)) * self.mesh_area
        return count



