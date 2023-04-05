import numpy as np
import torch
import torchgeometry
from typing import Optional
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
            batch_size: Optional[int] = None,
            gen_prob: Optional[int] = None,
            gen_sigma: Optional[float] = None,
            metal_distance: float = 1.,
            num_bzone: int = 0,
            device=device,
            enable_grad: bool = False,
            gpu_memory=0):
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
        self.enable_grad = enable_grad
        

    def initialize(self, n: Optional[torch.Tensor] = None):
        self.set_electrostatic_potential_kernel()
        size = (self.batch_size, 2, self.l, self.l)
        if n is None:
            self.n = spin_model.uniform_sphere(3, size=size, device=self.device)
        else:
            self.n = n.clone().to(device=self.device, dtype=torch.double)
            self.n = self.n.reshape(size + (3,))
        if self.enable_grad:
            self.n.requires_grad_()

    def _hamiltonian(self):
        n = self.n
        x_grad = n.roll(-1, -3) - n
        y_grad = n.roll(-1, -2) - n
        intra_hamiltonian = torch.sum(x_grad ** 2, dim=(-4,-3,-2,-1)) \
            + torch.sum(y_grad ** 2, dim=(-4,-3,-2,-1))
        intra_hamiltonian *= self.intra
        # print(hamiltonian)
        inter_hamiltonian = torch.sum(self.inter * (n[..., 0, :, :, :] - n[..., 1, :, :, :])**2,
            dim=(-3,-2,-1)) * (self.mesh_area / 4 / np.pi**2)
        ee = self._electrostatic_energy()
        print(intra_hamiltonian.item(), inter_hamiltonian.item(), ee.item())
        return intra_hamiltonian + inter_hamiltonian + ee
        # return hamiltonian


    def _force(self):
        """
        Calculate the NLSM force matrix. Only consider nearest neighbor
            interactions.
        @params n (numpy.ndarray): The input array.
            Constraints: n.shape == (2, n.shape[1], n.shape[1], 3)
        @params interaction_j (numpy.ndarray): Principle components of
            interaction strength.
            Constraints: interaction_j.shape == (3,)
        @return (numpy.ndarray): The NLSM rotation matrix.
            Constraints: RETURN.shape == (2, n.shape[1], n.shape[1], 3, 3)
        """
        n = self.n
        f = n.roll(-1, -3) + n.roll(1, -3) + n.roll(-1, -2) + n.roll(1, -2)
        f *= 2 * self.intra
        f -= n.flip(-4) * self.inter
        return f

    def force(self):
        f = self._force()
        if not self.batched:
            return f[0]
        return f

    def _evolution_matrix(self, delta_t: float):
        """
        Calculate the NLSM rotation matrix. Only consider nearest neighbor
            interactions.
        @params n (numpy.ndarray): The input array.
            Constraints: n.shape == (2, n.shape[1], n.shape[1], 3)
        @params delta_t (float): The time step of the evolution.
            Constraints: delta_t > 0
        @params interaction_j (numpy.ndarray): Principle components of
            interaction strength.
            Constraints: interaction_j.shape == (3,)
        @return (numpy.ndarray): The NLSM rotation matrix.
            Constraints: RETURN.shape == (2, n.shape[1], n.shape[1], 3, 3)
        """
        f = self._force().reshape((-1, 3))
        rot_vec = f * delta_t
        rot_vec[:, 1] = -rot_vec[:, 1]
        rot_mat = torchgeometry.angle_axis_to_rotation_matrix(rot_vec)
        rot_mat = rot_mat[:, :3, :3].reshape(
            (self.batch_size, 2, self.l, self.l, 3, 3))
        return rot_mat

    def evolution_matrix(self, delta_t):
        rot_mat = self._evolution_matrix(delta_t)
        if not self.batched:
            return rot_mat[0]
        return rot_mat

    def _evolution(self, delta_t, steps, show_bar=False):
        bs = self.batch_size
        l = self.l
        hamiltonian = torch.zeros((bs, steps + 1), device=self.device,
            dtype=torch.double)
        hamiltonian[:, 0] = self._hamiltonian()
        if show_bar:
            iterator = tqdm.trange(steps, desc="NLSM Evolution")
        else:
            iterator = range(steps)
        for i in iterator:
            old_n = self.n
            n1 = torch.matmul(self._evolution_matrix(delta_t/2),
                old_n.unsqueeze(-1))
            n1 = n1.squeeze(-1)
            self.n = n1
            n2 = torch.matmul(self._evolution_matrix(delta_t),
                old_n.unsqueeze(-1))
            self.n = n2.squeeze(-1)
            hamiltonian[:, i+1] = self._hamiltonian()
        return hamiltonian

    def _gd_matrix(self, delta_t):
        """
        Calculate the NLSM gradient descent rotation matrix. Only consider nearest
            neighbor interactions.
        @params delta_t (float): The time step of the evolution.
            Constraints: delta_t > 0
        @return (numpy.ndarray): The NLSM rotation matrix.
            Constraints: RETURN.shape == (2, n.shape[1], n.shape[1], 3, 3)
        """
        f = self._force().reshape((-1, 3))
        l = self.l
        rot_vec = torch.cross(self.n.reshape((-1, 3)), f) * delta_t
        rot_mat = torchgeometry.angle_axis_to_rotation_matrix(rot_vec)
        rot_mat = rot_mat[:, :3, :3].reshape(
            (self.batch_size, 2, l, l, 3, 3))
        return rot_mat

    def gradient_descent_matrix(self, delta_t):
        rot_mat = self._gd_matrix(delta_t)
        if not self.batched:
            return rot_mat[0]
        return rot_mat

    def _gd(self, delta_t, steps, show_bar=False):
        bs = self.batch_size
        l = self.l
        hamiltonian = torch.zeros((bs, steps + 1))
        hamiltonian[:, 0] = self._hamiltonian()
        for i in tqdm.trange(steps, desc="Gradient descent"):
            self.n = torch.matmul(
                self._gd_matrix(delta_t),
                self.n.reshape((bs, 2, l, l, 3, 1))
            ).reshape((bs, 2, l, l, 3))
            hamiltonian[:, i+1] = self._hamiltonian()
        return hamiltonian

    def _skyrmion_density(self):
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

    def skyrmion_density(self):
        density = self._skyrmion_density()
        if not self.batched:
            return density[0]
        return density

    def set_electrostatic_potential_kernel(self):
        # l = self.l
        # half_l = l // 2
        # bzone_coord = torch.arange(-self.num_bzone, self.num_bzone+1,
        #     dtype=torch.double, device=self.device) * (2 * np.pi)
        # reciprocal_coord_x = torch.arange(half_l, l + half_l,
        #     dtype=torch.double, device=self.device)
        # reciprocal_coord_x = reciprocal_coord_x.remainder(l) - half_l
        # reciprocal_coord_x *= 2 * np.pi / l
        # reciprocal_coord_x = reciprocal_coord_x[:, None] + bzone_coord
        # reciprocal_coord_x = reciprocal_coord_x[:, None, :, None]
        # reciprocal_coord_y = torch.arange(0, half_l + 1,
        #     dtype=torch.double, device=self.device)
        # reciprocal_coord_y *= 2 * np.pi / l
        # reciprocal_coord_y = reciprocal_coord_y[:, None] + bzone_coord
        # reciprocal_coord_y = reciprocal_coord_y[None, :, None, :]
        # reciprocal_distances = torch.sqrt(reciprocal_coord_x**2 + reciprocal_coord_y**2)
        # potential_kernel = \
        #     torch.tanh(reciprocal_distances * self.metal_distance / 2.) \
        #     / reciprocal_distances / (2. * self.electric_constant) \
        #     * torch.exp(-0.5 * (reciprocal_distances * self.moire_length) ** 2)
        # self.potential_kernel = potential_kernel.sum(axis=(-2, -1))
        if self.gpu_memory > 0:
            recip = torch.arange(self.l, device=self.device) * (2*np.pi / self.grid_size)
            self.potential_kernel = torch.zeros((self.l, self.l), device=self.device)
            for i in range(-self.num_bzone, self.num_bzone):
                recip_x2 = (recip + (2*np.pi*self.l*i/self.grid_size))**2
                for j in range(-self.num_bzone, self.num_bzone):
                    recip_y2 = (recip + (2*np.pi*self.l*j/self.grid_size))**2
                    recip_dist = torch.sqrt(recip_x2[:, None] + recip_y2[None, :])
                    potential_kernel = self.metal_distance * tanhc(recip_dist * self.metal_distance) 
                    potential_kernel *= torch.exp(-recip_dist**2/2)
                    self.potential_kernel += potential_kernel
        else:
            recip = torch.arange(self.l, device=self.device) * (2*np.pi / self.grid_size)
            block_idx = torch.arange(-self.num_bzone, self.num_bzone, device=self.device) * (2*np.pi*self.l/self.grid_size)
            total_recip = recip[None, :] + block_idx[:, None]
            recip_dist2 = total_recip[:, None, :, None]**2 + total_recip[None, :, None, :]**2
            recip_dist = torch.sqrt(recip_dist2)
            self.potential_kernel = self.metal_distance * tanhc(recip_dist * self.metal_distance)
            self.potential_kernel *= torch.exp(-recip_dist2/2)
            self.potential_kernel = self.potential_kernel.sum(axis=(0,1)) 
        self.potential_kernel *= self.mesh_area**2 / (4*np.pi*self.grid_size**2)

        # print(self.potential_kernel)

        


    def _electric_potential(self, *, density=None):
        if density is None:
            density = self._skyrmion_density().sum(axis=-3)
        # density_fourier = torch.fft.rfft2(density, axis=(-2, -1))
        # potential_fourier = self.potential_kernel * density_fourier
        # potential = torch.fft.irfft2(potential_fourier, density.shape[-2:])
        # return potential

        density_fourier = torch.fft.fft2(density, axis=(-2, -1)) 
        potential_fourier = density_fourier * self.potential_kernel
        potential = torch.fft.ifft2(potential_fourier) / self.mesh_area
        return potential

        




    def electric_potential(self):
        potential = self._electric_potential()
        if not self.batched:
            return potential[0]
        return potential

    def _electrostatic_energy(self):
        density = self._skyrmion_density().sum(axis=-3)
        density_dft = torch.fft.fft2(density)
        energy = density_dft.abs()**2 * self.potential_kernel
        energy = energy.sum(axis=(-2, -1))
        return energy

    def electrostatic_energy(self):
        energy = self._electrostatic_energy()
        if not self.batched:
            return energy[0]
        return energy

    def _skyrmion_count(self):
        density = self._skyrmion_density()
        if self.boundary == "open":
            raise ValueError("Invalid to count skyrmion in open boundary!")
        elif self.boundary == "circular":
            count = torch.sum(density, axis=(-3,-2,-1))
        return count

    def skyrmion_count(self):
        count = self._skyrmion_count()
        if not self.batched:
            return count[0]
        return count


def tanhc(x):
    return torch.where(x != 0, torch.tanh(x) / x, 1.)
