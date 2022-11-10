import numpy as np
import torch
import torchgeometry
from typing import Optional
import tqdm
import spin_model

cpu = torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else cpu

class NLSM_model(spin_model.Spin_Model):
    def __init__(self, l: int,
            inter_layer_interaction,
            intra_layer_interaction: float = 1.,
            boundary="open",
            batch_size: Optional[int] = None,
            gen_prob: Optional[int] = None,
            gen_sigma: Optional[float] = None,
            metal_distance: float = 1.,
            electric_constant: float = np.inf,
            num_bzone: int = 1,
            device=device):
        super().__init__(batch_size=batch_size,
            gen_sigma=gen_sigma, gen_prob=gen_prob, device=device)
        self.l = l
        self.intra = intra_layer_interaction
        self.inter = torch.tensor(inter_layer_interaction, device=device, dtype=torch.double)
        self.boundary = boundary.lower()
        self.metal_distance = metal_distance
        self.electric_constant = electric_constant
        self.num_bzone = num_bzone
        assert self.boundary in ["open", "circular"]

    def initialize(self, n: Optional[torch.Tensor] = None):
        size = (self.batch_size, 2, self.l, self.l)
        if n is None:
            self.n = spin_model.uniform_sphere(3, size=size, device=self.device)
        else:
            self.n = torch.tensor(n, device=self.device, dtype=torch.double)
            self.n = self.n.reshape(size + (3,))

    def _hamiltonian(self):
        n = self.n
        if self.boundary == "open":
            x_grad = n[:, :, 1:, :] - n[:, :, :-1, :]
            y_grad = n[:, :, :, 1:] - n[:, :, :, :-1]
        elif self.boundary == "circular":
            x_grad = n.roll(-1, -3) - n
            y_grad = n.roll(-1, -2) - n
        hamiltonian = torch.sum(x_grad ** 2, dim=(-4,-3,-2,-1)) \
            + torch.sum(y_grad ** 2, dim=(-4,-3,-2,-1))
        hamiltonian *= self.intra
        hamiltonian += torch.sum(self.inter * n[:, 0] * n[:, 1],
            dim=(-3,-2,-1))
        return hamiltonian + self._electrostatic_energy()


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
        if self.boundary == "open":
            f = torch.zeros_like(n)
            f[..., 1:, :, :] += n[..., :-1, :, :]
            f[..., :-1, :, :] += n[..., 1:, :, :]
            f[..., :, 1:, :] += n[..., :, :-1, :]
            f[..., :, :-1, :] += n[..., :, 1:, :]
        elif self.boundary == "circular":
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
        if self.boundary == "open":
            n_shape = n.shape
            l = n_shape[-2]
            n = n.reshape((-1, l, l, 3))
            dx, dy = torch.gradient(n, axis=(1,2))
            density = torch.sum(n * torch.cross(dx, dy, axis=-1), axis=-1)
            density = density.reshape(n_shape[:-1]) / (4 * np.pi)
        elif self.boundary == "circular":
            n_x, n_y = n.roll(-1, -3), n.roll(-1, -2)
            density_1 = torch.sum(n * torch.cross(nx, ny, axis=-1), axis=-1)
            density_1 /= 1. + torch.sum(n, n_x, axis=-1) \
                + torch.sum(n_x, n_y, axis=-1) \
                + torch.sum(n_y, n, axis=-1)
            n_x, n_y = n.roll(1, -3), n.roll(1, -2)
            density_2 = torch.sum(n * torch.cross(nx, ny, axis=-1), axis=-1)
            density_2 /= 1. + torch.sum(n, n_x, axis=-1) \
                + torch.sum(n_x, n_y, axis=-1) \
                + torch.sum(n_y, n, axis=-1)
            density = (density_1.arctan() + density_2.arctan()) / (2 * np.pi)
        density[..., 1, :, :] = -density[..., 1, :, :]
        return density

    def skyrmion_density(self):
        density = self._skyrmion_density()
        if not self.batched:
            return density[0]
        return density

    def _electric_potential(self, *, density=None):
        if density is None:
            density = self._skyrmion_density().sum(axis=-3)
        density_fourier = torch.rfft2(density, axis=(-2, -1))
        l = self.l
        half_l = l // 2
        bzone_coord = torch.arange(-self.num_bzone, self.num_bzone+1) * (2*np.pi)
        reciprocal_coord_x = torch.arange(half_l, l + half_l).remainder(l) - half_l
        reciprocal_coord_x *= 2 * np.pi / l
        reciprocal_coord_x = reciprocal_coord_x[:, None] + bzone
        reciprocal_coord_x = reciprocal_coord_x[:, None, :, None]
        reciprocal_coord_y = torch.arange(0, half_l + 1)
        reciprocal_coord_y *= 2 * np.pi / l
        reciprocal_coord_y = reciprocal_coord_y[:, None] + bzone
        reciprocal_coord_y = reciprocal_coord_y[None, :, None, :]
        reciprocal_distances = torch.sqrt(reciprocal_coord_x**2 + reciprocal_coord_y**2)
        potential_kernel = \
            torch.tanh(reciprocal_distances * self.metal_distance / 2.) \
            / reciprocal_distances / (2. * self.electric_constant) \
            * torch.exp(-0.5 * (reciprocal_distances * self.moire_length) ** 2)
        potential_kernel = potential_kernel.sum(axis=(-2, -1))
        potential_fourier = potential_kernel * density_fourier
        potential = torch.irfft2(potential_fourier, density.shape[-2:])
        return potential

    def electric_potential(self):
        potential = self._electric_potential()
        if not self.batched:
            return potential[0]
        return potential

    def _electrostatic_energy(self):
        density = self._skyrmion_density().sum(axis=-3)
        potential = self._electric_potential(density=density)
        energy = torch.sum(density * potential, axis=(-2, -1))
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
