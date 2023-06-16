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
    """
    A NLSM model of MATBG Chern band +1 and -1.

    Eq.1: Hamiltonian of the system
    
    Eq.2: Coulumb interaction energy
    $$
    E_C = \frac{e^2}{4\pi\epsilon l_B}
    $$

    @field l (int): the number of lattice sites in one dimension. 
    @field intra (float): The intra-layer interaction coefficient $g$ in Eq.1 
        in terms of $E_C$ in Eq.2. 
    @field inter (torch.tensor): The inter-layer interaction coefficients 
        $J_i$ in Eq.1. 
        inter[i]: The inter-layer interaction coefficient $J_i$.
    @field metal_distance (float): The metal gate distance $d$ in terms of 
        $l_B$.
    @field grid_size (float): The length of the grid in one dimension in terms 
        of $l_B$.
    @field num_bzone (int): The number of Broullain zone in one dimension used 
        to calculate the electrostatic potential kernel.
    @field mesh_area (float): The area of the meshgrid in terms of $l_B^2$. 
    @field gpu_memory (int): Indicator of level of GPU memory. General rule of 
        thumb: the large the index, the smaller amount of GPU memory used. 
        Smallest index is 0, assuming no memory limit.  
    @field n (torch.Tensor): The current state of the system. T
        CONSTRAINT: n.shape == (self.batch_size, 2, self.l, self.l, 3)
        n[i,g,x,y,k]: The k-th spin compoennt in i-th model with band g and 
            lattice position (x,y)
    @field potential_kernel (torch.Tensor): The potential kernel evaluated at 
        integer multiples of wavevectors.
        CONSTRAINT potential_kernel.shape == () 
        
    """
    def __init__(self, 
            grid_size: float,
            num_sample: int,
            inter_layer_interaction,
            intra_layer_interaction: float = 1.,
            batch_size: int = 1,
            gen_prob: Optional[float] = None,
            gen_sigma: Optional[float] = None,
            metal_distance: float = 1.,
            num_bzone: int = 0,
            device=device,
            gpu_memory=0
        ):
        """
        Constructor
        @params grid_size (float): The length of the grid in one dimension in 
            terms of $l_B$.
            CONSTRAINT: grid_size > 0
        @params num_sample (int): the number of lattice sites in one 
            dimension. 
            CONSTRAINT: num_sample > 1
        @params inter_layer_interaction (torch.Tensor): The inter-layer 
            interaction coefficients $J_i$ in Eq.1. 
            CONSTRAINT: inter_layer_interaction.shape == (3,)
            inter[i]: The inter-layer interaction coefficient $J_i$.
        @params intra_layer_interaction (float): The intra-layer interaction 
            coefficient $g$ in Eq.1 in terms of $E_C$ in Eq.2. 
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
        @params metal_distance (float): The metal gate distance $d$ in terms 
            of $l_B$.
            CONSTRAINT: metal_distance > 0
        @params device (torch.device): The device where the states are stored. 
        @params gpu_memory (int): Indicator of level of GPU memory. General 
            rule of thumb: the large the index, the smaller amount of GPU 
            memory used. Smallest index is 0, assuming no memory limit. If 
            index is greater than 0, memory similar to that of the state is 
            used. 
            CONSTRAINT: gpu_memory >= 0
        """
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
        """
        Initialize the state of the system.
        @params n (Optional[torch.Tensor], DEFAULT None): if this is not None, 
            the system is initialized according to this state. For each 
            model, each band, and each lattice site, the spin must be 
            normalized. Otherwise, the spin is uniformly generated on the unit 
            sphere. 
            CONSTRAINT: n.shape == (self.batch_size, 2, self.l, self.l, 3)
            CONSTRAINT: torch.all(torch.linalg.norm(n, axis=-1) == 1).item()
        """
        self.set_electrostatic_potential_kernel()
        size = (self.batch_size, 2, self.l, self.l)
        if n is None:
            self.n = spin_model.uniform_sphere(3, 
                size=size, device=self.device)
        else:
            if tuple(n.shape) != size + (3,):
                raise ValueError(
                    f"Input state shape {n.shape} is not correct !")
            self.n = n.clone().to(device=self.device, dtype=torch.double)
            # self.n = self.n.reshape(size + (3,))

    def get_grad_n_h(self, 
            grad: Optional[torch.Tensor] = None, 
            n: Optional[torch.Tensor] = None
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the gradient of a state, and possibly the state and the 
        corresponding Hamiltonian. 
        @params grad (Optional[torch.Tensor], DEFAULT None): The gradient of 
            the state respect to the Hamiltonian. If this is specified, no 
            Hamiltonian computation is made. 
            CONSTRAINT: grad.shape == self.n.shape
        @params n (Optional[torch.Tensor], DEFAULT None): The state used to 
            get the gradient and the Hamiltonian. If this is not specified, 
            self.n is used as the state. 
            CONSTRAINT: n.shape == self.n.shape
            CONSTRAINT: torch.all(torch.linalg.norm(n, axis=-1) == 1).item()
        @return (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): The 
            gradient of the state respect to the Hamiltonian, the state, and 
            the Hamiltonian in sequence.
            CONSTRAINT: RETURN[0].shape == RETURN[1].shape 
            CONSTRAINT: RETURN[0].shape == RETURN[2].shape 
        """
        if grad is None:
            h, grad = self.hamiltonian(compute_grad=True, n=n)
            if n is None:
                return grad, self.n, h
            else:
                return grad, n, h
        else:
            return grad, n, None

    def get_grad(self, 
            grad: Optional[torch.Tensor] = None, 
            n: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """
        Get the gradient of a state.
        @params grad (Optional[torch.Tensor], DEFAULT None): The gradient of 
            the state respect to the Hamiltonian. If this is specified, no 
            Hamiltonian computation is made. The returned value is itself. 
            CONSTRAINT: grad.shape == self.n.shape
        @params n (Optional[torch.Tensor], DEFAULT None): The state used to 
            get the gradient and the Hamiltonian. If this is not specified, 
            self.n is used as the state. 
            CONSTRAINT: n.shape == self.n.shape
            CONSTRAINT: torch.all(torch.linalg.norm(n, axis=-1) == 1).item()
        @return (torch.Tensor): The gradient of the state respect to the 
            Hamiltonian. 
            CONSTRAINT: RETURN.shape == self.n.shape
        """
        if grad is None:
            _, grad = self.hamiltonian(compute_grad=True, n=n)
            return grad
        return grad


    def hamiltonian(self, compute_grad: bool = False, 
            n: Optional[torch.Tensor] = None
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the Hamiltonian of the system.
        @params compute_grad (bool, DEFAULT False): True iff the gradient of 
            the state respect to tha Hamiltonian is calculated. If true, the 
            return type is tuple[torch.Tensor, torch.Tensor], where the zeroth 
            element is the hamiltonian and the first element is the gradient. 
            If false, the return type is torch.Tensor, which is just the 
            Hamiltonian. 
        @params n (Optional[torch.Tensor]): The state of the system. If not 
            specified, self.n is used. 
        @return (torch.Tensor | tuple[torch.Tensor, torch.Tensor]): If 
            compute_grad is true, it is a tuple of the Hamiltonian and the 
            gradient respect to the Hamiltonain. If compute_grad is False, it 
            is a tuple of the Hamiltonian and the gradient. 
            CONSTRAINT: not compute_grad or (isinstance(RETURN, tuple) 
                and RETURN[0].shape == torch.Size([self.batch_size]) 
                and RETURN[1].shape == self.n.shape)
            CONSTRAINT: compute_grad or \
                RETURN.shape == torch.Size([self.batch_size])
        """
        if n is None:
            n = self.n
        if compute_grad:
            # Set enable autograd on the state
            n.requires_grad_(True)
        x_grad = n.roll(-1, -3) - n
        y_grad = n.roll(-1, -2) - n
        intra_hamiltonian = torch.sum(x_grad ** 2, dim=(-4,-3,-2,-1)) \
            + torch.sum(y_grad ** 2, dim=(-4,-3,-2,-1))
        intra_hamiltonian *= self.intra
        inter_hamiltonian = torch.sum(self.inter * (n[..., 0, :, :, :] + n[..., 1, :, :, :])**2,
            dim=(-3,-2,-1)) * (self.mesh_area / 4 / np.pi**2)
        ee = self.electrostatic_energy(n=n)
        hamiltonian = intra_hamiltonian + inter_hamiltonian + ee
        # hamiltonian = intra_hamiltonian + inter_hamiltonian
        if compute_grad:
            # Compute the gradient via Autograd
            loss = hamiltonian.sum()
            loss.backward()
            hamiltonian.detach_()
            hamiltonian.requires_grad_(False)
            n.requires_grad_(False)
            return hamiltonian, n.grad
        else:
            return hamiltonian


    def evolution_matrix(self, step_size: float, 
            grad: Optional[torch.Tensor] = None, 
            n: Optional[torch.Tensor] = None
        ):
        """
        Calculate the NLSM rotation matrix. Only consider nearest neighbor
            interactions.
        @params step_size (float): The stepsize of the evolution.
            CONSTRAINT: step_size > 0
        @params grad (Optional[torch.Tensor], DEFAULT None): The gradient of 
            the state respect to the Hamiltonian. If this is specified, no 
            Hamiltonian computation is made. 
            CONSTRAINT: grad.shape == self.n.shape
        @params n (Optional[torch.Tensor], DEFAULT None): The state used to 
            get the gradient and the Hamiltonian. If this is not specified, 
            self.n is used as the state. 
            CONSTRAINT: n.shape == self.n.shape
            CONSTRAINT: torch.all(torch.linalg.norm(n, axis=-1) == 1).item()
        @return (torch.Tensor): The evolution matrix of each lattice site.
            CONSTRAINT: RETURN.shape == self.n.shape + (3,)
        """
        grad = self.get_grad(grad=grad, n=n)
        rot = grad * step_size
        rot[..., 1, :, :, :] = -rot[..., 1, :, :, :]
        rot = pypf.angle_axis_to_rot_mat(rot)
        return rot
    
    def evolution_update(self, step_size: float, n=None, grad_n=None, return_prev_hamiltonian=False):
        """
        A single evolution update step.
        @params step_size ()
        """
        if n is None:
            n = self.n
        if grad_n is None:
            grad_n = self.n
        h, grad = self.hamiltonian(compute_grad=True, n=grad_n)
        em = self.evolution_matrix(step_size, grad=grad)
        new_n = torch.matmul(em, n.unsqueeze(-1)).squeeze(-1)
        if return_prev_hamiltonian:
            return new_n, h
        return new_n

    def evolution(self, delta_t, steps, show_bar=False):
        """
        Deprecated!
        """
        raise DeprecationWarning(
            "Evolution method deprecated! Use evolver to evolve the model. "
        )
        bs = self.n.shape[:-4]
        hamiltonian = torch.zeros(bs + (steps+1,), device=self.device,
            dtype=torch.double)
        if show_bar:
            iterator = tqdm.trange(steps, desc="NLSM Evolution")
        else:
            iterator = range(steps)
        for i in iterator:
            hamiltonian[:, i], g0 = self.hamiltonian(
                compute_grad=True, n=self.n
            )
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
        return density[..., 0, :, :] - density[..., 1, :, :]

    def current_density(self, grad=None, n=None):
        if n is None:
            n = self.n
        if grad is None:
            _, grad = self.hamiltonian(n=n, compute_grad=True)
        curr_d = torch.zeros(n.shape[:-4] + (self.l, self.l, 2), device=self.device)
        nd = n.roll(-1, -3) - n
        curr_d_x = torch.sum(nd * grad, axis=-1)
        curr_d_x = curr_d_x[..., 0, :, :, :] - curr_d_x[..., 1, :, :, :]
        # curr_d[..., 0] = -torch.sum(nd * grad, axis=(-1, -4))
        nd = n.roll(-1, -2) - n
        # curr_d[..., 1] = torch.sum(nd * grad, axis=(-1, -4))
        curr_d_x = torch.sum(nd * grad, axis=-1)
        curr_d_x = curr_d_x[..., 1, :, :, :] - curr_d_x[..., 0, :, :, :]
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
        self.potential_kernel *= np.pi * self.mesh_area**2 / self.grid_size**2

        # print(self.potential_kernel)


    def electrostatic_energy(self, n=None):
        if n is None:
            n = self.n
        density = self.skyrmion_density(n=n)
        density_dft = torch.fft.fft2(density)
        energy = density_dft.abs()**2 * self.potential_kernel
        energy = energy.sum(axis=(-2, -1))
        return energy

    def skyrmion_count(self):
        density = self.skyrmion_density()
        count = torch.sum(density, axis=(-2,-1)) * self.mesh_area
        return count



