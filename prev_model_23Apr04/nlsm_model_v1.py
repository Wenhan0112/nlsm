import numpy as np
import scipy.spatial.transform
import scipy.stats
import matplotlib.pyplot as plt
from typing import Optional
import tqdm
from utils import WeightedLinearRegressor

LARGE_INT = 82381725
global_rng = np.random.default_rng()

def uniform_sphere(d: int, size=1, seed: Optional[int] = None) -> float:
    """
    Uniformly generate points on a unit sphere.
    @params d (int): The dimension of the sphere.
        Constraints: d >= 1
    @params size (int or sequence of ints): The shape of the samples.
        The output array shape is np.append(size, d)
    @params seed (Optional[int]): The seed for the random number generator.
        Constraint: seed for numpy.random.default_rng
    @return (np.ndarray): The samples from the uniform sphere.
        Constraint: RETURN.shape = size + (3,)
    """
    rng = np.random.default_rng(seed)

    if size == 1:
        n = rng.normal(size=d)
        return n / np.linalg.norm(n)
    else:
        n = rng.normal(size=np.append(size, d))
        norm = np.linalg.norm(n, axis=-1)
        result = n / np.stack([norm] * d, axis=-1)
        return result

def neighbor_generation(n: np.ndarray, sigma: float,
        seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rv = rng.normal(size=n.shape)
    n = n + rv * sigma
    norm = np.linalg.norm(n, axis=-1)
    result = n / np.stack([norm] * n.shape[-1], axis=-1)
    return result

class NLSM_model:
    INT = global_rng.integers(LARGE_INT)

    def __init__(self, l: int, interaction_j: np.ndarray,
            boundary="open", batch_size = None, seed=None):
        self.l = l
        self.j = interaction_j
        self.boundary = boundary.lower()
        self.batch_size = batch_size if batch_size is not None else 1
        self.batched = batch_size is not None
        self.rng = np.random.default_rng(seed)

    def initialize(self, n: Optional[np.ndarray] = None,
            seed: Optional[int] = None):
        size = (self.batch_size, 2, self.l, self.l)
        if n is None:
            self.n = uniform_sphere(3, size=size, seed=seed)
        else:
            self.n = n.copy().reshape(size + (3,))

    def _hamiltonian(self):
        n = self.n
        x_grad = -n[:, :, :-1, :] + n[:, :, 1:, :]
        y_grad = -n[:, :, :, :-1] + n[:, :, :, 1:]
        hamiltonian = np.sum(x_grad ** 2, axis=(-4,-3,-2,-1)) \
            + np.sum(y_grad ** 2, axis=(-4,-3,-2,-1))
        if self.boundary == "circular":
            x_grad_circ = n[:, :, 0, :] - n[:, :, -1, :]
            y_grad_circ = n[:, :, :, 0] - n[:, :, :, -1]
            hamiltonian += np.sum(x_grad_circ ** 2) + np.sum(y_grad_circ ** 2)
        hamiltonian += np.sum(self.j * n[:, 0] * n[:, 1],
            axis=(-3,-2,-1))
        return hamiltonian

    def hamiltonian(self):
        hamiltonian = self._hamiltonian()
        assert hamiltonian.shape == (self.batch_size,), \
            "Hamiltonian shape does not match. "
        if not self.batched:
            return float(hamiltonian)
        return hamiltonian

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
        f = np.zeros(n.shape)
        f[:, :, 1:, :] += n[:, :, :-1, :]
        f[:, :, :-1, :] += n[:, :, 1:, :]
        f[:, :, :, 1:] += n[:, :, :, :-1]
        f[:, :, :, :-1] += n[:, :, :, 1:]
        f *= 2
        f -= np.flip(n, axis=1) * self.j
        return f

    def force(self):
        f = self._force()
        if not self.batched:
            return f[0]
        return f

    def _evolution_matrix(self, delta_t):
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
        rot = scipy.spatial.transform.Rotation.from_rotvec(f * delta_t)
        rot_mat = rot.as_matrix().reshape(
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
        hamiltonian = np.zeros((bs, steps + 1))
        hamiltonian[:, 0] = self._hamiltonian()
        if show_bar:
            iterator = tqdm.trange(steps, desc="NLSM Evolution")
        else:
            iterator = range(steps)
        for i in iterator:
            old_n = self.n
            n1 = np.matmul(self._evolution_matrix(delta_t/2),
                old_n.reshape((bs,2,l,l,3,1)))
            n1 = n1.reshape((bs,2,l,l,3))
            self.n = n1
            n2 = np.matmul(self._evolution_matrix(delta_t),
                old_n.reshape((bs,2,l,l,3,1)))
            self.n = n2.reshape((bs,2,l,l,3))
            hamiltonian[:, i+1] = self._hamiltonian()
        return hamiltonian

    def evolution(self, delta_t, steps, show_bar=False):
        hamiltonian = self._evolution(delta_t, steps, show_bar=show_bar)
        if not self.batched:
            return hamiltonian[0]
        return hamiltonian

    def _gd_matrix(self, delta_t):
        """
        Calculate the NLSM gradient descent rotation matrix. Only consider nearest
            neighbor interactions.
        @params n (numpy.ndarray): The input array.
            Constraints: n.shape == (2, n.shape[1], n.shape[1], 3)
        @params delta_t (float): The time step of the evolution.
            Constraints: delta_t > 0
        @params interaction_j (numpy.ndarray): Principle components of interaction
            strength.
            Constraints: interaction_j.shape == (3,)
        @return (numpy.ndarray): The NLSM rotation matrix.
            Constraints: RETURN.shape == (2, n.shape[1], n.shape[1], 3, 3)
        """
        f = self._force().reshape((-1, 3))
        l = self.l
        rot = scipy.spatial.transform.Rotation.from_rotvec(
            np.cross(self.n.reshape((-1, 3)), f) * delta_t)
        rot_mat = rot.as_matrix().reshape(
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
        hamiltonian = np.zeros((bs, steps + 1))
        hamiltonian[:, 0] = self._hamiltonian()
        for i in tqdm.trange(steps, desc="Gradient descent"):
            self.n = np.matmul(
                self._gd_matrix(delta_t),
                self.n.reshape((bs, 2, l, l, 3, 1))
            ).reshape((bs, 2, l, l, 3))
            hamiltonian[:, i+1] = self._hamiltonian()
        return hamiltonian

    def gradient_descent(self, delta_t, steps, show_bar=False):
        hamiltonian = self._gd(delta_t, steps, show_bar=show_bar)
        if not self.batched:
            return hamiltonian[0]
        return hamiltonian

    def _micro_ens_sim(self, beta, steps, sigma, show_bar=False):

        if show_bar:
            iterator = tqdm.trange(steps,
                desc="Microcanonical Ensemble Simulation")
        else:
            iterator = range(steps)
        size = (self.batch_size, 2, self.l, self.l)
        hamiltonian = np.zeros((self.batch_size, steps + 1))
        hamiltonian[:, 0] = self._hamiltonian()
        for i in iterator:
            next_n = neighbor_generation(self.n, sigma, self.rng.integers(self.INT))
            self.n, old_n = next_n, self.n
            next_hamiltonian = self._hamiltonian()
            alpha = self.rng.random(size=self.batch_size)
            delta_hamiltonian = hamiltonian[:, i] - next_hamiltonian
            acceptance_rate = np.exp(beta * delta_hamiltonian)
            not_accepted = alpha > acceptance_rate
            self.n[not_accepted] = old_n[not_accepted]
            hamiltonian[:, i+1] = np.where(not_accepted,
                hamiltonian[:, i], next_hamiltonian)
        return hamiltonian

    def micro_ens_sim(self, beta, steps, sigma, show_bar=False):
        hamiltonian = self._micro_ens_sim(beta, steps, sigma, show_bar=show_bar)
        if self.batched:
            return hamiltonian
        return hamiltonian[0]

    def get_n(self):
        if self.batched:
            return self.n
        return self.n[0]

def fit_gaussian(data):
    """
    Fit the data to a Gaussian and validate the fit. Plot the fit.
    @params data (numpy.ndarray): The array of Hamiltonians.
        Constraints: data.ndim == 1
    """
    print("Fitting the data to the Gaussian")
    loc, scale = scipy.stats.norm.fit(data)
    print("Mean:", loc)
    print("Std:", scale)
    data_hist, bin_edges = np.histogram(data, bins="sqrt", density=True)
    bin_edges_center = (bin_edges[:-1] + bin_edges[1:]) / 2
    center = bin_edges.shape[0] // 2
    bin_width = bin_edges[center] - bin_edges[center - 1]
    data_hist_err = np.sqrt(data_hist / bin_width / data.shape[0])
    est_data_hist = scipy.stats.norm.pdf(bin_edges_center, loc, scale)
    data_res = np.where(data_hist_err == 0, 0,
        (est_data_hist - data_hist) / data_hist_err)
    chi_squared = np.sum(data_res**2) / (data_hist.shape[0] - 2)
    print("Chi-squared:", chi_squared)
    plt.hist(data, bins="sqrt", density=True, label="Sample PDF")
    # plt.plot(bin_edges_center, data_hist)
    plt.plot(bin_edges_center, est_data_hist, label="Fitted PDF")
    plt.xlabel("Hamiltonian")
    plt.ylabel("PDF")
    plt.tick_params(direction="in")
    plt.show()

def plot_hamiltonian(data: np.ndarray, title: Optional[str] = None,
        scale: Optional[str] = None):
    """
    Plot the time evolution of the Hamiltonian.
    @params data (numpy.ndarray): The time evolving Hamiltonian.
        Constraints: data.ndim in [1, 2]
    """
    if data.ndim == 1:
        plt.plot(data)
    elif data.ndim == 2:
        for i in range(data.shape[0]):
            plt.plot(data[i])
    else:
        raise ValueError
    plt.xlabel("Iteration")
    plt.ylabel("Hamiltonian")
    if title:
        plt.title(title)
    if scale:
        plt.yscale(scale)
        plt.ylabel("Hamiltonian Difference")
    plt.tick_params(direction="in")
    plt.show()

def plot_hist_hamiltonian(data, title: Optional[str] = None):
    """
    Plot the historgram of the Hamiltonian
    """
    data = data.flatten()
    plt.hist(data, bins="sqrt", density=True)
    plt.xlabel("Hamiltonian")
    plt.ylabel("PDF")
    if title:
        plt.title(title)
    plt.tick_params(direction="in")
    plt.show()

def visualize_state(n: np.ndarray):
    assert n.shape[-1] == 3, \
        f"Incorrect shape of visualization state {n.shape}!"
    if n.ndim == 1:
        n = n[None, None, None, ...]
    if n.ndim == 2:
        n = n[None, ]
    elif n.ndim == 3:
        n = n[None, ...]
    elif n.ndim > 4:
        assert False, f"Incorrect shape of visualization state {n.shape}!"

    ax = plt.figure().add_subplot(projection='3d')

    z, y, x = np.meshgrid(np.arange(n.shape[0]), np.arange(n.shape[1]),
        np.arange(n.shape[2]))


"""
MAIN FUNCTIONS
"""

def main_convergence():
    """ Total evolution time: 0.01 """
    total_time = 1e-1
    num_sample = 1000

    delta_ts = np.logspace(-4, -3, num=num_sample+2)[1:-1]
    init_delta_t = 1e-6
    """ Number of steps """
    steps = np.array(total_time / delta_ts, dtype=int)
    init_step = int(total_time / init_delta_t)
    """ Size of the grid """
    l = 20
    """ Interaction strength principle components """
    interaction_j = np.ones(3)
    """ Seed """
    seed = global_rng.integers(LARGE_INT)

    all_n = np.zeros((num_sample+1 , 2, l, l, 3))
    all_hamiltonian = np.zeros(num_sample + 1)

    model = NLSM_model(l=l, interaction_j=interaction_j, batch_size=None)
    model.initialize(seed=seed)
    h = model.evolution(init_delta_t, init_step, show_bar=True)
    all_hamiltonian[0] = h[-1]
    all_n[0] = model.get_n()

    for i in tqdm.trange(num_sample):
        delta_t = delta_ts[i]
        model = NLSM_model(l=l, interaction_j=interaction_j, batch_size=None)
        model.initialize(seed=seed)
        h = model.evolution(delta_t, steps[i], show_bar=False)
        all_hamiltonian[i+1] = h[-1]
        all_n[i+1] = model.get_n()

    final_diff = np.zeros(delta_ts.shape[0])
    final_diff_hamiltonian = np.zeros(delta_ts.shape[0])
    for i in range(num_sample):
        n_diff = all_n[i + 1] - all_n[0]
        final_diff[i] = np.linalg.norm(n_diff.flatten())
        final_diff_hamiltonian[i] = np.abs(all_hamiltonian[i + 1] - all_hamiltonian[0])


    lg_final_diff = np.log10(final_diff)
    lg_final_diff_hamiltonian = np.log10(final_diff_hamiltonian)
    lg_delta_ts = np.log10(delta_ts)

    fitter = WeightedLinearRegressor()
    fitter.fit(lg_delta_ts, None, lg_final_diff, None)
    state_slope = fitter.get_slope()
    state_intercept = fitter.get_intercept()
    print("State Error Slope:", state_slope)
    final_diff_est = 10 ** (lg_delta_ts * state_slope + state_intercept)

    fitter = WeightedLinearRegressor()
    fitter.fit(lg_delta_ts, None, lg_final_diff_hamiltonian, None)
    hamiltonian_slope = fitter.get_slope()
    hamiltonian_intercept = fitter.get_intercept()
    print("Hamiltonian Error Slope:", hamiltonian_slope)
    final_diff_hamiltonian_est = \
        10 ** (lg_delta_ts * hamiltonian_slope + hamiltonian_intercept)

    plt.plot(delta_ts, final_diff, label="Simulation")
    plt.plot(delta_ts, final_diff_est, label="Fitting")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Step size $\Delta t$")
    plt.ylabel("State error norm")
    plt.tick_params(direction="in")
    plt.title("Convergence Plot of State")
    plt.legend()
    plt.show()

    plt.plot(delta_ts, final_diff_hamiltonian, label="Simulation")
    plt.plot(delta_ts, final_diff_hamiltonian_est, label="Fitting")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Step size $\Delta t$")
    plt.ylabel("Hamiltonian error")
    plt.tick_params(direction="in")
    plt.title("Convergence Plot of Hamiltonian")
    plt.legend()
    plt.show()



def micro_canonical_calibration():
    """ Time step """
    delta_t = 1e-2
    """ Number of steps """
    grad_steps = 30000
    truncate_steps = 30000
    micro_ens_steps = 200000 + truncate_steps
    """ Size of the grid """
    l = 20
    """ Interaction strength principle components """
    interaction_j = np.ones(3)
    """ Number of systems """
    num_system = 10
    """ Seed """
    seed = global_rng.integers(LARGE_INT)
    """ Temperature and thermodynamic beta """
    temperatures = np.logspace(-1, 0, 5)
    betas = 1 / temperatures
    """ sigma = 0.05 """
    sigma = 5 / np.sqrt(micro_ens_steps)
    print("Sigma", sigma)


    """ Type """
    init_type = "truncate"

    """ All average energy """
    avg_energy = np.zeros(temperatures.shape[0])

    for i in range(len(temperatures)):
        temperature = temperatures[i]
        beta = betas[i]
        print("Temperature:", temperature)
        model = NLSM_model(l=l, interaction_j=interaction_j, batch_size=num_system)
        model.initialize(seed=seed)
        if init_type == "ground":
            hamiltonian = model.gradient_descent(delta_t, grad_steps, show_bar=True)
            # plot_hamiltonian(hamiltonian+400, scale="log")
        hamiltonian = model.micro_ens_sim(beta, micro_ens_steps, sigma, show_bar=True)
        plot_hamiltonian(hamiltonian)
        if init_type == "truncate":
            hamiltonian = hamiltonian[:, truncate_steps:]
        plot_hist_hamiltonian(hamiltonian,
            title=f"Microcanoncial Ensemble Simulation (Temperature {temperature})")
        avg_energy[i] = hamiltonian.mean()
    plt.plot(temperatures, avg_energy)
    plt.show()

def test_neighbor_generation():
    n = np.array([0,1])
    seed = 8888
    rng = np.random.default_rng(seed)
    sigma = 1
    size = 100
    samples = np.zeros((size, 2))
    for i in range(100):
        s = rng.integers(seed)
        samples[i] = neighbor_generation(n, sigma, s)
    plt.scatter(samples[:,0], samples[:,1])
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.show()

def test_gd_single():
    """ Time step """
    delta_t = 1e-2
    """ Number of steps """
    steps = 100000
    """ Size of the grid """
    l = 20
    """ Interaction strength principle components """
    interaction_j = np.ones(3)
    """ Seed """
    seed = global_rng.integers(LARGE_INT)

    model = NLSM_model(l=l, interaction_j=interaction_j)
    model.initialize(seed=seed)
    hamiltonian = model.gradient_descent(delta_t, steps, show_bar=True)
    plt.plot(hamiltonian + 400)
    plt.yscale("log")
    plt.show()
    print("Ground state energy:", hamiltonian[-1])

def test_gd():
    """ Time step """
    delta_t = 1e-2
    """ Number of steps """
    steps = 30000
    """ Size of the grid """
    l = 20
    """ Interaction strength principle components """
    interaction_j = np.ones(3)
    """ Number of systems """
    num_system = 10
    """ Seed """
    seed = global_rng.integers(LARGE_INT)

    model = NLSM_model(l=l, interaction_j=interaction_j, batch_size=num_system)
    model.initialize(seed=seed)
    hamiltonian = model.gradient_descent(delta_t, steps, show_bar=True)
    # plot_hamiltonian(hamiltonian + 400, title="Gradient Descent", scale="log")
    plot_hamiltonian(hamiltonian, title="Gradient Descent")

def test_evolution():
    """ Total evolution time: 0.01 """
    delta_t = 0.1

    """ Number of steps """
    steps = 10000

    """ Size of the grid """
    l = 20
    """ Interaction strength principle components """
    interaction_j = np.ones(3)

    model = NLSM_model(20, interaction_j, batch_size=10)
    model.initialize()
    hamiltonian = model.evolution(delta_t, steps, show_bar=True)
    plot_hamiltonian(hamiltonian)

if __name__ == "__main__":
    # test_evolution()
    # main_convergence()
    micro_canonical_calibration()
    # test_neighbor_generation()
    # test_gd_single()
    # test_gd()
