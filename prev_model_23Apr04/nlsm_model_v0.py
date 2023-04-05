import numpy as np
import scipy.spatial.transform
import scipy.stats
import matplotlib.pyplot as plt
from typing import Optional
import tqdm

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
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)

    if size == 1:
        n = rng.normal(size=d)
        return n / np.linalg.norm(n)
    else:
        n = rng.normal(size=np.append(size, d))
        norm = np.linalg.norm(n, axis=-1)
        result = n / np.stack([norm] * d, axis=-1)
        return result

def nlsm_hamiltonian(n: np.ndarray, interaction_j: np.ndarray) -> float:
    """
    Calculate the NLSM Hamiltonian.
    @params n (numpy.ndarray): The input array.
        Constraints: n.shape == (2, n.shape[1], n.shape[1], 3)
    @params interaction_j (numpy.ndarray): Principle components of interaction
        strength.
        Constraints: interaction_j.shape == (3,)
    @return (float): The Hamiltonian of the system.
    """
    l = n.shape[1]
    x_grad = -n[:, :-1, :] + n[:, 1:, :]
    y_grad = -n[:, :, :-1] + n[:, :, 1:]
    hamiltonian = np.sum(x_grad ** 2) + np.sum(y_grad ** 2)
    hamiltonian += np.sum(interaction_j * n[0] * n[1])
    return hamiltonian

def nlsm_hamiltonian_batch(ns: np.ndarray, interaction_j: np.ndarray) \
        -> np.ndarray:
    """
    Batchly calculate the NLSM Hamiltonian.
    @params ns (numpy.ndarray): The input array.
        Constraints: ns.shape[-4:] == (2, n.shape[1], n.shape[1], 3)
    @params interaction_j (numpy.ndarray): Principle components of interaction
        strength.
        Constraints: interaction_j.shape == (3,)
    @return (numpy.ndarray): The Hamiltonian of the system.
        Constraints: RETURN.shape == ns.shape[:-4]
    """
    l = ns.shape[-2]
    ns_shape = ns.shape
    ns = ns.reshape((-1, 2, l, l, 3))
    x_grad = -ns[:, :, :-1, :] + ns[:, :, 1:, :]
    y_grad = -ns[:, :, :, :-1] + ns[:, :, :, 1:]
    hamiltonian = np.sum(x_grad ** 2, axis=(-4,-3,-2,-1)) + \
        np.sum(y_grad ** 2, axis=(-4,-3,-2,-1))
    hamiltonian *= l / (l - 2)
    hamiltonian += np.sum(interaction_j * ns[:, 0] * ns[:, 1], axis=(-3,-2,-1))
    return hamiltonian.reshape(ns_shape[:-4])

def force_calculation(n: np.ndarray, interaction_j: np.ndarray) -> np.ndarray:
    """
    Calculate the NLSM force matrix. Only consider nearest neighbor
        interactions.
    @params n (numpy.ndarray): The input array.
        Constraints: n.shape == (2, n.shape[1], n.shape[1], 3)
    @params interaction_j (numpy.ndarray): Principle components of interaction
        strength.
        Constraints: interaction_j.shape == (3,)
    @return (numpy.ndarray): The NLSM rotation matrix.
        Constraints: RETURN.shape == (2, n.shape[1], n.shape[1], 3, 3)
    """
    l = n.shape[1]
    f = np.zeros(n.shape)
    f[:, 1:, :] += n[:, :-1, :]
    f[:, :-1, :] += n[:, 1:, :]
    f[:, :, 1:] += n[:, :, :-1]
    f[:, :, :-1] += n[:, :, 1:]
    f *= 2
    f -= np.flip(n, axis=0) * interaction_j
    return f

def nlsm_evolution_matrix(n: np.ndarray, delta_t: float,
        interaction_j: np.ndarray) -> np.ndarray:
    """
    Calculate the NLSM rotation matrix. Only consider nearest neighbor
        interactions.
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
    f = force_calculation(n, interaction_j).reshape((-1, 3))
    rot = scipy.spatial.transform.Rotation.from_rotvec(f * delta_t)
    rot_mat = rot.as_matrix().reshape((2, l, l, 3, 3))
    return rot_mat

def nlsm_evolution(delta_t: float, steps: int, l: int,
        interaction_j: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Perform NLSM time evolution.
    @params delta_t (float): The time step of the evolution.
        Constraints: delta_t > 0
    @params steps (float): The number of time steps of the evolution.
        Constraints: step > 0
    @params interaction_j (numpy.ndarray): Principle components of interaction
        strength.
        Constraints: interaction_j.shape == (3,)
    @return (numpy.ndarray): The Hamiltonian as a function of time.
        Constraints: RETURN.shape == (steps + 1,)
    """
    n = uniform_sphere(3, size=(2, l, l), seed=seed)
    hamiltonian = np.zeros(steps + 1)
    hamiltonian[0] = nlsm_hamiltonian(n, interaction_j)
    for i in tqdm.trange(steps, desc="NLSM Evolution"):
        n1 = np.matmul(nlsm_evolution_matrix(n, delta_t/2, interaction_j),
            n.reshape((2,l,l,3,1)))
        n1 = n1.reshape((2,l,l,3))
        n2 = np.matmul(nlsm_evolution_matrix(n1, delta_t, interaction_j),
            n.reshape((2,l,l,3,1)))
        n = n2.reshape((2,l,l,3))
        hamiltonian[i+1] = nlsm_hamiltonian(n, interaction_j)
    return hamiltonian

def gradient_descent_hamiltonian_matrix(n: np.ndarray, delta_t: float,
        interaction_j: np.ndarray) -> np.ndarray:
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
    f = force_calculation(n, interaction_j).reshape((-1, 3))
    rot = scipy.spatial.transform.Rotation.from_rotvec(
        np.cross(n.reshape((-1, 3)), f) * delta_t)
    rot_mat = rot.as_matrix().reshape((2, l, l, 3, 3))
    return rot_mat

def gradient_descent_hamiltonian(delta_t: float, steps: int, l: int,
        interaction_j: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Perform gradient descent on the Hamiltonian.
    @params delta_t (float): The time step of the evolution.
        Constraints: delta_t > 0
    @params steps (float): The number of time steps of the evolution.
        Constraints: step > 0
    @params interaction_j (numpy.ndarray): Principle components of interaction
        strength.
        Constraints: interaction_j.shape == (3,)
    @return (numpy.ndarray): The Hamiltonian as a function of time.
        Constraints: RETURN.shape == (steps + 1,)
    """
    n = uniform_sphere(3, size=(2, l, l), seed=seed)
    hamiltonian = np.zeros(steps + 1)
    hamiltonian[0] = nlsm_hamiltonian(n, interaction_j)
    for i in tqdm.trange(steps, desc="Gradient descent"):
        n = np.matmul(
            gradient_descent_hamiltonian_matrix(n, delta_t, interaction_j),
            n.reshape((2,l,l,3,1))
        ).reshape((2,l,l,3))
        hamiltonian[i+1] = nlsm_hamiltonian(n, interaction_j)
    return hamiltonian, n

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

def plot_hamiltonian(data):
    """
    Plot the time evolution of the Hamiltonian.
    @params data (numpy.ndarray): The time evolving Hamiltonian.
        Constraints: data.ndim == 1
    """
    plt.plot(data)
    plt.xlabel("Iteration")
    plt.ylabel("Hamiltonian")
    plt.title("Time evolution of a system")
    plt.tick_params(direction="in")
    plt.show()

if __name__ == "__main__":
    """ Time step """
    delta_t = 0.01
    """ Number of steps """
    steps = 50000
    """ Size of the grid """
    l = 20
    """ Interaction strength principle components """
    interaction_j = np.ones(3)
    """ Number of systems """
    num_system = 100000

    """ Perform time series evolution"""
    # print("Evolution of a system. ")
    # hamiltonian = nlsm_evolution(delta_t, steps, l, interaction_j)
    # plot_hamiltonian(hamiltonian)
    # print()

    """ Perform density of states function simulation. """
    # print("Density of stats simulation. ")
    # ns = uniform_sphere(3, size=(num_system, 2, l, l))
    # hamiltonian = nlsm_hamiltonian_batch(ns, interaction_j)
    # fit_gaussian(hamiltonian)

    """ Perform gradient descent on Hamiltonian. """
    # print("Gradient descent on Hamiltonian. ")
    # hamiltonian, final_state = gradient_descent_hamiltonian(
    #     delta_t, steps, l, interaction_j
    # )
    # plot_hamiltonian(hamiltonian)
    # plt.plot(-np.diff(hamiltonian))
    # plt.yscale("log")
    # plt.show()
    # print(hamiltonian[-10:])
    # print()

    """ Test class """
    model = NLSM_model(l=20, interaction_j=interaction_j, batch_size=100)
    model.initialize()
    print(model.hamiltonian())
    # model = NLSM_model(20, interaction_j)
