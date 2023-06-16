import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import pickle
from typing import Optional
import tqdm
from utils import WeightedLinearRegressor
import os
import shutil

"""
Visualization Utils
"""

def plot_hamiltonian(data: np.ndarray, title: str = "",
        scale: Optional[str] = None, save_fig: str = ""):
    """
    Plot the time evolution of the Hamiltonian.
    @params data (numpy.ndarray): The time evolving Hamiltonian.
        Constraints: data.ndim in [1, 2]
    """
    data = convert_array(data)
    fig, ax = plt.subplots()
    if data.ndim == 1:
        ax.plot(data)
    elif data.ndim == 2:
        for i in range(data.shape[0]):
            ax.plot(data[i])
    else:
        raise ValueError
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Hamiltonian")
    if title:
        ax.set_title(title)
    if scale:
        ax.set_yscale(scale)
        ax.set_ylabel("Hamiltonian Difference")
    ax.tick_params(direction="in")
    if save_fig:
        fig.savefig(save_fig)
    fig.show()
    plt.close(fig)

def plot_hist_hamiltonian(data, title: str = "", save_fig="", vline=None):
    """
    Plot the historgram of the Hamiltonian
    """
    data = convert_array(data)
    data = data.flatten()
    plt.hist(data, bins="sqrt", density=True)
    if vline:
        plt.axvline(vline, color="r")
    plt.xlabel("Hamiltonian")
    plt.ylabel("PDF")
    if title:
        plt.title(title)
    plt.tick_params(direction="in")
    if save_fig:
        plt.savefig(save_fig)
    plt.show()
    plt.close()

def visualize_state(n: np.ndarray):
    raise NotImplementedError("This method is not implemented!")
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

def visualize_3d_color_state(n, save_fig=""):
    n = convert_array(n)
    assert n.ndim == 3, "Incorrect dimension of visualization state!"
    img = (n + 1) / 2
    plt.imshow(img)
    if save_fig:
        plt.savefig(save_fig)
    plt.show()
    plt.close()

def energy_temperature_dependence(temperature, energy, fit=False, log=False):
    plt.plot(temperature, energy, marker="s", linestyle="None",
        label="Simulation")
    if fit:
        model = WeightedLinearRegressor()
        model.fit(temperature, None, energy, None)
        slope = model.get_slope()
        _, energy_est = model.get_y_est()
        correlation_coef = model.get_correlation_coef()
        print("Energy vs Temperature Fitting")
        print("Slope:", slope)
        print("Correlation coefficient:", correlation_coef)
        plt.plot(temperature, energy_est, label="Fitting")
        plt.legend()
    plt.title("Energy vs Temperature Relation")
    plt.xlabel("Temperature")
    plt.ylabel("Energy")
    plt.tick_params(direction="in")
    if log:
        plt.xscale("log")
    plt.show()
    plt.close()

def create_animation_state(data, step=1, show_bar=False, save_figs=None):
    assert data.ndim == 5 and data.shape[1] == 2 and data.shape[4] == 3
    data = data.detach().cpu().numpy()
    fig, (ax0, ax1) = plt.subplots(ncols=2)
    num_frame = int(data.shape[0] / step)

    data = (data + 1.) / 2.
    data = data.clip(0, 1)

    if show_bar:
        iterator = tqdm.trange(num_frame)

    if save_figs:
        if os.path.exists(save_figs):
            shutil.rmtree(save_figs)
        os.makedirs(save_figs)

    def animate(i):
        i = i * step
        if show_bar:
            iterator.update(1)
        ax0.clear()
        ax1.clear()
        ax0.set_title(f"Step {i} Layer 0")
        ax1.set_title(f"Step {i} Layer 1")
        im0 = ax0.imshow(data[i, 0])
        im1 = ax1.imshow(data[i, 1])
        if save_figs:
            fig.savefig(
                os.path.join(save_figs, "step_{i}.png")
            )
        return im0, im1

    animation = matplotlib.animation.FuncAnimation(
        fig, animate, interval=1, repeat=True, frames=num_frame
    )
    return animation

def create_animation_count(data, step=1, show_bar=False):
    data = data.detach().cpu().numpy()
    fig, (ax0, ax1) = plt.subplots(ncols=2)
    im0 = ax0.imshow(data[0, 0], vmin=-1, vmax=1)
    fig.colorbar(im0, ax=(ax0, ax1))
    num_frame = int(data.shape[0] / step)

    if show_bar:
        iterator = tqdm.trange(num_frame)

    def animate(i):
        i = i * step
        if show_bar:
            iterator.update(1)
        ax0.clear()
        ax1.clear()
        ax0.set_title(f"Step {i} Layer 0")
        ax1.set_title(f"Step {i} Layer 1")
        im0 = ax0.imshow(data[i, 0], vmin=-1, vmax=1)
        im1 = ax1.imshow(data[i, 1], vmin=-1, vmax=1)
        return im0, im1

    animation = matplotlib.animation.FuncAnimation(
        fig, animate, interval=1, repeat=True, frames=num_frame
    )
    return animation

"""
Data Format Manipulation
"""

def convert_array(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, np.ndarray):
        return np.array(data)

def load_data(file_name):
    return pickle.load(open(file_name, "rb"))

def dump_data(data, file_name):
    pickle.dump(data, open(file_name, "wb"))

def export_data(data_dict, keys=None, omit_keys=[]):
    result = {}
    if not keys:
        keys = data_dict.keys()
    for k in keys:
        if k in omit_keys:
            continue
        if isinstance(data_dict[k], torch.Tensor):
            result[k] = data_dict[k].cpu()
        else:
            result[k] = data_dict[k]
    return result
