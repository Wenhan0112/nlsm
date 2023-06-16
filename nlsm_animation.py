"""
The animation classes for the NLSM model. 
    Author: Wenhan Sun
    Date: 2023 Spring
"""

import tqdm
import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
import os
import torch
import shutil

class Animation():
    def __init__(self, *, save_fig: str = ""):
        self.states = []
        self.iterator = None
        self.save_fig = save_fig
        self.fig = None
    
    def add_state(self, state: list):
        raise NotImplementedError()

    def animate(self, i: int):
        raise NotImplementedError()
    
    def export_animation(self, 
            gif_file: str = "", verbose: int = 0,
            anim_interval = 200, repeat: bool = True):
        if verbose > 0:
            self.iterator = tqdm.trange(len(self.states))

        if self.save_fig:
            if os.path.exists(self.save_fig):
                shutil.rmtree(self.save_fig)
            os.makedirs(self.save_fig)

    
        animation = matplotlib.animation.FuncAnimation(
            self.fig, self.animate, interval=anim_interval, 
            repeat=repeat, frames=len(self.states)
        )
        if gif_file: 
            animation.save(gif_file)
        return animation

class Multilayer_Animation(Animation):
    def __init__(self, num_layers: int = 1, 
            invert_layer: bool|list[bool] = False, *, 
            save_fig: str = ""):
        super(Multilayer_Animation, self).__init__(save_fig=save_fig)
        self.fig, self.axes = plt.subplots(ncols=num_layers)
        if num_layers == 1:
            self.axes = [self.axes]
        if isinstance(invert_layer, bool):
            self.invert_layer = [invert_layer] * num_layers
    
    def animate(self, i: int):
        if self.iterator:
            self.iterator.update(1)
        ims = []
        for j in range(len(self.axes)):
            ax = self.axes[j]
            ax.clear()
            ax.set_title(f"Step {i} Layer {j}")
            ims.append(self.plot_layer(ax, self.states[i][j], 
                invert=self.invert_layer[j]))
        if self.save_fig:
            self.fig.savefig(
                os.path.join(self.save_fig, f"step_{i}.png")
            )
        return tuple(ims)

class Spin_Animation(Multilayer_Animation):
    def add_state(self, state: torch.Tensor):
        state = state.clone().detach().cpu().numpy()
        state = (state + 1.) / 2.
        state = state.clip(0, 1)
        self.states.append(state)
    
    def plot_layer(self, ax: matplotlib.axes.Axes, data, *, 
            invert: bool = False):
        if invert:
            data = 1. - data 
        im = ax.imshow(data)
        return im
    
    
    
class Field_Animation(Multilayer_Animation):
    def add_state(self, state: torch.Tensor):
        state = state.clone().detach().cpu()
        self.states.append(state)
    
    def plot_layer(self, ax: matplotlib.axes.Axes, data, *, 
            invert: bool = False):
        if invert:
            data = -data
        im = ax.imshow(data)
        return im