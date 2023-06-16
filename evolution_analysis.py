import numpy as np
import torch
import os
import shutil
import nlsm_utils
import nlsm_model
import evolver as evolver_pkg
import nlsm_animation
import tqdm
import matplotlib.pyplot as plt

cpu = torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else cpu
if __name__ == "__main__":
    print(f"Currently using device: {device}")

def test_alignment(data_name):
    """ Step size in time evolution """
    dt = 1e-2
    """ Number of steps in time evolution """
    time_evolution_step = 10000
    """ Size of the grid """
    grid_size = 16
    """ Number of samples in a grid in 1D. """
    num_sample = 128
    """ Intra-layer interaction strength """
    intra_layer_interaction = 1
    """ Interaction strength principle components """
    inter_layer_interaction = torch.ones(3)
    """ Number of systems """
    batch_size = 1
    """ True iff the figures are saved """
    save_fig = True
    """ Number of bzones used in electrostatic potential integration """
    num_bzone = 2
    """ Metal distance """
    metal_distance = 3.
    """ Test """
    if_test = False
    """ Save animation step size """
    anim_step_size = 50

    if if_test:
        time_evolution_step = time_evolution_step // 20
        anim_step_size = anim_step_size // 10
        num_sample = num_sample // 16
        batch_size = batch_size // 4

    """ Return dict keys """
    # omit_keys = set(locals().keys())

    img_folder_name = f"{data_name}_imgs"
    if os.path.exists(img_folder_name):
        shutil.rmtree(img_folder_name)
    os.makedirs(img_folder_name)

    """
    NLSM Model
    """
    model = nlsm_model.NLSM_model(
        grid_size,
        num_sample,
        inter_layer_interaction=inter_layer_interaction,
        intra_layer_interaction=intra_layer_interaction,
        metal_distance=metal_distance,
        num_bzone=num_bzone,
        device=device,
        batch_size=batch_size
    )

    n = torch.empty((1, 2, num_sample, num_sample, 3))
    skyrmion_radius = 0.8
    pos = torch.arange(0, grid_size, grid_size / num_sample)
    disp_to_center = pos - grid_size / 2
    dist_to_center = torch.sqrt(disp_to_center[:, None]**2 + disp_to_center[None, :]**2)
    theta = 2*torch.arcsin(torch.exp(-dist_to_center / (2*skyrmion_radius)))
    phi = torch.atan2(disp_to_center[None, :], disp_to_center[:, None])
    n[:, 0, :, :, 0] = torch.sin(theta) * torch.cos(phi)
    n[:, 0, :, :, 1] = torch.sin(theta) * torch.sin(phi)
    n[:, 0, :, :, 2] = torch.cos(theta)
    n[:, 1] = -n[:, 0]
    # n1 = n.roll(num_sample // 8, dims=-3)
    # n2 = n.roll(-num_sample // 8, dims=-3)
    # n = torch.nn.functional.normalize(n1 + n2, dim=-1)
    # n[:, 1] = -n[:, 0]

    evolver = evolver_pkg.Midpoint_Evolver(model)
    model.initialize(n)
    # print("Skyrmion count:", model.skyrmion_count())
    # plt.imshow(model.skyrmion_density()[0])
    # plt.show()
    hamiltonian = torch.empty((batch_size, time_evolution_step))
    evolver.initialize()
    skyrmion_animation_folder = os.path.join(
        f"{data_name}_imgs", "skyrmion_animation_plots"
    ) if save_fig else ""
    state_animation = nlsm_animation.Spin_Animation(
        2, invert_layer=False, save_fig=""
    )
    diff_animation = nlsm_animation.Spin_Animation(
        1, invert_layer=False, save_fig=""
    )
    skyrmion_animation = nlsm_animation.Field_Animation(
        1, invert_layer=False, save_fig=""
    )


    # curr_density = torch.empty(curr_corr_shape)
    for i_ev in tqdm.trange(time_evolution_step):
        # curr_density[:, i_ev] = model.current_density()
        hamiltonian[:, i_ev] = model.hamiltonian()
        if i_ev % anim_step_size == 0:
            state_animation.add_state(model.n[0])
            diff_animation.add_state((model.n[0, 0, None] + model.n[0, 1, None]) / 2)
            skyrmion_animation.add_state(model.skyrmion_density()[0, None])
        evolver.step(dt)
        
    # curr_correlator.update_single(curr_density)
        
    
    hamiltonian = hamiltonian.squeeze()

    fig_name = os.path.join(f"{data_name}_imgs",
        f"evolution.png") if save_fig else ""
    nlsm_utils.plot_hamiltonian(
        hamiltonian,
        title="Evolution",
        save_fig=fig_name
    )
    fig_name = os.path.join(f"{data_name}_imgs", 
        "state.gif")
    state_animation.export_animation(fig_name, anim_interval=200)
    fig_name = os.path.join(f"{data_name}_imgs", 
        "diff.gif")
    diff_animation.export_animation(fig_name, anim_interval=200)
    fig_name = os.path.join(f"{data_name}_imgs", 
        "skyrmion_density.gif")
    skyrmion_animation.export_animation(fig_name, anim_interval=200)
    # if truncate_steps:
    #     hamiltonian = hamiltonian[:, truncate_steps:]
    # fig_name = os.path.join(f"{data_name}_imgs",
    #     f"micro_ens_sim_hist_{i:02d}.png") if save_fig else ""
    # nlsm_utils.plot_hist_hamiltonian(
    #     hamiltonian,
    #     title="Microcanoncial Ensemble Simulation "
    #     f"(Temperature {temperature:.2f})",
    #     save_fig=fig_name
    # )
    # n = nlsm_utils.convert_array(model.get_n())
    # n_samples.append(n)
    # nlsm_utils.visualize_3d_color_state(n[0, 0])
    print()
    # omit_keys = set(locals().keys()) - omit_keys
    # nlsm_utils.energy_temperature_dependence(temperatures, avg_energy, log=True)
    return_data = nlsm_utils.export_data(locals())
    return return_data

def test_midpoint_verlet(data_name):
    """ Step size in time evolution """
    dt = 1e-2
    """ Number of steps in time evolution """
    time_evolution_step = 100
    """ Size of the grid """
    grid_size = 16
    """ Number of samples in a grid in 1D. """
    num_sample = 128
    """ Intra-layer interaction strength """
    intra_layer_interaction = 1
    """ Interaction strength principle components """
    inter_layer_interaction = torch.ones(3)
    """ Number of systems """
    batch_size = 4
    """ True iff the figures are saved """
    save_fig = True
    """ Number of bzones used in electrostatic potential integration """
    num_bzone = 2
    """ Metal distance """
    metal_distance = 3.
    """ Test """
    if_test = False
    """ Save animation step size """
    anim_step_size = 10

    if if_test:
        time_evolution_step = time_evolution_step // 20
        anim_step_size = anim_step_size // 10
        num_sample = num_sample // 16
        batch_size = batch_size // 4

    """ Return dict keys """
    omit_keys = set(locals().keys())

    img_folder_name = f"{data_name}_imgs"
    if os.path.exists(img_folder_name):
        shutil.rmtree(img_folder_name)
    os.makedirs(img_folder_name)

    """
    NLSM Model
    """
    

    n = torch.empty((1, 2, num_sample, num_sample, 3))
    skyrmion_radius = 0.8
    pos = torch.arange(0, grid_size, grid_size / num_sample)
    disp_to_center = pos - grid_size / 2
    dist_to_center = torch.sqrt(disp_to_center[:, None]**2 + disp_to_center[None, :]**2)
    theta = 2*torch.arcsin(torch.exp(-dist_to_center / (2*skyrmion_radius)))
    phi = torch.atan2(disp_to_center[None, :], disp_to_center[:, None])
    n[:, 0, :, :, 0] = torch.sin(theta) * torch.cos(phi)
    n[:, 0, :, :, 1] = torch.sin(theta) * torch.sin(phi)
    n[:, 0, :, :, 2] = torch.cos(theta)
    n[:, 1] = -n[:, 0]
    n1 = n.roll(num_sample // 8, dims=-3)
    n2 = n.roll(-num_sample // 8, dims=-3)
    n = torch.nn.functional.normalize(n1 + n2, dim=-1)
    n[:, 1] = -n[:, 0]

    model = nlsm_model.NLSM_model(
        grid_size,
        num_sample,
        inter_layer_interaction=inter_layer_interaction,
        intra_layer_interaction=intra_layer_interaction,
        metal_distance=metal_distance,
        num_bzone=num_bzone,
        device=device,
        batch_size=batch_size
    )
    model.initialize(n)

    hamiltonian = torch.empty((batch_size, time_evolution_step))
    evolver = evolver_pkg.Midpoint_Evolver(model)
    evolver.initialize()
    state_animation_folder = os.path.join(
        f"{data_name}_imgs", "animation_plots"
    ) if save_fig else ""
    state_animation = nlsm_animation.Spin_Animation(
        2, invert_layer=False, save_fig=""
    )
    # curr_density = torch.empty(curr_corr_shape)
    for i_ev in tqdm.trange(time_evolution_step):
        # curr_density[:, i_ev] = model.current_density()
        hamiltonian[:, i_ev] = model.hamiltonian()
        if i_ev % anim_step_size == 0:
            state_animation.add_state(model.n[0])
        evolver.step(dt)
        
    # curr_correlator.update_single(curr_density)
        
    
    hamiltonian = hamiltonian.squeeze()
    print(hamiltonian.shape)

    fig_name = os.path.join(f"{data_name}_imgs",
        f"evolution.png") if save_fig else ""
    nlsm_utils.plot_hamiltonian(
        hamiltonian,
        title="Evolution",
        save_fig=fig_name
    )
    fig_name = os.path.join(f"{data_name}_imgs", 
        "state.gif")
    state_animation.export_animation(fig_name, anim_interval=1000)

    model = nlsm_model.NLSM_model(
        grid_size,
        num_sample,
        inter_layer_interaction=inter_layer_interaction,
        intra_layer_interaction=intra_layer_interaction,
        metal_distance=metal_distance,
        num_bzone=num_bzone,
        device=device,
        batch_size=batch_size
    )
    model.initialize(n)

    hamiltonian = torch.empty((batch_size, time_evolution_step))
    evolver = evolver_pkg.Midpoint_Evolver(model)
    evolver.initialize()
    state_animation_folder = os.path.join(
        f"{data_name}_imgs", "animation_plots"
    ) if save_fig else ""
    state_animation = nlsm_animation.Spin_Animation(
        2, invert_layer=False, save_fig=""
    )
    # curr_density = torch.empty(curr_corr_shape)
    for i_ev in tqdm.trange(time_evolution_step):
        # curr_density[:, i_ev] = model.current_density()
        hamiltonian[:, i_ev] = model.hamiltonian()
        if i_ev % anim_step_size == 0:
            state_animation.add_state(model.n[0])
        evolver.step(dt)
    # if truncate_steps:
    #     hamiltonian = hamiltonian[:, truncate_steps:]
    # fig_name = os.path.join(f"{data_name}_imgs",
    #     f"micro_ens_sim_hist_{i:02d}.png") if save_fig else ""
    # nlsm_utils.plot_hist_hamiltonian(
    #     hamiltonian,
    #     title="Microcanoncial Ensemble Simulation "
    #     f"(Temperature {temperature:.2f})",
    #     save_fig=fig_name
    # )
    # n = nlsm_utils.convert_array(model.get_n())
    # n_samples.append(n)
    # nlsm_utils.visualize_3d_color_state(n[0, 0])
    print()
    omit_keys = set(locals().keys()) - omit_keys
    # nlsm_utils.energy_temperature_dependence(temperatures, avg_energy, log=True)
    return_data = nlsm_utils.export_data(locals(), omit_keys=omit_keys)
    return return_data

if __name__ == "__main__":
    test_alignment("test_alignment")