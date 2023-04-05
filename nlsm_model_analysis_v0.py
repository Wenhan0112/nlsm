import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import nlsm_model
import spin_field_model
import nlsm_utils
import two_level_model
import tqdm
import os
import shutil

cpu = torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else cpu
if __name__ == "__main__":
    print(f"Currently using device: {device}")

def load_and_simulate(target):
    data = nlsm_utils.load_data("part_data")
    temp_idx = 1
    sample_idx = 0
    temperature = data["temperatures"][temp_idx]
    n = data["n_samples"][temp_idx, sample_idx]
    print("Temperature:", temperature)
    model = nlsm_model.NLSM_model(
        l = data["l"],
        interaction_j = data["interaction_j"],
        device = device
    )
    model.initialize(n)

    """ Evolution time step """
    delta_t = 1e-3

    """ Number of steps """
    num_steps = 10000


    n_samples = torch.zeros((num_steps+1,) + n.shape, device=device)
    n_samples[0] = torch.tensor(n, device=device)
    for i in tqdm.trange(num_steps):
        model.evolution(delta_t, 1)
        n_samples[i+1] = model.get_n()


    if target == "state":
        animation = nlsm_utils.create_animation_state(
            n_samples, step=100, show_bar=True)
        # animation.save("evolution.gif")
    elif target == "skyrmion":
        """ Block size: 10 x 10 """
        block_size = 10

        """ Step size: 100 """
        steps = 100

        result = nlsm_model.skyrmion_density(n_samples[::steps], block_size)
        animation = nlsm_utils.create_animation_count(result, show_bar=True)
        # animation.save("skyrmion.gif")
    return animation

def test_micro_canonical_calibration(data_name):
    """ Time step """
    delta_t = 1e-2
    """ Number of steps """
    grad_steps = 50
    truncate_steps = 80000
    micro_ens_steps = 1000000 + truncate_steps
    """ Size of the grid """
    l = 1
    """ Interaction strength principle components """
    interaction_j = np.ones(3)
    """ Number of systems """
    num_system = 10
    """ Temperature and thermodynamic beta """
    temperatures = np.logspace(-1, 1, 20)
    betas = 1 / temperatures
    """ sigma = 0.05 """
    sigmas = np.full_like(temperatures, np.sqrt(3 / micro_ens_steps))
    """ Number of changed sites in the neighbor generation algorithm """
    # nums_update = (1000 * temperatures).astype(int)
    nums_update = np.ones_like(temperatures, dtype=int) + 1
    """ True iff the figures are saved """
    save_fig = True
    """ Test neighbor generation hamiltonian distribution """
    num_gen = 1000
    """ Number of bzones used in electrostatic potential integration """
    num_bzone = 1
    """ Electric constant """
    ec = 1.

    """ All average energy """
    avg_energy = np.zeros_like(temperatures)
    n_samples = []

    """ Return dict keys """
    omit_keys = set(locals().keys())

    img_folder_name = f"{data_name}_imgs"
    if os.path.exists(img_folder_name):
        shutil.rmtree(img_folder_name)
    os.makedirs(img_folder_name)

    """
    Test NLSM Model
    """
    for i in range(len(temperatures)):
        # i = 0
        temperature = temperatures[i]
        beta = betas[i]
        sigma = sigmas[i]
        num_update = nums_update[i]
        print(f"Step {i} out of {len(temperatures)}")
        print("Temperature:", temperature)
        print("Sigma", sigma)
        model = nlsm_model.NLSM_model(
            l=l,
            interaction_j=interaction_j,
            batch_size=num_system,
            electric_constant=ec,
            num_bzone=num_bzone,
            device=device,
            boundary="circular"
        )
        model.initialize()
        if grad_steps:
            hamiltonian = model.gradient_descent(
                delta_t, grad_steps,
                show_bar=True
            )
            fig_name = os.path.join(f"{data_name}_imgs",
                f"gd_{i:02d}.png") if save_fig else ""
            nlsm_utils.plot_hamiltonian(
                hamiltonian,
                title=f"Gradient Descent (Temperature {temperature:.2f})",
                save_fig=fig_name
            )
        if num_gen:
            original_h, new_h = model.test_generation_h_distribution(
                num_gen, sigma, num_update, show_bar=True)
            if not model.batched:
                original_h, new_h = [original_h], [new_h]
            for j in range(model.batch_size):
                fig_name = os.path.join(f"{data_name}_imgs",
                    f"gen_h_dist_{i:02d}_model_{j:02d}.png") if save_fig else ""
                nlsm_utils.plot_hist_hamiltonian(
                    new_h[j],
                    title=f"Neighbor Generation Distribution Model {j:02d} "
                    f"(Temperature {temperature:.2f})",
                    save_fig=fig_name,
                    vline=original_h[j].cpu().item()
                )
        hamiltonian = model.micro_ens_sim(
            beta,
            micro_ens_steps,
            sigma,
            num_update,
            show_bar=True
        )
        fig_name = os.path.join(f"{data_name}_imgs",
            f"micro_ens_sim_{i:02d}.png") if save_fig else ""
        nlsm_utils.plot_hamiltonian(
            hamiltonian,
            title="Microcanonical Ensemble Simulation "
            f"(Temperature {temperature:.2f})",
            save_fig=fig_name
        )
        if truncate_steps:
            hamiltonian = hamiltonian[:, truncate_steps:]
        fig_name = os.path.join(f"{data_name}_imgs",
            f"micro_ens_sim_hist_{i:02d}.png") if save_fig else ""
        nlsm_utils.plot_hist_hamiltonian(
            hamiltonian,
            title="Microcanoncial Ensemble Simulation "
            f"(Temperature {temperature:.2f})",
            save_fig=fig_name
        )
        avg_energy[i] = hamiltonian.mean().cpu().numpy()
        n = nlsm_utils.convert_array(model.get_n())
        n_samples.append(n)
        # nlsm_utils.visualize_3d_color_state(n[0, 0])
        print()
    omit_keys = set(locals().keys()) - omit_keys
    nlsm_utils.energy_temperature_dependence(temperatures, avg_energy, log=True)
    n_samples = np.array(n_samples)
    return_data = nlsm_utils.export_data(locals(), omit_keys=omit_keys)
    return return_data

def micro_canonical_calibration(data_name):
    """ Time step """
    delta_t = 1e-2
    """ Number of steps """
    grad_steps = 0
    truncate_steps = 80000
    micro_ens_steps = 10000000 + truncate_steps
    """ Size of the grid """
    l = 128
    """ Interaction strength principle components """
    inter_layer_interaction = np.ones(3)
    """ Number of systems """
    num_system = 16
    """ Temperature and thermodynamic beta """
    # temperatures = np.logspace(-1, 1, 5)
    temperatures = np.array([0.1, 10.])
    betas = 1 / temperatures
    """ sigma = 0.05 """
    gen_sigmas = np.full_like(temperatures, 0.01)
    """ Number of changed sites in the neighbor generation algorithm """
    # nums_update = (1000 * temperatures).astype(int)
    gen_probs = np.full_like(gen_sigmas, 1/256.)
    """ True iff the figures are saved """
    save_fig = True
    """ Test neighbor generation hamiltonian distribution """
    num_gen = 0
    """ Number of bzones used in electrostatic potential integration """
    num_bzone = 1
    """ Electric constant """
    ec = 1.
    """ Metal distance """
    metal_distance = 1.
    """ Moire length """
    moire_length = 10.
    """ Test """
    if_test = True

    """ All average energy """
    avg_energy = np.zeros_like(temperatures)
    n_samples = []

    if if_test:
        grad_steps = grad_steps // 100
        truncate_steps = truncate_steps // 100
        micro_ens_steps = micro_ens_steps // 100
        l = l // 16
        num_system = num_system // 4

    """ Return dict keys """
    omit_keys = set(locals().keys())

    img_folder_name = f"{data_name}_imgs"
    if os.path.exists(img_folder_name):
        shutil.rmtree(img_folder_name)
    os.makedirs(img_folder_name)

    """
    NLSM Model
    """
    for i in range(len(temperatures)):
        # i = 0
        temperature = temperatures[i]
        beta = betas[i]
        gen_sigma = gen_sigmas[i]
        gen_prob = gen_probs[i]
        print(f"Step {i} out of {len(temperatures)}")
        print("Temperature:", temperature)
        print("Neighbor generation sigma", gen_sigma)
        model = nlsm_model.NLSM_model(
            grid_size,
            num_sample,
            inter_layer_interaction=inter_layer_interaction,
            intra_layer_interaction=intra_layer_interaction,
            metal_distance=3.,
            num_bzone=2,
            device=device
        )
        model.initialize()
        if grad_steps:
            hamiltonian = model.gradient_descent(
                delta_t, grad_steps,
                show_bar=True
            )
            fig_name = os.path.join(f"{data_name}_imgs",
                f"gd_{i:02d}.png") if save_fig else ""
            nlsm_utils.plot_hamiltonian(
                hamiltonian,
                title=f"Gradient Descent (Temperature {temperature:.2f})",
                save_fig=fig_name
            )
        if num_gen:
            original_h, new_h = model.test_generation_h_distribution(
                num_gen, show_bar=True)
            if not model.batched:
                original_h, new_h = [original_h], [new_h]
            for j in range(model.batch_size):
                fig_name = os.path.join(f"{data_name}_imgs",
                    f"gen_h_dist_{i:02d}_model_{j:02d}.png") if save_fig else ""
                nlsm_utils.plot_hist_hamiltonian(
                    new_h[j],
                    title=f"Neighbor Generation Distribution Model {j:02d} "
                    f"(Temperature {temperature:.2f})",
                    save_fig=fig_name,
                    vline=original_h[j].cpu().item()
                )
        hamiltonian = model.micro_ens_sim(
            beta,
            micro_ens_steps,
            show_bar=True
        )
        fig_name = os.path.join(f"{data_name}_imgs",
            f"micro_ens_sim_{i:02d}.png") if save_fig else ""
        nlsm_utils.plot_hamiltonian(
            hamiltonian,
            title="Microcanonical Ensemble Simulation "
            f"(Temperature {temperature:.2f})",
            save_fig=fig_name
        )
        if truncate_steps:
            hamiltonian = hamiltonian[:, truncate_steps:]
        fig_name = os.path.join(f"{data_name}_imgs",
            f"micro_ens_sim_hist_{i:02d}.png") if save_fig else ""
        nlsm_utils.plot_hist_hamiltonian(
            hamiltonian,
            title="Microcanoncial Ensemble Simulation "
            f"(Temperature {temperature:.2f})",
            save_fig=fig_name
        )
        avg_energy[i] = hamiltonian.mean().cpu().numpy()
        n = nlsm_utils.convert_array(model.get_n())
        n_samples.append(n)
        # nlsm_utils.visualize_3d_color_state(n[0, 0])
        print()
    omit_keys = set(locals().keys()) - omit_keys
    nlsm_utils.energy_temperature_dependence(temperatures, avg_energy, log=True)
    n_samples = np.array(n_samples)
    return_data = nlsm_utils.export_data(locals(), omit_keys=omit_keys)
    return return_data

def spin_field_micro_canoncial_calibration(data_name):
    """ Number of steps """
    truncate_steps = 0
    micro_ens_steps = 10000000 + truncate_steps
    """ Interaction strength principle components """
    magnetic_field = np.array([0,0,1])
    """ Number of systems """
    num_system = 2
    """ Temperature and thermodynamic beta """
    temperatures = np.logspace(-1, 1, 10)
    # temperatures = np.array([1.])
    betas = 1 / temperatures
    """ sigma = 0.05 """
    gen_sigmas = np.full_like(temperatures, 0.01)
    """ Probability of a site is changed in the neighbor generation algorithm """
    # nums_update = (1000 * temperatures).astype(int)
    gen_probs = np.ones_like(temperatures)
    # gen_probs = np.full_like(temperatures, 0.5)
    """ True iff the figures are saved """
    save_fig = True
    """ Test neighbor generation hamiltonian distribution """
    num_gen = 0

    """ All average energy """
    avg_energy = np.zeros_like(temperatures)
    n_samples = []

    """ Return dict keys """
    omit_keys = set(locals().keys())

    img_folder_name = f"{data_name}_imgs"
    if os.path.exists(img_folder_name):
        shutil.rmtree(img_folder_name)
    os.makedirs(img_folder_name)

    """
    Spin Field Model
    """
    for i in range(len(temperatures)):
        # i = 0
        temperature = temperatures[i]
        beta = betas[i]
        gen_sigma = gen_sigmas[i]
        gen_prob = gen_probs[i]
        print(f"Step {i} out of {len(temperatures)}")
        print("Temperature:", temperature)
        print("Gen Sigma", gen_sigma)
        model = spin_field_model.Spin_Field_Model(
            gen_prob=gen_prob,
            gen_sigma=gen_sigma,
            magnetic_field=magnetic_field,
            batch_size=num_system,
            device=device
        )
        model.initialize()
        if num_gen:
            original_h, new_h = model.test_generation_h_distribution(
                num_gen, sigma, num_update, show_bar=True)
            if not model.batched:
                original_h, new_h = [original_h], [new_h]
            for j in range(model.batch_size):
                fig_name = os.path.join(f"{data_name}_imgs",
                    f"gen_h_dist_{i:02d}_model_{j:02d}.png") if save_fig else ""
                nlsm_utils.plot_hist_hamiltonian(
                    new_h[j],
                    title=f"Neighbor Generation Distribution Model {j:02d} "
                    f"(Temperature {temperature:.2f})",
                    save_fig=fig_name,
                    vline=original_h[j].cpu().item()
                )
        hamiltonian = model.micro_ens_sim(
            beta,
            micro_ens_steps,
            show_bar=True
        )
        fig_name = os.path.join(f"{data_name}_imgs",
            f"micro_ens_sim_{i:02d}.png") if save_fig else ""
        nlsm_utils.plot_hamiltonian(
            hamiltonian,
            title="Microcanonical Ensemble Simulation "
            f"(Temperature {temperature:.2f})",
            save_fig=fig_name
        )
        if truncate_steps:
            hamiltonian = hamiltonian[:, truncate_steps:]
        fig_name = os.path.join(f"{data_name}_imgs",
            f"micro_ens_sim_hist_{i:02d}.png") if save_fig else ""
        nlsm_utils.plot_hist_hamiltonian(
            hamiltonian,
            title="Microcanoncial Ensemble Simulation "
            f"(Temperature {temperature:.2f})",
            save_fig=fig_name
        )
        avg_energy[i] = hamiltonian.mean().cpu().numpy()
        n = nlsm_utils.convert_array(model.get_n())
        n_samples.append(n)
        # nlsm_utils.visualize_3d_color_state(n[0, 0])
        print()
    omit_keys = set(locals().keys()) - omit_keys
    nlsm_utils.energy_temperature_dependence(temperatures, avg_energy, log=True)
    n_samples = np.array(n_samples)
    return_data = nlsm_utils.export_data(locals(), omit_keys=omit_keys)
    return return_data

def two_level_simple_micro_canonical_calibration(data_name):
    """ Number of steps """
    grad_steps = 0
    truncate_steps = 80000
    micro_ens_steps = 1000000 + truncate_steps
    """ Number of systems """
    num_system = 10
    """ Temperature and thermodynamic beta """
    temperatures = np.logspace(-1, 1, 20)
    betas = 1 / temperatures
    """ True iff the figures are saved """
    save_fig = True

    """ All average energy """
    avg_energy = np.zeros_like(temperatures)
    n_samples = []

    """ Return dict keys """
    omit_keys = set(locals().keys())

    img_folder_name = f"{data_name}_imgs"
    if os.path.exists(img_folder_name):
        shutil.rmtree(img_folder_name)
    os.makedirs(img_folder_name)

    """
    Two level model
    """
    for i in range(len(temperatures)):
        # i = 0
        temperature = temperatures[i]
        beta = betas[i]
        print(f"Step {i} out of {len(temperatures)}")
        print("Temperature:", temperature)
        model = two_level_model.Two_Level_Simple_Model(
            batch_size=num_system,
            device=device
        )
        model.initialize()
        hamiltonian = model.micro_ens_sim(
            beta,
            micro_ens_steps,
            show_bar=True
        )
        fig_name = os.path.join(f"{data_name}_imgs",
            f"micro_ens_sim_{i:02d}.png") if save_fig else ""
        nlsm_utils.plot_hamiltonian(
            hamiltonian,
            title="Microcanonical Ensemble Simulation "
            f"(Temperature {temperature:.2f})",
            save_fig=fig_name
        )
        if truncate_steps:
            hamiltonian = hamiltonian[:, truncate_steps:]
        fig_name = os.path.join(f"{data_name}_imgs",
            f"micro_ens_sim_hist_{i:02d}.png") if save_fig else ""
        nlsm_utils.plot_hist_hamiltonian(
            hamiltonian,
            title="Microcanoncial Ensemble Simulation "
            f"(Temperature {temperature:.2f})",
            save_fig=fig_name
        )
        avg_energy[i] = hamiltonian.mean().cpu().numpy()
        n = nlsm_utils.convert_array(model.get_n())
        n_samples.append(n)
        # nlsm_utils.visualize_3d_color_state(n[0, 0])
        print()
    omit_keys = set(locals().keys()) - omit_keys
    nlsm_utils.energy_temperature_dependence(temperatures, avg_energy, log=True)
    n_samples = np.array(n_samples)
    return_data = nlsm_utils.export_data(locals(), omit_keys=omit_keys)
    return return_data

def evolution(data_name):
    """ Number of steps """
    steps = np.logspace(1, 3, 5, dtype=int)
    print(steps)
    """ Time step """
    delta_ts = 1 / steps.astype(float)
    """ Size of the grid """
    grid_size = 8
    """ Total number of sample points """
    num_sample = 128
    """ Interaction strength principle components """
    inter_layer_interaction = torch.ones(3)
    """ Intra layer interaction """
    intra_layer_interaction = 1
    """ Number of systems """
    num_system = 4
    """ True iff the figures are saved """
    save_fig = True
    """ Number of bzones used in electrostatic potential integration """
    num_bzone = 2
    """ Metal distance """
    metal_distance = 3.
    """ Test """
    if_test = False

    if if_test:
        steps //= 100
        grid_size /= 2
        num_sample //= 4
        num_system //= 4

    

    img_folder_name = f"{data_name}_imgs"
    if os.path.exists(img_folder_name):
        shutil.rmtree(img_folder_name)
    os.makedirs(img_folder_name)

    hamiltonians = []

    """ Return dict keys """
    omit_keys = set(locals().keys())

    for i in range(len(delta_ts)):
        delta_t = delta_ts[i]
        step = steps[i]
        model = nlsm_model.NLSM_model(
            grid_size=grid_size,
            num_sample=num_sample,
            inter_layer_interaction=inter_layer_interaction,
            intra_layer_interaction=intra_layer_interaction,
            batch_size=num_system,
            metal_distance=3.,
            num_bzone=2,
            device=device
        )
        model.initialize()
        hamiltonian = model.evolution(
            delta_t=delta_t,
            steps = step,
            show_bar=True
        )
        hamiltonians.append(hamiltonian)
        # fig_name = os.path.join(f"{data_name}_imgs",
        #     f"time_evolution.png") if save_fig else ""
        # nlsm_utils.plot_hamiltonian(
        #     hamiltonian,
        #     title="Time evolution",
        #     save_fig=fig_name
        # )

    omit_keys = set(locals().keys()) - omit_keys
    return_data = nlsm_utils.export_data(locals(), omit_keys=omit_keys)
    return return_data

if __name__ == "__main__":
    # animation = load_and_simulate("skyrmion")
    # data_name = "data_0612"
    # data = micro_canonical_calibration(data_name)
    data_name = "evolution"
    # data = micro_canonical_calibration(data_name)
    data = evolution(data_name)
    # nlsm_utils.dump_data(data, os.path.join("Data", data_name))
