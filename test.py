import numpy as np
import torch
import matplotlib.pyplot as plt
import nlsm_model
import time
import torchgeometry
import physical_fn as pypf
import utils
import two_level_model
import spin_field_model
import canonical_ens_sim

cpu = torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else cpu

def plot_unit_circle(res = 100):
    theta = np.arange(res+1) * 2 * np.pi / res
    plt.plot(np.cos(theta), np.sin(theta))

def plot_data(n):
    plt.scatter(n[:, 0], n[:, 1])
    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5, 1.5)

def test_neighbor_generation_1():
    n = torch.zeros((1, 10, 2))
    n[:, :, 0] = 1.
    plot_data(n[0])
    sigma = 1
    num = 5
    new_n = nlsm_model.neighbor_generation(n, sigma, num)
    plot_unit_circle()
    plot_data(new_n[0])
    plt.show()

def test_neighbor_generation_2():
    n = nlsm_model.uniform_sphere(3, (4,2,3,3))
    ones = (n**2).sum(axis=-1)
    eps = 1e-8
    assert torch.all(torch.abs(ones - 1.) < eps)
    sigma, num = 0.1, 4
    new_n = nlsm_model.neighbor_generation(n, sigma, num)
    print(new_n-n)
    count = torch.eq(new_n, n).sum()
    print(count)

def test_plot():
    a = np.arange(100)
    plt.plot(a, a, "r")
    plt.savefig("nlsm_long_imgs/test_1.png")
    plt.show()
    plt.close()
    plt.plot(a, 2*a, "b")
    plt.savefig("nlsm_long_imgs/test_2.png")
    plt.show()

def show_array(x, title=None):
    plt.matshow(torch.transpose(x, 0, 1), origin="lower")
    if title:
        plt.title(title)
    plt.show()

def test_hamiltonian():
    grid_sizes = [16, 32, 64]
    # num_samples = torch.logspace(7, 12, 12-7+1, base=2, dtype=int)
    # grid_sizes = torch.logspace(3,5,5-3+1,base=2, dtype=int)
    d = 3
    h=[]
    # num_bzones = torch.arange(2, 8, 2)
    # print(num_samples)
    for grid_size in grid_sizes:
        # print(num_sample.item())
        num_sample = 1024
        # grid_size = 16
        num_bzone=2
        intra_layer_interaction = 1
        inter_layer_interaction = torch.ones(3)

        n = torch.zeros((1, 2, num_sample, num_sample, 3))
        n[:, 1, :, :, 2] = -1
        skyrmion_radius = 0.8
        pos = torch.arange(0, grid_size, grid_size / num_sample)
        disp_to_center = pos - grid_size / 2
        dist_to_center = torch.sqrt(disp_to_center[:, None]**2 + disp_to_center[None, :]**2)
        theta = 2*torch.arcsin(torch.exp(-dist_to_center / (2*skyrmion_radius)))
        phi = torch.atan2(disp_to_center[None, :], disp_to_center[:, None])
        n[:, 0, :, :, 0] = torch.sin(theta) * torch.cos(phi)
        n[:, 0, :, :, 1] = torch.sin(theta) * torch.sin(phi)
        n[:, 0, :, :, 2] = torch.cos(theta)
        # show_array(theta)
        # show_array(n[0,0,:,:,0])
        
        # plt.quiver(n[0, 0, :, :, 1], n[0, 0, :, :, 0])
        # x, y = np.meshgrid(np.arange(10), np.arange(10))
        # plt.quiver(x / np.sqrt(x**2+y**2), y / np.sqrt(x**2+y**2))
        # plt.show()

        model = nlsm_model.NLSM_model(
            grid_size = grid_size,
            num_sample = num_sample,
            inter_layer_interaction=inter_layer_interaction,
            intra_layer_interaction=intra_layer_interaction,
            batch_size=1,
            metal_distance=d,
            num_bzone=num_bzone,
            gpu_memory=0
        )
        model.initialize(n)
        # density = model.skyrmion_density()
        # plt.imshow(density[0,0])
        # plt.show()
        # plt.imshow(density[0,1])
        # plt.show()
        print(model.n.size())
        h.append(model.hamiltonian(compute_grad=False))
    print(h)

def test_solid_angle():
    n = torch.zeros((2,2,3))
    n[0,0] = torch.tensor([0,0,1])
    n[0,1] = torch.tensor([0,1,0])
    n[1,1] = torch.tensor([0,0,1])
    n[1,0] = torch.tensor([1,0,0])
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
    print(density_2)
    density = (density_1.arctan() + density_2.arctan()) * 2
    print(density[0,0])

def test_speed():
    a = torch.randn((2**15, 3))
    b = torch.randn((2**15, 3))
    t1 = time.time()
    torch.vecdot(a, b, axis=-1)
    print(time.time() - t1)
    t1 = time.time()
    torch.sum(a * b, axis=-1)
    print(time.time() - t1)

class A():
    def __init__(self):
        self.v = torch.arange(5, dtype=float)

    def test(self):
        self.v.requires_grad_()
        print(self.v)
        b = self.v.pow(2)
        loss = b.sum()
        loss.backward()
        print(self.v.grad)
        with torch.no_grad():
            self.v += 1
            self.v.grad.zero_()
        print()
        print(self.v)
        print(self.v.grad)
        print()
        b = self.v.pow(2)
        loss = b.sum()
        loss.backward()
        print(self.v)
        print(self.v.grad)

def test_grad():
    a = A()
    a.v.requires_grad_()
    print(a.v)
    b = a.v.pow(2)
    loss = b.sum()
    loss.backward()
    print(a.v.grad)
    with torch.no_grad():
        a.v += 1
        a.v.grad.zero_()
    print()
    print(a.v)
    print(a.v.grad)
    print()
    b = a.v.pow(2)
    loss = b.sum()
    loss.backward()
    print(a.v)
    print(a.v.grad)

def test_no_grad():
    def fn(x):
        with torch.no_grad():
            x *= 2
            return x
    a = torch.arange(5, dtype=float)
    a.requires_grad_()
    loss = a.pow(2).sum()
    loss.backward()
    print(a, a.grad)
    with torch.no_grad():
        a += 1
        print(a, a.grad)
        a = fn(a)
        print(a, a.grad)

def test_torch_geometry():
    a = torch.tensor([[0,0,1],[0,1,0]]) * np.pi / 2
    a = a.unsqueeze(0)
    print(a)
    print(a.shape)
    rot = torchgeometry.angle_axis_to_rotation_matrix(a)
    print(rot)

class A2():
    def __init__():
        self.n = torch.arange(5, dtype=float)

def test_grad2():
    a = A2()
    a.n.requires_grad_(True)
    loss = a.n.pow(2).sum()
    loss.backward()
    grad = a.n.grad
    print(a.n, a.n.grad, loss)
    a = a.n.detach()
    a.requires_grad_(False)
    loss = loss.detach()
    loss.requires_grad_(False)
    a += 1
    print(a, a.grad, loss)

def test_evolution_convergence():
    import pickle
    d = pickle.load(open("pf","rb"))
    # print(d.keys())
    hamiltonians = d["hamiltonians"]
    final_h = []
    for hamiltonian in hamiltonians:
        final_h.append(hamiltonian[:, -1])
    final_h = torch.stack(final_h)
    print(final_h)
    for i in range(final_h.shape[1]):
        h = final_h[:, i]
        hdiff = h[:-1] - h[-1]
        plt.plot(d["delta_ts"][:-1], hdiff.abs())
        plt.xscale("log")
        plt.yscale("log")
        plt.show()

def test_tanhc_var1():
    x1 = torch.logspace(-8, 1, 1000)
    d = 3
    plt.plot(x1, pypf.tanhc_var1(x1, d), label="tanhc_var1")
    plt.plot(x1, d * pypf.tanhc(x1 * d), label="tanhc")
    plt.plot(x1, torch.tanh(x1 * d) / x1, label="tanh")
    plt.legend()
    plt.xscale("log")
    plt.show()

def test_grad():
    grid_sizes = [32]
    # num_samples = torch.logspace(7, 12, 12-7+1, base=2, dtype=int)
    # grid_sizes = torch.logspace(3,5,5-3+1,base=2, dtype=int)
    d = 3
    h=[]
    # num_bzones = torch.arange(2, 8, 2)
    # print(num_samples)
    for grid_size in grid_sizes:
        # print(num_sample.item())
        num_sample = 1024
        # grid_size = 16
        num_bzone=2
        intra_layer_interaction = 1
        inter_layer_interaction = torch.ones(3)

        n = torch.zeros((1, 2, num_sample, num_sample, 3))
        n[:, 1, :, :, 2] = -1
        skyrmion_radius = 0.8
        pos = torch.arange(0, grid_size, grid_size / num_sample)
        disp_to_center = pos - grid_size / 2
        dist_to_center = torch.sqrt(disp_to_center[:, None]**2 + disp_to_center[None, :]**2)
        theta = 2*torch.arcsin(torch.exp(-dist_to_center / (2*skyrmion_radius)))
        phi = torch.atan2(disp_to_center[None, :], disp_to_center[:, None])
        n[:, 0, :, :, 0] = torch.sin(theta) * torch.cos(phi)
        n[:, 0, :, :, 1] = torch.sin(theta) * torch.sin(phi)
        n[:, 0, :, :, 2] = torch.cos(theta)
        # n = torch.nn.functional.normalize(n, dim=-1)

        model = nlsm_model.NLSM_model(
            grid_size = grid_size,
            num_sample = num_sample,
            inter_layer_interaction=inter_layer_interaction,
            intra_layer_interaction=intra_layer_interaction,
            batch_size=1,
            metal_distance=d,
            num_bzone=num_bzone,
            gpu_memory=0
        )
        model.initialize(n)
        h, grad = model.hamiltonian(compute_grad=True, n=n)
        grad = grad[0]
        # print(model.n[0].flip(dims=[0]))
        theo_grad = (model.mesh_area / 2 / np.pi**2) * (model.n[0] + model.n[0].flip(dims=[0])) * model.inter
        theo_grad = theo_grad.float()
        curr = torch.empty((1, num_sample, num_sample, 2))
        # curr[:, 0] = 

        # print(grad.dtype, theo_grad.dtype)
        print("Success:", torch.allclose(theo_grad[0], grad[0]))
        # print((grad[0]).abs().max())
        # print((theo_grad[0] / grad[0])[300,300])
        # print(theo_grad[0,300,300], grad[0,300,300])
        # plt.matshow(theo_grad[0] / grad[0])
        # plt.show()
        # print(theo_grad.shape, grad.shape)
        # print(grad, theo_grad)

def test_grad_algorithm():
    def hamiltonian(compute_grad=False, n=None):
        if compute_grad:
            n.requires_grad_(True)
        hamiltonian = torch.sum((n[:, 0, :, :, :] +n[:, 1, :, :, :])**2, dim=(-1,-2,-3))
        if compute_grad:
            loss = hamiltonian.sum()
            loss.backward()
            hamiltonian.detach_()
            hamiltonian.requires_grad_(False)
            n.requires_grad_(False)
            return hamiltonian, n.grad
        else:
            return hamiltonian
    
    # n = torch.arange(2*10*10*3).reshape((1,2,10,10,3)).float()
    n = torch.arange(600).reshape((1,2,10,10,3)).float()
    n = torch.nn.functional.normalize(n, dim=-1)
    print(n.shape, n.dtype)
    h, grad = hamiltonian(True, n)
    print(h, grad.shape)
    # flip_n = torch.empty_like(n)
    # flip_n[:, 0] = n[:, 1]
    # flip_n[:, 1] = n[:, 0]
    # print(grad)
    # print("flip:", torch.allclose(n.flip(1), flip_n))
    print("Success:", torch.allclose(grad, 2*n+2*n.flip(1)))
    

def test_hamiltonian_speed():
    num_samples = torch.logspace(7, 11, 9, base=2, dtype=int)
    # grid_sizes = torch.logspace(3,5,5-3+1,base=2, dtype=int)
    d = 3
    time_no_grad = []
    time_grad = []
    num_bzone=2
    intra_layer_interaction = 1
    inter_layer_interaction = torch.ones(3)
    # num_bzones = torch.arange(2, 8, 2)
    # print(num_samples)
    grid_size = 32
    for num_sample in num_samples:

        model = nlsm_model.NLSM_model(
            grid_size = grid_size,
            num_sample = num_sample,
            inter_layer_interaction=inter_layer_interaction,
            intra_layer_interaction=intra_layer_interaction,
            batch_size=1,
            metal_distance=d,
            num_bzone=num_bzone,
            gpu_memory=0
        )
        model.initialize()
        # density = model.skyrmion_density()
        # plt.imshow(density[0,0])
        # plt.show()
        # plt.imshow(density[0,1])
        # plt.show()
        start = time.time_ns()
        model.hamiltonian()
        end = time.time_ns()
        time_no_grad.append(end-start)
        start = time.time_ns()
        model.hamiltonian(compute_grad=True)
        end = time.time_ns()
        time_grad.append(end-start)
    print(time_no_grad)
    print(time_grad)

    # regressor = utils.WeightedLinearRegressor()
    time_no_grad = np.array(time_no_grad)
    time_grad = np.array(time_grad)
    # regressor.fit(np.log(time_no_grad), None, np.log(time_grad), None)
    # regressor.fit(time_no_grad, None, time_grad, None)
    # slope = regressor.get_slope()
    # intercept = regressor.get_intercept()
    # print("Slope", slope)
    # print("Intercept", intercept)
    # time_pred = np.exp(slope * np.log(time_no_grad) + intercept)
    # time_pred = slope * time_no_grad + intercept
    time_ratio = time_grad / time_no_grad
    time_ratio_avg = np.average(time_ratio, weights=num_samples.numpy())

    plt.plot(num_samples, time_ratio, "b.", label="Experiment")
    # plt.plot(time_no_grad, time_pred, "r-", label="Regression")
    plt.axhline(time_ratio_avg, color="r", label=f"Average ratio {time_ratio_avg:.3f}")
    plt.plot()
    plt.xscale("log")
    # plt.yscale("log")
    plt.tick_params(direction="in")
    plt.xlabel("Grid size")
    plt.ylabel("Time ratio")
    plt.ylim(bottom=0)
    plt.legend()
    plt.show()
    

def test_regressor():
    a = np.arange(10)
    regressor = utils.WeightedLinearRegressor(biased=True)
    regressor.fit(a, None, 2 * a +2 + np.random.randn(10)*0.5, None)
    slope = regressor.get_slope()
    print("Slope", slope)


def test_canonical_ensemble_sim():
    temp = np.logspace(-2, 2, 5)
    result = np.empty_like(temp)
    batch_size = 3
    num_steps = 10000
    for i in range(len(temp)):
        model = spin_field_model.Spin_Field_Model(
            torch.tensor([1, 0, 0], dtype=float), batch_size=batch_size,
            gen_prob=0.5, gen_sigma=0.1)
        model.initialize()
        canonical_ens_simulator = canonical_ens_sim.Canonical_Ensemble_Simulator(model, 1/temp[i])
        canonical_ens_simulator.initialize()
        hamiltonian = torch.empty((batch_size, num_steps+1))
        hamiltonian[:, 0] = canonical_ens_simulator.hamiltonian
        for j in range(num_steps):
            canonical_ens_simulator.step()
            hamiltonian[:, j+1] = canonical_ens_simulator.hamiltonian
        result[i] = hamiltonian.mean().item()
    plt.plot(temp,result, "r.")
    temp = np.logspace(-2, 2, 100)
    plt.plot(temp, temp - 1 / np.tanh(1/temp), "b-")
    plt.show()


if __name__ == "__main__":
    # test_neighbor_generation_1()
    # test_neighbor_generation_2()

    # test_plot()
    # test_hamiltonian()
    # test_solid_angle()
    # test_speed()
    # test_grad()
    # a = A()
    # a.test()
    # test_no_grad()
    # test_torch_geometry()
    # test_grad2()
    # test_evolution_convergence()
    # test_tanhc_var1()
    # test_grad()
    # test_grad_algorithm()
    # test_hamiltonian_speed()
    # test_regressor()
    test_canonical_ensemble_sim()
