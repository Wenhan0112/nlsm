import numpy as np
import torch
import matplotlib.pyplot as plt
import nlsm_model
import time
import torchgeometry
import physical_fn as pypf

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
        n[:, 1, :, :, 2] = 1
        skyrmion_radius = 0.6
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

if __name__ == "__main__":
    # test_neighbor_generation_1()
    # test_neighbor_generation_2()
    # test_plot()
    test_hamiltonian()
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
