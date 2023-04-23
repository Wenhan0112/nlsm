import correlator
import torch
import ising_model
import numpy as np
import itertools
import bidict

cpu = torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else cpu

CONVERSION_TABLE = bidict.bidict({
    None: 0,
    "n": 1,
    "t": 2,
    "tp": 3,
    "r": 4
})

def test_correlator():
    batch_size = 2
    l = 6
    steps = 100000
    temperature = 1.
    beta = 1 / temperature
    boundary = "periodic"

    model = ising_model.Ising_Model_1D(l, gen_prob = 0.2, batch_size=batch_size, boundary=boundary)
    model.initialize()
    corr = correlator.Correlator((batch_size, l), ("N", "tp"))
    def callback(m, step):
        if step > steps // 10:
            corr.update_single(m.get_spin())
    model.canonical_ens_sim(beta, steps, show_bar=True, callback=callback)
    corrfn = corr.get_result().mean(axis=0)

    model = ising_model.Ising_Model_1D(l, gen_prob = 0.2, batch_size=2**l, boundary=boundary)
    model.initialize(model.get_all_states())
    hamiltonian = model.hamiltonian()
    prob = torch.exp(-beta * hamiltonian)
    prob /= prob.sum()
    spin = model.get_spin()
    if boundary == "periodic":
        corr_t = np.empty(l)
        for i in range(l):
            corr_t[i] = torch.sum(spin * spin.roll(i, -1) * prob[:, None])
        print(corrfn)
        print(corr_t / l)
    elif boundary == "open":
        corr_t = np.empty(l)
        for i in range(l):
            corr_t[i] = torch.sum(spin[:, :l-i] * spin[:,i:] * prob[:, None])
        print(corrfn)
        print(corr_t / torch.arange(l,0,-1))

def test_iterator():
    shape = [2,8,10,10,2]
    corr_axis = [[1],[],[2,3],[4]]
    iterator = (itertools.product(*[range(shape[i]) for i in axis])
        for axis in corr_axis)
    iterator = itertools.product(*iterator)
    for i in iterator:
        print(i)

def test_dict_product():
    get_axis = {"t":(1,), "tp":tuple(), "r":(2, 3)}
    iter_len = (1, 5, 3, 3, 1)
    iterator = {}
    for t in ("t","tp","r"):
        iterator[t] = itertools.product(
            *[range(iter_len[i]) for i in get_axis[t]])
    iterator = correlator.dict_product(iterator)
    for i in iterator:
        print(i)

def test_slicer():
    slicer = correlator.Slicer()
    slicer.add_slice(5)
    slicer.add_element("r")
    slicer.add_element("t")
    slicer.add_element("r")
    slicer.add_slice(2)
    idx_dict = {
        "t": (2,),
        "r" : (3, 4)
    }
    print(slicer.get_idx(idx_dict))

# def bool_idx(l, b, invert=False):
#     if invert:
#         return [i for i, j in zip(l, b) if not j]
#     return [i for i, j in zip(l, b) if j]

# def convert_sym_type(sym_type, conversion_table=CONVERSION_TABLE):
#     return np.array([conversion_table[i] for i in sym_type], dtype=int)

if __name__ == "__main__":
    test_correlator()
    # test_iterator()