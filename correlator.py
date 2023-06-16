import torch
import numpy as np
import string
import itertools

ALPHABET_L = string.ascii_lowercase
ALPHABET_U = string.ascii_uppercase
cpu = torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else cpu

class Slicer():
    def __init__(self):
        self.slice = []
        self.labels = {}
    
    def add_slice(self, l):
        self.slice.append(slice(l))
    
    def add_element(self, label=""):
        if label not in self.labels.keys():
            self.labels[label] = []
        self.labels[label].append(len(self.slice))
        self.slice.append(None)
    
    def get_labels(self):
        return self.labels

    def get_idx(self, idx_dict):
        s = self.slice.copy()
        for t, idx in idx_dict.items():
            for i in range(len(idx)):
                s[self.labels[t][i]] = idx[i]
        return tuple(s)

class Accumulator():
    def __init__(self, shape, device=device, count_shape="Scalar"):
        if count_shape == "Scalar":
            self.count = torch.tensor(0, device=device, dtype=int)
        elif count_shape == "Same":
            self.count = torch.zeros(shape, device=device, dtype=int)
        else:
            self.count = torch.zeros(count_shape, device=device, dtype=int)
        self.shape = shape
        self.result = torch.zeros(shape, device=device)
    
    def reset(self):
        self.count[...] = 0
        self.result[...] = 0
    
    def update_batch(self, x):
        assert list(x.shape[-len(self.shape):]) == self.shape
        sum_axis = x.shape[:-len(self.shape)]
        self.count += np.prod(sum_axis)
        self.result += x.sum(list(range(len(sum_axis))))
    
    def update_single(self, x, count=1):
        assert list(x.shape) == self.shape
        self.count += count
        self.result += x
    
    def get_result(self):
        return self.result / self.count

class Base_Correlator():
    def reset(self):
        self.acc.reset()
    
    def get_result(self):
        return self.acc.get_result()

class Naive_Correlator(Base_Correlator):
    def __init__(self, shape, axis=[-1], device=device):
        dim = len(shape)
        assert dim > 0 and dim < len(ALPHABET_L)
        str1 = ALPHABET_U[:dim]
        str2 = ""
        str3 = ""
        new_shape = []
        axis = neg_to_pos_index(dim, axis)
        self.axis = axis
        for i in range(dim):
            new_shape.append(shape[i])
            if i in axis:
                str2 += ALPHABET_L[i]
                str3 += ALPHABET_U[i] + ALPHABET_L[i]
                new_shape.append(shape[i])
            else:
                str2 += ALPHABET_U[i]
                str3 += ALPHABET_U[i]
        self.shape = shape
        self.einsum_str = f"...{str1},...{str2}->...{str3}"
        self.acc = Accumulator(new_shape, device=device)
        self.device=device
    
    def update_batch(self, x):
        x = torch.einsum(self.einsum_str, x, x)
        self.acc.update_batch(x)
    
    def update_single(self, x, count=1):
        assert list(x.shape) == self.shape
        x = torch.einsum(self.einsum_str, x, x)
        self.acc.update_single(x, count)

class Correlator(Base_Correlator):

    suppoerted_boundaries = ("N","n","t","tp","r")

    def __init__(self, shape, sym_type, device=device):
        """
        @params sym_type
        """
        self.device = device
        self.shape = np.array(shape)
        self.sym_type = np.array(sym_type)
        dim = len(shape)
        self.axis = {t:np.flatnonzero(self.sym_type == t) 
            for t in self.suppoerted_boundaries}
        self.axis_shape = {t:self.shape[self.axis[t]]
            for t in self.suppoerted_boundaries}
        

        self.new_shape = []
        self.single_count = []
        self.slicer = {"t": Slicer(), "result": Slicer()}
        str1, str2, str3 = "", "", ""
        
        for i in range(dim):
            if i in self.axis["t"]:
                str1 += ALPHABET_U[i]
                str2 += ALPHABET_U[i]
                self.new_shape.append(self.shape[i])
                self.slicer["t"].add_element("t")
                self.slicer["result"].add_element("t")
                self.single_count.append(np.arange(self.shape[i], 0, -1))
            elif i in self.axis["tp"]:
                str1 += ALPHABET_U[i]
                str2 += ALPHABET_U[i]
                self.new_shape.append(shape[i])
                self.slicer["t"].add_slice(self.shape[i])
                self.slicer["result"].add_element("tp")
                self.single_count.append([shape[i]])
            elif i in self.axis["r"]:
                str1 += ALPHABET_U[i]
                str2 += ALPHABET_U[i]
                self.new_shape.append(self.shape[i] // 2 + 1)
                self.slicer["t"].add_slice(self.shape[i])
                self.slicer["result"].add_element("r")
                single_count_slice = np.full(self.shape[i] // 2 + 1, 2)
                single_count_slice[0] = 1
                if shape[i] % 2 == 0:
                    single_count_slice[-1] = 1
                self.single_count.append(single_count_slice)
            elif i in self.axis["n"]:
                str1 += ALPHABET_U[i]
                str2 += ALPHABET_L[i]
                str3 += ALPHABET_U[i] + ALPHABET_L[i]
                self.new_shape.append(self.shape[i])
                self.new_shape.append(self.shape[i])
                self.slicer["t"].add_slice(self.shape[i])
                self.slicer["result"].add_slice(self.shape[i])
                self.slicer["result"].add_slice(self.shape[i])
                self.single_count.append([1])
                self.single_count.append([1])
            else :
                str1 += ALPHABET_U[i]
                str2 += ALPHABET_U[i]
                str3 += ALPHABET_U[i]
                self.new_shape.append(self.shape[i])
                self.slicer["t"].add_slice(self.shape[i])
                self.slicer["result"].add_slice(self.shape[i])
                self.single_count.append([1])
        self.einsum_str = f"...{str1},...{str2}->...{str3}"

        count_einsum_str = ",".join(ALPHABET_L[:len(self.single_count)])
        count_einsum_str += "->" + ALPHABET_L[:len(self.single_count)]
        self.single_count = np.einsum(count_einsum_str, *self.single_count)
        self.single_count = torch.tensor(self.single_count, 
            device=device, dtype=int)
        self.acc = Accumulator(self.new_shape, device=self.device, 
            count_shape=self.single_count.shape)

    def update_single(self, x, count=1):
        """
        
        """
        assert tuple(x.shape) == tuple(self.shape)
        iterator = {}
        for t in ("t","tp","r"):
            iterator[t] = itertools.product(
                *[range(self.shape[i]) for i in self.axis[t]])
        iterator = dict_product(iterator)
        
        result = torch.empty(self.new_shape, device=self.device)
        for shifts in iterator:
            xs = x
            x0 = x
            if shifts["t"]:
                xs = xs[self.slicer["t"].get_idx({"t":
                    [slice(i, None) for i in shifts["t"]]})]
                x0 = x0[self.slicer["t"].get_idx({"t":
                    [slice(self.axis_shape["t"][i] - s) 
                    for i, s in enumerate(shifts["t"])]})]
            if shifts["tp"]:
                xs = xs.roll(shifts=shifts["tp"], dims=tuple(self.axis["tp"]))
            if shifts["r"]:
                xs = xs.roll(shifts=shifts["r"], dims=tuple(self.axis["r"]))
            shifts["r"] = [min(self.axis_shape["r"][i] - s, s) 
                for i, s in enumerate(shifts["r"])]
            idx = self.slicer["result"].get_idx(shifts)
            result[idx] = torch.einsum(self.einsum_str, x0, xs)
        self.acc.update_single(result, count=count * self.single_count)

class Homogeneous_Isotropic_Correlator(Base_Correlator):
    def __init__(self, shape, sym_axis=[], non_sym_axis=[], device=device):
        self.device = device
        self.shape = shape
        self.new_shape = []
        self.no_sym_shape = []
        self.slicer = Slicer()
        dim = len(shape)
        sym_axis = neg_to_pos_index(dim, sym_axis)
        non_sym_axis = neg_to_pos_index(dim, non_sym_axis)
        self.degeneracy = 1
        str1, str2, str3 = "", "", ""
        for i in range(dim):
            if i in sym_axis:
                str1 += ALPHABET_U[i]
                str2 += ALPHABET_U[i]
                self.new_shape.append((shape[i]+1)//2)
                self.slicer.add_element()
                self.degeneracy *= shape[i] * 2
            elif i in non_sym_axis:
                str1 += ALPHABET_U[i]
                str2 += ALPHABET_L[i]
                str3 += ALPHABET_U[i] + ALPHABET_L[i]
                self.new_shape.append(shape[i])
                self.new_shape.append(shape[i])
                self.no_sym_shape.append(shape[i])
                self.no_sym_shape.append(shape[i])
                self.slicer.add_slice(shape[i])
                self.slicer.add_slice(shape[i])
            else:
                str1 += ALPHABET_U[i]
                str2 += ALPHABET_U[i]
                str3 += ALPHABET_U[i]
                self.new_shape.append(shape[i])
                self.no_sym_shape.append(shape[i])
                self.slicer.add_slice(shape[i])
        self.acc = Accumulator(self.new_shape, device=self.device)
        self.sym_axis = list(sym_axis)
        self.non_sym_axis = list(non_sym_axis)
        self.einsum_str = f"...{str1},...{str2}->...{str3}"
    
    def reset(self):
        self.acc.reset()
    
    def update_single(self, x, count=1):
        """
        C_{pqkK} = \sum_{b}\sum_{|i-I|=p}\sum_{|j-J|=q}J_{bijk}J_{bIJK}

        where b is the currernt density in each MC iteration, i represent the x-position 
        in the grid, j represent the y-position in the grid, and k represents the direction
        """
        assert tuple(x.shape) == tuple(self.shape)
        iterator = [range((self.shape[i]+1)//2) for i in self.sym_axis]
        iterator = itertools.product(*iterator)
        result = torch.empty(self.new_shape, device=self.device)
        for shifts in iterator:
            shift_iterator = [(s, -s) for s in shifts]
            shift_iterator = itertools.product(*shift_iterator)
            res = torch.zeros(self.no_sym_shape, device=self.device)
            for pshifts in shift_iterator:
                xs = x.roll(shifts=pshifts, dims=self.sym_axis)
                res += torch.einsum(self.einsum_str, x, xs)
            idx = self.slicer.get_idx(shifts)
            result[idx] = res
        self.acc.update_single(result, count=self.degeneracy)
    
    def get_result(self):
        return self.acc.get_result()

def neg_to_pos_index(dim, axis):
    axis = np.array(axis)
    return np.where(axis >= 0, axis, axis + dim)

def dict_product(dicts):
    return (dict(zip(dicts.keys(), x)) for x in itertools.product(*dicts.values()))


if __name__ == "__main__":


    # c = Naive_Correlator([5,2,10,10,3], [-3,-2,-1])
    # c = Homogeneous_Correlator([5,8,10,10,2], sym_axis=[-4,-3,-2], if_roll=[False, True, True], non_sym_axis=[-1],)
    # c.update_single(torch.ones(5,8,10,10,2))
    # print(c.get_result())
    # print(list(neg_iterator((1,1,0))))
    # test_slicer()
    # print(neg_to_pos_index(5, [0,1,3,-4,-1]))
    # print(convert_sym_type([None, "t","r", "r", "n"]))
    pass

    
