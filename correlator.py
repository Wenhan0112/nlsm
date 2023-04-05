import torch
import numpy as np
import matplotlib.pyplot as plt
import string
import itertools

ALPHABET_L = string.ascii_lowercase
ALPHABET_U = string.ascii_uppercase
cpu = torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else cpu

class Slicer():
    def __init__(self):
        self.slice = []
        self.dict = []
        self.labels = []
    
    def add_slice(self, l):
        self.slice.append(slice(l))
    
    def add_element(self, label=""):
        self.dict.append(len(self.slice))
        self.labels.append(label)
        self.slice.append(None)
    
    def get_labels(self, if_print=False):
        if if_print:
            print(self.labels)
        return self.labels

    def get_idx(self, idx):
        s = self.slice.copy()
        for i in range(len(idx)):
            s[self.dict[i]] = idx[i]
        return tuple(s)

class Accumulator():
    def __init__(self, shape, device=device, count_shape="Scalar"):
        if count_shape == "Scalar":
            self.count = 0
        elif count_shape == "Same":
            self.count = torch.zeros(shape, device=device, dtype=int)
        else:
            raise ValueError(f"Count shape {count_shape} not implemented!")
        self.shape = shape
        self.result = torch.zeros(shape, device=device)
    
    def reset(self):
        self.count = 0
        self.result = 0
    
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

class Naive_Correlator():
    def __init__(self, shape, axis=[-1], device=device):
        dim = len(shape)
        assert dim > 0 and dim < len(ALPHABET_L)
        str1 = ALPHABET_U[:dim]
        strtemp = ALPHABET_L[:dim]
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
    
    def reset(self):
        self.acc.reset()
    
    def update_batch(self, x):
        x = torch.einsum(self.einsum_str, x, x)
        self.acc.update_batch(x)
    
    def update_single(self, x, count=1):
        assert list(x.shape) == self.shape
        x = torch.einsum(self.einsum_str, x, x)
        self.acc.update_single(x, count)
    
    def get_result(self):
        return self.acc.get_result()
    
class Homogeneous_Isotropic_Correlator():
    def __init__(self, shape, sym_axis=[-1], non_sym_axis=[], device=device):
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
        self.sym_axis = sym_axis
        self.non_sym_axis = non_sym_axis
        self.einsum_str = f"...{str1},...{str2}->...{str3}"
    
    def reset(self):
        self.acc.reset()
    
    def update_single(self, x, count=1):
        assert list(x.shape) == self.shape
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
    for i, ax in enumerate(axis.copy()):
        if ax < 0:
            axis[i] = dim + ax
    return axis


if __name__ == "__main__":
    def neg_iterator(shifts):
        shift_iterator = [set((s, -s)) for s in shifts]
        shift_iterator = itertools.product(*shift_iterator)
        yield from shift_iterator


    def test_slicer():
        slicer = Slicer()
        slicer.add_slice(5)
        slicer.add_slice(2)
        slicer.add_element()
        slicer.add_element()
        slicer.add_slice(3)

        print(slicer.get_idx([2, 3]))


    # c = Naive_Correlator([5,2,10,10,3], [-3,-2,-1])
    # c = Homogeneous_Isotropic_Correlator([5,2,10,10,3], sym_axis=[-3,-2], non_sym_axis=[-1])
    # c.update_single(torch.ones(5,2,10,10,3)*2)
    # print(c.get_result())
    # print(list(neg_iterator((1,1,0))))
    # test_slicer()

    
