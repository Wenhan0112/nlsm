import torch
import torchgeometry
import matplotlib.pyplot as plt

def tanhc(x: torch.tensor) -> torch.tensor:
    """
    Compute the tanhc function of the input, elementwise
    @params x (torch.tensor): Input tensor
    @return (torch.tensor): Output tensor
    """
    return torch.where(x != 0, torch.tanh(x) / x, 1.)

def tanhc_var1(x1: torch.tensor, x2: torch.tensor, eps=1e-6) -> torch.tensor:
    """
    Compute tanhc(x1 * x2) * x2
    """
    x1x2 = x1 * x2
    small = x1x2 < eps
    return torch.where(small, x2, torch.tanh(x1x2) / x1)

def angle_axis_to_rot_mat(x: torch.tensor) -> torch.tensor:
    """
    Compute the rotation matrix representation given the axis-angle 
    representation batchly. 
    @params x (torch.tensor): Input tensor.
        Constriant: x.ndim >= 1 and x.shape[-1] == 3
        x[..., i, j]: The axis angle representation index J of rotation I.
    @return (torch.tensor): Output rotation matrix representation.
        Constraint: RETURN.shape == x.shape + (3,)
        RETURN[..., i, j, k]: The rotation matrix representation index (J, K) 
            of rotation I. 
    """
    assert x.ndim >= 1 and x.shape[-1] == 3, \
        f"Incompatible input shape: {x.shape}"
    x_batchs = x.shape[:-1]
    x = x.reshape((-1, 3))
    y = torchgeometry.angle_axis_to_rotation_matrix(x)
    y = y[:, :3, :3].reshape(x_batchs + (3, 3))
    return y


