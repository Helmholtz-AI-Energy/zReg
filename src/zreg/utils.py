import torch


def squared_kernel_sum(x: torch.Tensor, y: torch.Tensor) -> float:
    return squared_kernel(x, y).sum() / (x.shape[0] * x.shape[1] * y.shape[0])


def squared_kernel(x, y):
    # this is the l2**2 norm of the rows with respect to all the other rows
    # NOTE: this may need to be transposed. the result will be [y.shape[0], x.shape[0]]
    #       easy to fix, but need to check later
    return torch.sum((x[None, :, :] - y[:, None, :]) ** 2, dim=2)


def rbf_kernel(x, y, beta: float):
    diff2 = squared_kernel(x, y)
    return torch.exp(-diff2 / (2.0 * beta))


def tps_kernel_2d(x, y):
    eps = 1e-9
    diff2 = squared_kernel(x, y)
    return torch.where(diff2 > eps, diff2 * torch.log(torch.sqrt(diff2)), 0.0)


def tps_kernel_3d(x, y):
    diff2 = squared_kernel(x, y)
    return -diff2.sqrt()


def inverse_multiquadric_kernel(x, y, c: float):
    diff2 = squared_kernel(x, y)
    return 1.0 / (diff2 + c).sqrt()
