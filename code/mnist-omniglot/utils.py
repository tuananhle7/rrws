import torch


def safe_log(x, epsilon=1e-8):
    return torch.log(torch.clamp(x, min=epsilon))


def logsumexp(values, dim=0, keepdim=False):
    """Logsumexp of a Tensor/Variable.

    See https://en.wikipedia.org/wiki/LogSumExp.

    input:
        values: Tensor/Variable [dim_1, ..., dim_N]
        dim: n

    output: result Tensor/Variable
        [dim_1, ..., dim_{n - 1}, dim_{n + 1}, ..., dim_N] where

        result[i_1, ..., i_{n - 1}, i_{n + 1}, ..., i_N] =
            log(sum_{i_n = 1}^N exp(values[i_1, ..., i_N]))
    """

    values_max, _ = torch.max(values, dim=dim, keepdim=True)
    result = values_max + torch.log(torch.sum(
        torch.exp(values - values_max), dim=dim, keepdim=True
    ))
    return result if keepdim else result.squeeze(dim)


def continuous_relaxation(z, temperature, epsilon=1e-8):
    return 1 / (1 + torch.exp(-z / (temperature + epsilon)))


def heaviside(x):
    return x >= 0


def reparam(u, theta, epsilon=1e-8):
    return safe_log(theta) - safe_log(1 - theta) + safe_log(u) - safe_log(1 - u)


def conditional_reparam(v, theta, b, epsilon=1e-8):
    # NB: This is a buggy implementation that performs the best
    # if b.data[0] == 1:
    #     return torch.log(v / ((1 - v) * (1 - theta)) + 1 + epsilon)
    # else:
    #     return -torch.log(v / ((1 - v) * theta) + 1 + epsilon)

    # NB: This implementation gives "Segmentation Fault 11"
    # result = Variable(torch.zeros(*b.size()))
    # for i in range(2):
    #     v_i = v[b.data == i]
    #     theta_i = theta[b.data == i]
    #     if i == 1:
    #         result[b.data == i] = torch.log(v_i / ((1 - v_i) * (1 - theta_i)) + 1 + epsilon)
    #     else:
    #         result[b.data == i] = -torch.log(v_i / ((1 - v_i) * theta_i) + 1 + epsilon)
    #
    # return result

    # NB: This implementation is inefficient but should be correct
    return (
        safe_log(v / ((1 - v) * (1 - theta)) + 1) * (b == 1).float() +
        (-safe_log(v / ((1 - v) * theta) + 1)) * (b == 0).float()
    )
