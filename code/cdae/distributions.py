import torch
import numpy as np

def normal_sample(mean, var):
    '''
    Returns a torch.Tensor of samples from Normal(mean, var)

    input:
        mean: torch.Tensor [dim_1 * ... * dim_N]
        var: torch.Tensor [dim_1 * ... * dim_N]
    output: torch.Tensor [dim_1 * ... * dim_N]
    '''

    ret = torch.FloatTensor(mean.size()).normal_()
    return ret.mul_(torch.sqrt(var)).add_(mean)

def normal_logpdf(x, mean, var):
    '''
    Returns normal logpdfs.

    input:
        x: Tensor/Variable [dim_1, ..., dim_N]
        mean: Tensor/Variable [dim_1, ..., dim_N]
        var: Tensor/Variable [dim_1, ..., dim_N]

    output: Tensor/Variable [dim_1, ..., dim_N]
    '''

    return (-0.5 * torch.pow(x - mean, 2) / var - 0.5 * torch.log(2 * var * np.pi))

def categorical_sample(categories, probabilities):
    '''
    Returns a torch.Tensor of samples from Categorical(categories, probabilities)

    input:
        categories: torch.Tensor [num_categories, dim_1, ..., dim_N]
        probabilities: torch.Tensor [num_categories, dim_1, ..., dim_N]

    output: torch.Tensor [dim_1, ..., dim_N]
    '''

    cat_size = categories.size()
    num_categories, output_size = cat_size[0], cat_size[1:]
    categories_flattened = categories.contiguous().view(num_categories, -1)
    probabilities_flattened = probabilities.contiguous().view(num_categories, -1)

    output_nelement = categories_flattened.size(1)
    output_numpy_flattened = np.zeros(output_nelement)
    for n in range(output_nelement):
        output_numpy_flattened[n] = np.random.choice(categories_flattened[:, n].numpy(), p=probabilities_flattened[:, n].numpy())

    return torch.from_numpy(output_numpy_flattened).float().view(*output_size)

def categorical_logpdf(x, categories, probabilities):
    '''
    Returns categorical logpdfs

    input:
        x: Tensor/Variable [dim_1, ..., dim_N]
        categories: Tensor/Variable [num_categories, dim_1, ..., dim_N]
        probabilities: Tensor/Variable [num_categories, dim_1, ..., dim_N]

    output: Tensor/Variable [dim_1, ..., dim_N]
    '''

    num_categories = categories.size(0)
    x_expanded = x.unsqueeze(0).expand_as(categories)
    mask = (x_expanded == categories).float()
    return torch.log(torch.sum(probabilities * mask, dim=0))
