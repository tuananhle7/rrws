import torch
import numpy as np
import cdae.util as util


def normal_sample(mean, var):
    """
    Returns a Tensor of samples from Normal(mean, var)

    input:
        mean: Tensor [dim_1 * ... * dim_N]
        var: Tensor [dim_1 * ... * dim_N]
    output: Tensor [dim_1 * ... * dim_N]
    """

    ret = torch.Tensor(mean.size()).normal_()
    return ret.mul_(torch.sqrt(var)).add_(mean)


def normal_logpdf(x, mean, var):
    """
    Returns normal logpdfs.

    input:
        x: Tensor/Variable [dim_1, ..., dim_N]
        mean: Tensor/Variable [dim_1, ..., dim_N]
        var: Tensor/Variable [dim_1, ..., dim_N]

    output: Tensor/Variable [dim_1, ..., dim_N]
    """

    return -0.5 * (torch.pow(x - mean, 2) / var + torch.log(2 * var * np.pi))


def categorical_sample(categories, probabilities):
    """
    Returns a Tensor of samples from Categorical(categories, probabilities)

    input:
        categories: Tensor [num_categories, dim_1, ..., dim_N]
        probabilities: Tensor [num_categories, dim_1, ..., dim_N]

    output: Tensor [dim_1, ..., dim_N]
    """

    cat_size = categories.size()
    num_categories, output_size = cat_size[0], cat_size[1:]
    categories_flattened = categories.contiguous().view(num_categories, -1)
    probabilities_flattened = probabilities.contiguous().view(num_categories, -1)

    output_nelement = categories_flattened.size(1)
    output_numpy_flattened = np.zeros(output_nelement)
    for n in range(output_nelement):
        output_numpy_flattened[n] = np.random.choice(
            categories_flattened[:, n].numpy(),
            p=probabilities_flattened[:, n].numpy()
        )

    if util.cuda:
        return torch.from_numpy(output_numpy_flattened).float().view(*output_size).cuda()
    else:
        return torch.from_numpy(output_numpy_flattened).float().view(*output_size)


def categorical_logpdf(x, categories, probabilities):
    """
    Returns categorical logpdfs

    input:
        x: Tensor/Variable [dim_1, ..., dim_N]
        categories: Tensor/Variable [num_categories, dim_1, ..., dim_N]
        probabilities: Tensor/Variable [num_categories, dim_1, ..., dim_N]

    output: Tensor/Variable [dim_1, ..., dim_N]
    """

    num_categories = categories.size(0)
    x_expanded = x.unsqueeze(0).expand_as(categories)
    mask = (x_expanded == categories).float().cuda() if \
        util.cuda else \
        (x_expanded == categories).float()
    return torch.log(torch.sum(probabilities * mask, dim=0))


def gumbel_sample(location, scale):
    """
    Returns a Tensor of samples from Gumbel(location, scale).

    input:
        location: Tensor [dim_1, ..., dim_N]
        scale: Tensor [dim_1, ..., dim_N]

    output: Tensor [dim_1, ..., dim_N]
    """

    return location - scale * torch.log(-torch.log(torch.rand(location.size())))


def gumbel_logpdf(value, location, scale):
    """
    Returns Gumbel logpdfs.

    input:
        value: Tensor/Variable [dim_1, ..., dim_N]
        location: Tensor/Variable [dim_1, ..., dim_N]
        scale: Tensor/Variable [dim_1, ..., dim_N]

    output: Tensor/Variable [dim_1, ..., dim_N]
    """

    temp = (value - location) / scale

    return -(temp + torch.exp(-temp)) - torch.log(scale)


def concrete_sample(location, temperature):
    """
    Returns a Tensor of samples from Concrete(location, temperature).

    input:
        location: Tensor [num_categories, dim_1, ..., dim_N] (or [num_categories])
        temperature: Tensor [dim_1, ..., dim_N] (or int/float/[1])

    output: Tensor [num_categories, dim_1, ..., dim_N] (or [num_categories])
    """

    if location.ndimension() == 1:
        if isinstance(temperature, (int, float)):
            temperature = torch.Tensor([temperature])
        temperature_expanded = temperature.expand_as(location)
    else:
        temperature_expanded = temperature.unsqueeze(0).expand_as(location)
    gumbels = gumbel_sample(torch.zeros(location.size()), torch.ones(location.size()))

    numerator = torch.exp((torch.log(location) + gumbels) / temperature_expanded)
    denominator = torch.sum(numerator, dim=0).expand_as(numerator)

    return numerator / denominator


def concrete_logpdf(value, location, temperature):
    """
    Returns a Tensor of Concrete logpdfs.

    input:
        value: Tensor/Variable [num_categories, dim_1, ..., dim_N] (or [num_categories])
        location: Tensor/Variable [num_categories, dim_1, ..., dim_N] (or [num_categories])
        temperature: Tensor/Variable [dim_1, ..., dim_N] (or int/float/[1])
    output: Tensor/Variable [dim_1, ..., dim_N] (or [1])
    """

    num_categories, *_ = value.size()

    if location.ndimension() == 1:
        if isinstance(temperature, (int, float)):
            if isinstance(location, Variable):
                temperature = Variable(torch.Tensor([temperature]))
            else:
                temperature = torch.Tensor([temperature])
        temperature_expanded = temperature.expand_as(location)
    else:
        temperature_expanded = temperature.unsqueeze(0).expand_as(location)

    return torch.sum(torch.arange(1, num_categories)) + \
        (num_categories - 1) * torch.log(temperature) + \
        torch.sum(
            torch.log(location) - (temperature_expanded + 1) * torch.log(value),
            dim=0
        ).squeeze(0) - \
        num_categories * torch.log(torch.sum(
            location * (value**(-temperature_expanded)),
            dim=0
        ).squeeze(0))
