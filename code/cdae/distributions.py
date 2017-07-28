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

    return -0.5 * (torch.pow(x - mean, 2) / var + torch.log(2 * var * np.pi + util.epsilon))


def categorical_sample(categories, probabilities):
    """
    Returns a Tensor of samples from Categorical(categories, probabilities)

    input:
        categories: Tensor [num_categories, dim_1, ..., dim_N] (or [num_categories])
        probabilities: Tensor [num_categories, dim_1, ..., dim_N] (or [num_categories])

    output: Tensor [dim_1, ..., dim_N] (or [1])
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

    if len(output_size) == 0:
        if util.cuda:
            return torch.from_numpy(output_numpy_flattened).float().cuda()
        else:
            return torch.from_numpy(output_numpy_flattened).float()
    else:
        if util.cuda:
            return torch.from_numpy(output_numpy_flattened).float().view(*output_size).cuda()
        else:
            return torch.from_numpy(output_numpy_flattened).float().view(*output_size)


def categorical_logpdf(value, categories, probabilities):
    """
    Returns categorical logpdfs.

    input:
        value: Tensor/Variable [dim_1, ..., dim_N] (or [1])
        categories: Tensor/Variable [num_categories, dim_1, ..., dim_N] (or [num_categories])
        probabilities: Tensor/Variable [num_categories, dim_1, ..., dim_N] (or [num_categories])

    output: Tensor/Variable [dim_1, ..., dim_N] (or [1])
    """

    cat_size = categories.size()
    num_categories, output_size = cat_size[0], cat_size[1:]

    if len(output_size) == 0:
        value_expanded = value.expand_as(categories)
    else:
        value_expanded = value.unsqueeze(0).expand_as(categories)
    mask = (value_expanded == categories).float().cuda() if \
        util.cuda else \
        (value_expanded == categories).float()
    return torch.log(torch.sum(probabilities * mask, dim=0) + util.epsilon)


def gumbel_sample(location, scale):
    """
    Returns a Tensor of samples from Gumbel(location, scale).

    input:
        location: Tensor [dim_1, ..., dim_N]
        scale: Tensor [dim_1, ..., dim_N]

    output: Tensor [dim_1, ..., dim_N]
    """

    return location - scale * torch.log(
        -torch.log(torch.rand(location.size() + util.epsilon)) + util.epsilon
    )


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

    return -(temp + torch.exp(-temp)) - torch.log(scale + util.epsilon)


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

    numerator = torch.exp((torch.log(location + util.epsilon) + gumbels) / temperature_expanded)
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
        (num_categories - 1) * torch.log(temperature + util.epsilon) + \
        torch.sum(
            torch.log(
                location + util.epsilon
            ) - (temperature_expanded + 1) * torch.log(value + util.epsilon),
            dim=0
        ).squeeze(0) - \
        num_categories * torch.log(torch.sum(
            location * (value**(-temperature_expanded)),
            dim=0
        ).squeeze(0) + util.epsilon)


def discrete_sample(probabilities):
    """
    Returns a Tensor of samples from a Discrete(probabilities).

    input:
        probabilities: Tensor [num_categories, dim_1, ..., dim_N] (or [num_categories])

    output: Tensor [dim_1, ..., dim_N] (or [1])
    """

    num_categories = probabilities.size(0)
    categories = torch.arange(0, num_categories)
    for n in range(probabilities.ndimension() - 1):
        categories = categories.unsqueeze(-1)
    categories = categories.expand_as(probabilities)

    return categorical_sample(categories, probabilities)


def discrete_logpdf(value, probabilities):
    """
    Returns Discrete logpdfs.

    input:
        value: Tensor/Variable [dim_1, ..., dim_N] (or [1])
        probabilities: Tensor/Variable [num_categories, dim_1, ..., dim_N] (or [num_categories])

    output: Tensor/Variable [dim_1, ..., dim_N] (or [1])
    """

    num_categories = probabilities.size(0)
    categories = torch.arange(0, num_categories)
    for n in range(probabilities.ndimension() - 1):
        categories = categories.unsqueeze(-1)
    categories = categories.expand_as(probabilities)
    if isinstance(probabilities, Variable):
        categories = Variable(categories)

    return categorical_logpdf(value, categories, probabilities)


def one_hot_discrete_sample(probabilities):
    """
    Returns a Tensor of samples from a Discrete(probabilities) in a one-hot form.

    input:
        probabilities: Tensor [num_categories, dim_1, ..., dim_N] (or [num_categories])

    output: Tensor [num_categories, dim_1, ..., dim_N] (or [num_categories])
    """

    output = torch.zeros(probabilities.size())
    d = discrete_sample(probabilities)

    if probabilities.ndimension() == 1:
        return output.scatter_(0, d.long(), 1)
    else:
        return output.scatter_(0, d.long().unsqueeze(0), 1)


def one_hot_discrete_logpdf(value, probabilities):
    """
    Returns logpdfs of one-hot valued Discrete.

    input:
        value: Tensor/Variable [num_categories, dim_1, ..., dim_N] (or [num_categories])
        probabilities: Tensor/Variable [num_categories, dim_1, ..., dim_N] (or [num_categories])

    output: Tensor/Variable [dim_1, ..., dim_N] (or [1])
    """

    return torch.log(torch.sum(value * probabilities, dim=0) + util.epsilon).squeeze(0)
