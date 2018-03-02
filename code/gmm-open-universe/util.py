import numpy as np
import scipy.stats
import torch


def lognormexp(values, dim=0):
    """Exponentiates, normalizes and takes log of a Tensor/Variable/np.ndarray.

    input:
        values: Tensor/Variable/np.ndarray [dim_1, ..., dim_N]
        dim: n
    output:
        result: Tensor/Variable/np.ndarray [dim_1, ..., dim_N]
            where result[i_1, ..., i_N] =

                                 exp(values[i_1, ..., i_N])
            log( ------------------------------------------------------------ )
                    sum_{j = 1}^{dim_n} exp(values[i_1, ..., j, ..., i_N])
    """

    if isinstance(values, np.ndarray):
        log_denominator = scipy.special.logsumexp(
            values, axis=dim, keepdims=True
        )
        # log_numerator = values
        return values - log_denominator
    else:
        log_denominator = logsumexp(values, dim=dim, keepdim=True)
        # log_numerator = values
        return values - log_denominator


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


def generate_traces(num_traces, num_clusters_probs, mean_1, std_1, mixture_probs, means_2, stds_2, obs_std, generate_obs=True):
    traces = []
    for trace_idx in range(num_traces):
        trace = []
        num_traces = np.random.choice(np.array([1, 2], dtype=float), replace=True, p=num_clusters_probs)
        trace.append(num_traces)
        if num_traces == 1:
            x = np.random.normal(mean_1, std_1)
            trace.append(x)
        else:
            z = np.random.choice(np.arange(len(mixture_probs), dtype=float), replace=True, p=mixture_probs)
            trace.append(z)
            x = np.random.normal(means_2[int(z)], stds_2[int(z)])
            trace.append(x)

        if generate_obs:
            y = np.random.normal(x, obs_std)
            trace.append(y)

        traces.append(trace)

    return traces


def get_pdf_from_traces(traces_without_obs, k_points, z_points, x_points):
    ks = []
    zs = []
    xs = []
    for trace in traces_without_obs:
        if len(trace) == 2:
            ks.append(trace[0])
            xs.append(trace[1])
        else:
            ks.append(trace[0])
            zs.append(trace[1])
            xs.append(trace[2])

    return [np.sum(np.array(ks) == i) / len(ks) for i in k_points], \
        [np.sum(np.array(zs) == i) / len(zs) for i in z_points], \
        scipy.stats.gaussian_kde(xs).evaluate(x_points)


def get_prior_pdf(x_points, num_samples, num_clusters_probs, mean_1, std_1, mixture_probs, means_2, stds_2, obs_std):
    traces = generate_traces(num_samples, num_clusters_probs, mean_1, std_1, mixture_probs, means_2, stds_2, obs_std, False)
    return get_pdf_from_traces(traces, range(1, 1 + len(num_clusters_probs)), range(len(mixture_probs)), x_points)


def get_posterior_pdf(x_points, num_samples, obs, num_clusters_probs, mean_1, std_1, mixture_probs, means_2, stds_2, obs_std, num_importance_samples=None):
    if num_importance_samples is None:
        num_importance_samples = num_samples

    traces = generate_traces(num_importance_samples, num_clusters_probs, mean_1, std_1, mixture_probs, means_2, stds_2, obs_std, False)
    log_weights = np.zeros([num_importance_samples])
    for trace_idx, trace in enumerate(traces):
        log_weights[trace_idx] = scipy.stats.norm.logpdf(obs, trace[-1], obs_std)

    weights = np.exp(lognormexp(np.array(log_weights)))
    resampled_traces = np.random.choice(traces, size=num_samples, replace=True, p=weights)

    return get_pdf_from_traces(resampled_traces, range(1, 1 + len(num_clusters_probs)), range(len(mixture_probs)), x_points)


def get_obs_pdf(obs_points, num_samples, num_clusters_probs, mean_1, std_1, mixture_probs, means_2, stds_2, obs_std):
    traces = generate_traces(num_samples, num_clusters_probs, mean_1, std_1, mixture_probs, means_2, stds_2, obs_std, True)
    return scipy.stats.gaussian_kde([trace[-1] for trace in traces]).evaluate(obs_points)


def heaviside(x):
    return x >= 0


def reparam(u, theta, epsilon=1e-10):
    return torch.log(theta + epsilon) - torch.log(1 - theta + epsilon) + torch.log(u + epsilon) - torch.log(1 - u + epsilon)


def conditional_reparam(v, theta, b, epsilon=1e-10):
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
        torch.log(v / ((1 - v) * (1 - theta)) + 1 + epsilon) * (b == 1).float() +
        (-torch.log(v / ((1 - v) * theta) + 1 + epsilon)) * (b == 0).float()
    )
