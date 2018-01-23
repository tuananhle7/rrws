from torch.autograd import Variable

import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import torch
import torch.nn as nn
import torch.distributions
import torch.optim as optim

epsilon = 1e-10


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


def get_posterior_params(obs, prior_mean, prior_std, obs_std):
    posterior_var = 1 / (1 / prior_std**2 + 1 / obs_std**2)
    posterior_std = np.sqrt(posterior_var)
    posterior_mean = posterior_var * (prior_mean / prior_std**2 + obs / obs_std**2)

    return posterior_mean, posterior_std


def get_proposal_params(prior_mean, prior_std, obs_std):
    posterior_var = 1 / (1 / prior_std**2 + 1 / obs_std**2)
    posterior_std = np.sqrt(posterior_var)
    multiplier = posterior_var / obs_std**2
    offset = posterior_var * prior_mean / prior_std**2

    return multiplier, offset, posterior_std


def generate_traces(num_traces, prior_mean, prior_std, obs_std):
    x = np.random.normal(loc=prior_mean, scale=prior_std, size=num_traces)
    obs = np.random.normal(loc=x, scale=obs_std)

    return x, obs


class InferenceNetwork(nn.Module):
    def __init__(self, init_multiplier, init_offset, init_std):
        super(InferenceNetwork, self).__init__()
        self.multiplier = nn.Parameter(torch.Tensor([init_multiplier]))
        self.offset = nn.Parameter(torch.Tensor([init_offset]))
        self.std = nn.Parameter(torch.Tensor([init_std]))

    def forward(self, x, obs):
        return -torch.mean(torch.distributions.Normal(
            mean=self.multiplier.expand_as(obs) * obs + self.offset.expand_as(obs),
            std=self.std.expand_as(obs)
        ).log_prob(x))


class GenerativeNetwork(nn.Module):
    def __init__(self, init_prior_mean, prior_std, obs_std):
        super(GenerativeNetwork, self).__init__()
        self.prior_mean = nn.Parameter(torch.Tensor([init_prior_mean]))
        self.prior_std = prior_std
        self.obs_std = obs_std

    def forward(self, obs, inference_network, num_particles):
        num_traces = len(obs)
        elbos = Variable(torch.zeros(num_traces))

        for o_idx, o in enumerate(obs):
            o_var = Variable(torch.Tensor([o]).float()).expand(num_particles)
            q_x_mean = inference_network.multiplier.expand_as(o_var) * o_var + inference_network.offset.expand_as(o_var)
            q_x_std = inference_network.std.expand_as(o_var)
            q_x_dist = torch.distributions.Normal(mean=q_x_mean.detach(), std=q_x_std.detach())
            x = q_x_dist.sample().detach()

            prior_mean = self.prior_mean.expand_as(o_var)
            prior_std = Variable(torch.Tensor([self.prior_std]).expand_as(o_var))
            obs_std = Variable(torch.Tensor([self.obs_std]).expand_as(o_var))

            log_weights = torch.distributions.Normal(mean=prior_mean, std=prior_std).log_prob(x) + torch.distributions.Normal(mean=x, std=obs_std).log_prob(o_var) - q_x_dist.log_prob(x)

            elbos[o_idx] = logsumexp(log_weights) - np.log(num_particles)

        return -torch.mean(elbos)


def train_cdae(prior_mean, prior_std, obs_std, num_iterations, num_theta_iterations, num_phi_iterations, num_traces, learning_rate, init_prior_mean, init_multiplier, init_offset, init_std, num_particles):
    prior_mean_history = np.zeros([num_iterations])
    multiplier_history = np.zeros([num_iterations])
    offset_history = np.zeros([num_iterations])
    std_history = np.zeros([num_iterations])

    theta_loss_history = np.zeros([num_iterations, num_theta_iterations])
    phi_loss_history = np.zeros([num_iterations, num_phi_iterations])

    generative_network = GenerativeNetwork(init_prior_mean, prior_std, obs_std)
    inference_network = InferenceNetwork(init_multiplier, init_offset, init_std)

    theta_optimizer = optim.Adam(generative_network.parameters(), lr=learning_rate)
    phi_optimizer = optim.Adam(inference_network.parameters(), lr=learning_rate)

    for i in range(num_iterations):
        for theta_idx in range(num_theta_iterations):
            _, obs = generate_traces(num_traces, prior_mean, prior_std, obs_std)
            theta_optimizer.zero_grad()
            loss = generative_network(obs, inference_network, num_particles)
            loss.backward()
            theta_optimizer.step()
            theta_loss_history[i, theta_idx] = loss.data[0]
        for phi_idx in range(num_phi_iterations):
            x, obs = generate_traces(num_traces, generative_network.prior_mean.data[0], prior_std, obs_std)
            x_var = Variable(torch.from_numpy(x).float())
            obs_var = Variable(torch.from_numpy(obs).float())
            phi_optimizer.zero_grad()
            loss = inference_network(x_var, obs_var)
            loss.backward()
            phi_optimizer.step()
            phi_loss_history[i, phi_idx] = loss.data[0]

        prior_mean_history[i] = generative_network.prior_mean.data[0]
        multiplier_history[i] = inference_network.multiplier.data[0]
        offset_history[i] = inference_network.offset.data[0]
        std_history[i] = inference_network.std.data[0]

        if i % 10 == 0:
            print('Iteration {}'.format(i))

    return prior_mean_history, multiplier_history, offset_history, std_history, theta_loss_history, phi_loss_history


def main():
    prior_mean, prior_std, obs_std = 0, 1, 1
    num_iterations = 20000
    num_theta_iterations, num_phi_iterations = 1, 1
    num_traces = 1
    num_particles = 1
    learning_rate = 0.001
    init_prior_mean = 2
    init_multiplier, init_offset, init_std = 2, 2, 2
    true_multiplier, true_offset, true_std = get_proposal_params(prior_mean, prior_std, obs_std)

    prior_mean_history, multiplier_history, offset_history, std_history, theta_loss_history, phi_loss_history = train_cdae(prior_mean, prior_std, obs_std, num_iterations, num_theta_iterations, num_phi_iterations, num_traces, learning_rate, init_prior_mean, init_multiplier, init_offset, init_std, num_particles)

    fig, axs = plt.subplots(4, 1, sharex=True)
    fig.set_size_inches(3, 5)
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axs[0].plot(prior_mean_history, color='black')
    axs[0].axhline(prior_mean, color='black', linestyle='dashed', label='true')
    axs[0].set_ylabel('$\mu_0$')

    axs[1].plot(multiplier_history, color='black')
    axs[1].axhline(true_multiplier, color='black', linestyle='dashed', label='true')
    axs[1].set_ylabel('$a$')

    axs[2].plot(offset_history, color='black')
    axs[2].axhline(true_offset, color='black', linestyle='dashed', label='true')
    axs[2].set_ylabel('$b$')

    axs[3].plot(std_history, color='black')
    axs[3].axhline(true_std, color='black', linestyle='dashed', label='true')
    axs[3].set_ylabel('$c$')

    axs[0].legend()
    axs[-1].set_xlabel('Iteration')

    filename = 'gaussian_cdae_params.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))


if __name__ == '__main__':
    main()
