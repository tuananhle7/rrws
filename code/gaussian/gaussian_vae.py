import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import torch
import torch.nn as nn
import torch.distributions
from torch.autograd import Variable
import torch.optim as optim


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


class VAE(nn.Module):
    def __init__(self, init_multiplier, init_offset, init_std, init_prior_mean, prior_std, obs_std):
        super(VAE, self).__init__()
        self.multiplier = nn.Parameter(torch.Tensor([init_multiplier]))
        self.offset = nn.Parameter(torch.Tensor([init_offset]))
        self.std = nn.Parameter(torch.Tensor([init_std]))
        self.prior_mean = nn.Parameter(torch.Tensor([init_prior_mean]))
        self.prior_std = prior_std
        self.obs_std = obs_std

    def forward(self, _, obs):
        q_mean = self.multiplier.expand_as(obs) * obs + self.offset.expand_as(obs)
        q_std = self.std.expand_as(obs)
        x = Variable(torch.distributions.Normal(
            mean=torch.zeros(*obs.size()),
            std=torch.ones(*obs.size())
        ).sample()) * q_std + q_mean

        prior_mean = self.prior_mean.expand_as(obs)
        prior_std = Variable(torch.Tensor([self.prior_std]).expand_as(obs))
        obs_std = Variable(torch.Tensor([self.obs_std]).expand_as(obs))

        return -torch.mean(
            torch.distributions.Normal(mean=prior_mean, std=prior_std).log_prob(x) +
            torch.distributions.Normal(mean=x, std=obs_std).log_prob(obs) -
            torch.distributions.Normal(mean=q_mean, std=q_std).log_prob(x)
        )


def train_vae(prior_mean, prior_std, obs_std, num_iterations, num_traces, learning_rate, init_multiplier, init_offset, init_std):
    prior_mean_history = np.zeros([num_iterations])
    multiplier_history = np.zeros([num_iterations])
    offset_history = np.zeros([num_iterations])
    std_history = np.zeros([num_iterations])
    loss_history = np.zeros([num_iterations])

    init_prior_mean = 2
    vae = VAE(init_multiplier, init_offset, init_std, init_prior_mean, prior_std, obs_std)

    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    for i in range(num_iterations):
        x, obs = generate_traces(num_traces, prior_mean, prior_std, obs_std)
        x_var = Variable(torch.from_numpy(x).float())
        obs_var = Variable(torch.from_numpy(obs).float())

        optimizer.zero_grad()
        loss = vae(x_var, obs_var)
        loss.backward()
        optimizer.step()

        prior_mean_history[i] = vae.prior_mean.data[0]
        multiplier_history[i] = vae.multiplier.data[0]
        offset_history[i] = vae.offset.data[0]
        std_history[i] = vae.std.data[0]
        loss_history[i] = loss.data[0]

        if i % 10 == 0:
            print('Iteration {}'.format(i))

    return prior_mean_history, multiplier_history, offset_history, std_history, loss_history


def main():
    prior_mean, prior_std, obs_std = 0, 1, 1
    num_iterations = 20000
    num_traces = 1
    learning_rate = 0.001
    init_multiplier, init_offset, init_std = 2, 2, 2
    true_multiplier, true_offset, true_std = get_proposal_params(prior_mean, prior_std, obs_std)

    prior_mean_history, multiplier_history, offset_history, std_history, loss_history = train_vae(prior_mean, prior_std, obs_std, num_iterations, num_traces, learning_rate, init_multiplier, init_offset, init_std)

    for [data, filename] in zip(
        [prior_mean_history, multiplier_history, offset_history, std_history, loss_history],
        ['prior_mean_history.npy', 'multiplier_history.npy', 'offset_history.npy', 'std_history.npy', 'loss_history.npy']
    ):
        np.save(filename, data)
        print('Saved to {}'.format(filename))


    # Plot stats
    fig, axs = plt.subplots(4, 1, sharex=True)
    fig.set_size_inches(3, 5)
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axs[0].plot(prior_mean_history, color='black')
    axs[0].axhline(prior_mean, color='black', linestyle='dashed', label='true')
    axs[0].set_ylabel('$\mu_0$')

    axs[1].plot(multiplier_history)
    axs[1].axhline(true_multiplier, color='black', linestyle='dashed', label='true')
    axs[1].set_ylabel('$a$')

    axs[2].plot(offset_history)
    axs[2].axhline(true_offset, color='black', linestyle='dashed', label='true')
    axs[2].set_ylabel('$b$')

    axs[3].plot(std_history)
    axs[3].axhline(true_std, color='black', linestyle='dashed', label='true')
    axs[3].set_ylabel('$c$')

    axs[0].legend()
    axs[-1].set_xlabel('Iteration')

    filename = 'gaussian_vae_params.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))


if __name__ == '__main__':
    main()
