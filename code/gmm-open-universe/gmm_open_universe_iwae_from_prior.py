from torch.autograd import Variable
from util import *

import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import torch
import torch.nn as nn
import torch.distributions
import torch.optim as optim

epsilon = 1e-10


class GenerativeNetwork(nn.Module):
    def __init__(self, num_clusters_probs, init_mean_1, std_1, mixture_probs, init_means_2, stds_2, obs_std):
        super(GenerativeNetwork, self).__init__()
        self.num_clusters_max = len(num_clusters_probs)
        self.num_mixtures = len(mixture_probs)
        self.num_clusters_probs = num_clusters_probs
        self.mean_1 = nn.Parameter(torch.Tensor([init_mean_1]))
        self.std_1 = std_1
        self.mixture_probs = mixture_probs
        self.means_2 = Variable(torch.Tensor(init_means_2))
        self.stds_2 = stds_2
        self.obs_std = obs_std

    def sort_traces(self, traces):
        short_traces = []
        long_traces = []
        for trace in traces:
            if len(trace) == 2:
                short_traces.append(trace)
            else:
                long_traces.append(trace)

        if len(short_traces) == 0:
            k_short = torch.Tensor([])
            x_short = torch.Tensor([])
        else:
            short_traces_tensor = torch.Tensor(short_traces)
            k_short, x_short = short_traces_tensor.t()

        if len(long_traces) == 0:
            k_long = torch.Tensor([])
            z_long = torch.Tensor([])
            x_long = torch.Tensor([])
        else:
            long_traces_tensor = torch.Tensor(long_traces)
            k_long, z_long, x_long = long_traces_tensor.t()

        return Variable(k_short), Variable(x_short), Variable(k_long), Variable(z_long), Variable(x_long)

    def get_traces(self, num_samples):
        traces = []
        for sample_idx in range(num_samples):
            trace = []
            k = np.random.choice(2, p=self.num_clusters_probs) + 1
            trace.append(k)
            if k == 1:
                x = np.random.normal(self.mean_1.data[0], self.std_1)
                trace.append(x)
            else:
                z = np.random.choice(2, p=self.mixture_probs)
                trace.append(z)
                x = np.random.normal(self.means_2.data[z], self.stds_2[z])
                trace.append(x)
            y = np.random.normal(trace[-1], self.obs_std)
            trace.append(y)
            traces.append(trace)

        return traces

    def forward(self, obs, num_particles, inference_network, traces_from_prior=False):
        num_traces = len(obs)
        elbos = Variable(torch.zeros(num_traces))
        if traces_from_prior:
            for o_idx, o in enumerate(obs):
                traces = self.get_traces(num_particles)
                traces = [trace[:-1] for trace in traces]
                k_short, x_short, k_long, z_long, x_long = self.sort_traces(traces)
                log_weights = Variable(torch.zeros(num_particles))
                if len(k_long) > 0:
                    # long traces
                    obs_long = Variable(torch.Tensor([o])).expand(len(k_long))

                    p_x_long_mean = torch.gather(
                        self.means_2.unsqueeze(0).expand(len(obs_long), self.num_mixtures),
                        1,
                        z_long.long().unsqueeze(-1)
                    ).view(-1)
                    p_x_long_std = torch.gather(
                        Variable(torch.Tensor(self.stds_2).unsqueeze(0).expand(len(obs_long), self.num_mixtures)),
                        1,
                        z_long.long().unsqueeze(-1)
                    ).view(-1)
                    log_p_k_long = torch.gather(
                        Variable(torch.log(torch.Tensor(self.num_clusters_probs)).unsqueeze(0).expand(len(obs_long), self.num_clusters_max)),
                        1,
                        k_long.long().unsqueeze(-1) - 1
                    ).view(-1)
                    log_p_z_long = torch.gather(
                        Variable(torch.log(torch.Tensor(self.mixture_probs)).unsqueeze(0).expand(len(obs_long), self.num_mixtures)),
                        1,
                        z_long.long().unsqueeze(-1)
                    ).view(-1)
                    log_p_x_long = torch.distributions.Normal(
                        mean=p_x_long_mean,
                        std=p_x_long_std
                    ).log_prob(x_long).view(-1)
                    log_p_y_long = torch.distributions.Normal(
                        mean=x_long,
                        std=self.obs_std
                    ).log_prob(obs_long).view(-1)

                    log_weights_long = (log_p_k_long + log_p_z_long + log_p_x_long + log_p_y_long) - (log_p_k_long + log_p_z_long + log_p_x_long).detach()
                    log_weights[:len(k_long)] = log_weights_long
                if len(k_short) > 0:
                    # short traces
                    obs_short = Variable(torch.Tensor([o])).expand(len(k_short))

                    log_p_k_short = torch.gather(
                        Variable(torch.log(torch.Tensor(self.num_clusters_probs)).unsqueeze(0).expand(len(obs_short), self.num_clusters_max)),
                        1,
                        k_short.long().unsqueeze(-1) - 1
                    ).view(-1)
                    log_p_x_short = torch.distributions.Normal(
                        mean=self.mean_1.expand(len(x_short)),
                        std=self.std_1
                    ).log_prob(x_short).view(-1)
                    log_p_y_short = torch.distributions.Normal(
                        mean=x_short,
                        std=self.obs_std
                    ).log_prob(obs_short).view(-1)

                    log_weights_short = (log_p_k_short + log_p_x_short + log_p_y_short) - (log_p_k_short + log_p_x_short).detach()
                    log_weights[len(k_long):] = log_weights_short
                elbos[o_idx] = logsumexp(log_weights) - np.log(num_particles)

            return -torch.mean(elbos)
        else:
            for o_idx, o in enumerate(obs):
                traces = inference_network.get_traces(o, num_particles)
                k_short, x_short, k_long, z_long, x_long = self.sort_traces(traces)
                log_weights = Variable(torch.zeros(num_particles))
                if len(k_long) > 0:
                    # long traces
                    obs_long = Variable(torch.Tensor([o])).expand(len(k_long))

                    q_k_long_prob = inference_network.get_k_params(obs_long)
                    q_z_long_prob = inference_network.get_z_params_from_obs_k(obs_long, k_long)
                    q_x_long_mean, q_x_long_std = inference_network.get_x_params_from_obs_k_z(obs_long, k_long, z_long)
                    log_q_k_long = torch.gather(torch.log(q_k_long_prob + epsilon), 1, k_long.long().unsqueeze(-1) - 1).view(-1)
                    log_q_z_long = torch.gather(torch.log(q_z_long_prob + epsilon), 1, z_long.long().unsqueeze(-1)).view(-1)
                    log_q_x_long = torch.distributions.Normal(q_x_long_mean, q_x_long_std).log_prob(x_long.unsqueeze(-1)).view(-1)

                    p_x_long_mean = torch.gather(
                        self.means_2.unsqueeze(0).expand(len(obs_long), self.num_mixtures),
                        1,
                        z_long.long().unsqueeze(-1)
                    ).view(-1)
                    p_x_long_std = torch.gather(
                        Variable(torch.Tensor(self.stds_2).unsqueeze(0).expand(len(obs_long), self.num_mixtures)),
                        1,
                        z_long.long().unsqueeze(-1)
                    ).view(-1)
                    log_p_k_long = torch.gather(
                        Variable(torch.log(torch.Tensor(self.num_clusters_probs)).unsqueeze(0).expand(len(obs_long), self.num_clusters_max)),
                        1,
                        k_long.long().unsqueeze(-1) - 1
                    ).view(-1)
                    log_p_z_long = torch.gather(
                        Variable(torch.log(torch.Tensor(self.mixture_probs)).unsqueeze(0).expand(len(obs_long), self.num_mixtures)),
                        1,
                        z_long.long().unsqueeze(-1)
                    ).view(-1)
                    log_p_x_long = torch.distributions.Normal(
                        mean=p_x_long_mean,
                        std=p_x_long_std
                    ).log_prob(x_long).view(-1)
                    log_p_y_long = torch.distributions.Normal(
                        mean=x_long,
                        std=self.obs_std
                    ).log_prob(obs_long).view(-1)

                    log_weights_long = (log_p_k_long + log_p_z_long + log_p_x_long + log_p_y_long) - (log_q_k_long + log_q_z_long + log_q_x_long).detach()
                    log_weights[:len(k_long)] = log_weights_long
                if len(k_short) > 0:
                    # short traces
                    obs_short = Variable(torch.Tensor([o])).expand(len(k_short))

                    q_k_short_prob = inference_network.get_k_params(obs_short)
                    q_x_short_mean, q_x_short_std = inference_network.get_x_params_from_obs_k(obs_short, k_short)
                    log_q_k_short = torch.gather(torch.log(q_k_short_prob + epsilon), 1, k_short.long().unsqueeze(-1) - 1).view(-1)
                    log_q_x_short = torch.distributions.Normal(q_x_short_mean, q_x_short_std).log_prob(x_short.unsqueeze(-1)).view(-1)

                    log_p_k_short = torch.gather(
                        Variable(torch.log(torch.Tensor(self.num_clusters_probs)).unsqueeze(0).expand(len(obs_short), self.num_clusters_max)),
                        1,
                        k_short.long().unsqueeze(-1) - 1
                    ).view(-1)
                    log_p_x_short = torch.distributions.Normal(
                        mean=self.mean_1.expand(len(x_short)),
                        std=self.std_1
                    ).log_prob(x_short).view(-1)
                    log_p_y_short = torch.distributions.Normal(
                        mean=x_short,
                        std=self.obs_std
                    ).log_prob(obs_short).view(-1)

                    log_weights_short = (log_p_k_short + log_p_x_short + log_p_y_short) - (log_q_k_short + log_q_x_short).detach()
                    log_weights[len(k_long):] = log_weights_short
                elbos[o_idx] = logsumexp(log_weights) - np.log(num_particles)

            return -torch.mean(elbos)


def train_iwae_from_prior(
    num_clusters_probs, mean_1, std_1, mixture_probs, means_2, stds_2, obs_std,
    num_iterations, num_traces, num_particles,
    learning_rate,
    resume=False, resume_iteration=None, resume_mean_1_history=None,
    resume_loss_history=None, resume_generative_network_state_dict=None,
    resume_optimizer_state_dict=None
):
    mean_1_history = np.zeros([num_iterations])
    loss_history = np.zeros([num_iterations])
    init_mean_1 = np.random.normal(2, 0.1)
    init_means_2 = means_2
    generative_network = GenerativeNetwork(num_clusters_probs, init_mean_1, std_1, mixture_probs, init_means_2, stds_2, obs_std)
    optimizer = optim.Adam(generative_network.parameters(), lr=learning_rate)
    start_iteration = 0
    if resume:
        mean_1_history = resume_mean_1_history
        loss_history = resume_loss_history
        generative_network.load_state_dict(resume_generative_network_state_dict)
        optimizer.load_state_dict(resume_optimizer_state_dict)
        start_iteration = resume_iteration

    for i in range(start_iteration, num_iterations):
        traces = generate_traces(num_traces, num_clusters_probs, mean_1, std_1, mixture_probs, means_2, stds_2, obs_std, generate_obs=True)
        obs = [trace[-1] for trace in traces]
        optimizer.zero_grad()
        loss = generative_network(obs, num_particles, None, True)
        loss.backward()
        optimizer.step()
        loss_history[i] = loss.data[0]
        mean_1_history[i] = generative_network.mean_1.data[0]
        if i % 10 == 0:
            print('iteration {}'.format(i))

        if i % 10 == 0:
            np.save('iwae_from_prior_iteration.npy', i + 1)
            np.save('iwae_from_prior_mean_1_history.npy', mean_1_history)
            np.save('iwae_from_prior_loss_history.npy', loss_history)
            torch.save(generative_network.state_dict(), 'iwae_from_prior_generative_network.pt')
            torch.save(optimizer.state_dict(), 'iwae_from_prior_optimizer.pt')
            print('IWAE_from_prior saved for iteration {}'.format(i))

    np.save('iwae_from_prior_iteration.npy', i)
    np.save('iwae_from_prior_mean_1_history.npy', mean_1_history)
    np.save('iwae_from_prior_loss_history.npy', loss_history)
    torch.save(generative_network.state_dict(), 'iwae_from_prior_generative_network.pt')
    torch.save(optimizer.state_dict(), 'iwae_from_prior_optimizer.pt')
    print('IWAE_from_prior saved for iteration {}'.format(i))

    return loss_history, generative_network, mean_1_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    # True parameters
    num_clusters_probs = [0.5, 0.5]
    mean_1 = 0
    std_1 = 1
    mixture_probs = [0.5, 0.5]
    means_2 = [-5, 5]
    stds_2 = [1, 1]
    obs_std = 1

    # IWAE with inference network from prior
    num_iterations = 10000
    num_traces = 100
    num_particles = 10
    learning_rate = 0.001
    if args.resume:
        filename = 'iwae_from_prior_iteration.npy'
        iwae_from_prior_iteration = int(np.load(filename))
        print('Loaded from {}'.format(filename))

        filename = 'iwae_from_prior_loss_history.npy'
        iwae_from_prior_loss_history = np.load(filename)
        print('Loaded from {}'.format(filename))

        filename = 'iwae_from_prior_mean_1_history.npy'
        iwae_from_prior_mean_1_history = np.load(filename)
        print('Loaded from {}'.format(filename))

        filename = 'iwae_from_prior_generative_network.pt'
        iwae_from_prior_generative_network_state_dict = torch.load(filename)
        print('Loaded from {}'.format(filename))

        filename = 'iwae_from_prior_optimizer.pt'
        iwae_from_prior_optimizer_state_dict = torch.load(filename)
        print('Loaded from {}'.format(filename))

        iwae_from_prior_loss_history, iwae_from_prior_generative_network, iwae_from_prior_mean_1_history = train_iwae_from_prior(
            num_clusters_probs, mean_1, std_1, mixture_probs, means_2, stds_2, obs_std,
            num_iterations, num_traces, learning_rate,
            resume=True,
            resume_iteration=iwae_from_prior_iteration,
            resume_mean_1_history=iwae_from_prior_mean_1_history,
            resume_loss_history=iwae_from_prior_loss_history,
            resume_iwae_from_prior_generative_network_state_dict=iwae_from_prior_generative_network_state_dict,
            resume_optimizer_state_dict=iwae_from_prior_optimizer_state_dict
        )
    else:
        iwae_from_prior_loss_history, iwae_from_prior_generative_network, iwae_from_prior_mean_1_history = train_iwae_from_prior(
            num_clusters_probs, mean_1, std_1, mixture_probs, means_2, stds_2, obs_std,
            num_iterations, num_traces, num_particles,
            learning_rate
        )

    # IWAE From Prior Loss Plot
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(4, 3)
    ax.plot(iwae_from_prior_loss_history, color='black')
    ax.set_xlabel('Iteration')
    ax.set_ylabel("-ELBO")
    ax.set_title('IWAE from Prior Loss')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    filename = 'iwae_from_prior_loss.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))

    # IWAE From Prior Model Params Plot
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(4, 3)
    ax.plot(iwae_from_prior_mean_1_history, color='black')
    ax.axhline(mean_1, color='black', linestyle='dashed', label='true')
    ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('$\mu_{1, 1}$')
    ax.set_title('IWAE From Prior Model Parameter')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    filename = 'iwae_from_prior_model_param.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))


if __name__ == '__main__':
    main()
