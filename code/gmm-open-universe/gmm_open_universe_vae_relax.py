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


def heaviside(x):
    return x >= 0


def reparam(u, theta, epsilon=1e-6):
    return torch.log(theta + epsilon) - torch.log(1 - theta + epsilon) + torch.log(u + epsilon) - torch.log(1 - u + epsilon)


def conditional_reparam(v, theta, b, epsilon=1e-6):
    if b.data[0] == 1:
        return torch.log(v / ((1 - v) * (1 - theta)) + 1 + epsilon)
    else:
        return -torch.log(v / ((1 - v) * theta) + 1 + epsilon)


class VAE(nn.Module):
    def __init__(self, num_clusters_probs, init_mean_1, std_1, mixture_probs, means_2, stds_2, obs_std):
        super(VAE, self).__init__()
        # Generative network parameters
        self.num_clusters_max = len(num_clusters_probs)
        self.num_mixtures = len(mixture_probs)
        self.num_clusters_probs = num_clusters_probs
        self.mean_1 = nn.Parameter(torch.Tensor([init_mean_1]))
        self.std_1 = std_1
        self.mixture_probs = mixture_probs
        self.means_2 = means_2
        self.stds_2 = stds_2
        self.obs_std = obs_std
        self.generative_network_params = [self.mean_1]

        # Inference network parameters
        self.obs_to_k_params = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, self.num_clusters_max),
            nn.Softmax(dim=1)
        )
        self.obs_k_to_x_mean = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )
        self.obs_k_to_x_logstd = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )
        self.obs_k_to_z_params = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, self.num_mixtures),
            nn.Softmax(dim=1)
        )
        self.obs_k_z_to_x_mean = nn.Sequential(
            nn.Linear(1 + self.num_mixtures, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )
        self.obs_k_z_to_x_logstd = nn.Sequential(
            nn.Linear(1 + self.num_mixtures, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )
        self.inference_network_params = (
            list(self.obs_to_k_params.parameters()) +
            list(self.obs_k_to_x_mean.parameters()) +
            list(self.obs_k_to_x_logstd.parameters()) +
            list(self.obs_k_to_z_params.parameters()) +
            list(self.obs_k_z_to_x_mean.parameters()) +
            list(self.obs_k_z_to_x_logstd.parameters())
        )

        # Control variate parameters
        self.control_variate_k = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Tanh()
        )
        self.control_variate_z = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Tanh()
        )
        self.control_variate_params = list(self.control_variate_k.parameters()) + list(self.control_variate_z.parameters())

    def get_k_params(self, obs):
        prob = self.obs_to_k_params(obs.unsqueeze(-1))
        return prob

    def get_x_params_from_obs_k(self, obs, k):
        mean = self.obs_k_to_x_mean(obs.unsqueeze(-1))
        std = torch.exp(self.obs_k_to_x_logstd(obs.unsqueeze(-1)))
        return mean, std

    def get_z_params_from_obs_k(self, obs, k):
        prob = self.obs_k_to_z_params(obs.unsqueeze(-1))
        return prob

    def get_x_params_from_obs_k_z(self, obs, k, z):
        z_one_hot = Variable(torch.zeros(len(z), self.num_mixtures)).scatter_(1, z.long().unsqueeze(-1), 1)
        obs_z = torch.cat([obs.unsqueeze(-1), z_one_hot], dim=1)
        mean = self.obs_k_z_to_x_mean(obs_z)
        std = torch.exp(self.obs_k_z_to_x_logstd(obs_z))
        return mean, std

    def sort_traces(self, traces):
        short_traces = []
        long_traces = []
        for trace in traces:
            if len(trace) == 3:
                short_traces.append(trace)
            else:
                long_traces.append(trace)

        short_traces_tensor = torch.Tensor(short_traces)
        long_traces_tensor = torch.Tensor(long_traces)
        k_short, x_short, obs_short = short_traces_tensor.t()
        k_long, z_long, x_long, obs_long = long_traces_tensor.t()

        return Variable(k_short), Variable(x_short), Variable(obs_short), Variable(k_long), Variable(z_long), Variable(x_long), Variable(obs_long)

    def get_traces(self, obs, num_samples):
        obs_var = Variable(torch.Tensor([obs]))
        traces = []
        for sample_idx in range(num_samples):
            trace = []
            k_prob = self.get_k_params(obs_var)
            k = torch.multinomial(k_prob, 1, True).view(-1) + 1
            trace.append(k.data[0])
            if k.data[0] == 1:
                x_mean, x_std = self.get_x_params_from_obs_k(obs_var, k)
                x = torch.distributions.Normal(x_mean, x_std).sample().view(-1)
                trace.append(x.data[0])
            else:
                z_prob = self.get_z_params_from_obs_k(obs_var, k)
                z = torch.multinomial(z_prob, 1, True).view(-1)
                trace.append(z.data[0])
                x_mean, x_std = self.get_x_params_from_obs_k_z(obs_var, k, z)
                x = torch.distributions.Normal(x_mean, x_std).sample().view(-1)
                trace.append(x.data[0])
            traces.append(trace)

        return traces

    def get_pdf(self, x_points, obs, num_samples):
        traces = self.get_traces(obs, num_samples)
        return get_pdf_from_traces(traces, range(1, 1 + self.num_clusters_max), range(self.num_mixtures), x_points)

    def forward(self, obs):
        num_samples = len(obs)
        k_prob = self.get_k_params(obs)

        u_k = Variable(torch.rand(num_samples))
        v_k = Variable(torch.rand(num_samples))
        aux_k = reparam(u_k, k_prob[:, 1])
        # print('aux_k = {}'.format(aux_k))
        k = heaviside(aux_k).detach() + 1
        # print('k = {}'.format(k))
        aux_k_tilde = conditional_reparam(v_k, k_prob[:, 1], k - 1)
        # print('aux_k_tilde = {}'.format(aux_k_tilde))

        obs_long, obs_short, k_long, k_short = obs[k == 2], obs[k == 1], k[k == 2], k[k == 1]
        aux_k_long, aux_k_short, aux_k_tilde_long, aux_k_tilde_short = aux_k[k == 2], aux_k[k == 1], aux_k_tilde[k == 2], aux_k_tilde[k == 1]
        log_q_k = torch.log(torch.gather(k_prob, 1, k.long().unsqueeze(-1) - 1)).view(-1)
        loss = Variable(torch.Tensor([0]))

        if len(obs_long) > 0:
            # long traces
            z_long_prob = self.get_z_params_from_obs_k(obs_long, k_long)

            u_z_long = Variable(torch.rand(len(obs_long)))
            v_z_long = Variable(torch.rand(len(obs_long)))
            aux_z_long = reparam(u_z_long, z_long_prob[:, 1])
            z_long = heaviside(aux_z_long).detach()
            aux_z_tilde_long = conditional_reparam(v_z_long, z_long_prob[:, 1], z_long)

            x_long_mean, x_long_std = self.get_x_params_from_obs_k_z(obs_long, k_long, z_long)
            x_long = Variable(torch.distributions.Normal(
                mean=torch.zeros(*obs_long.size()),
                std=torch.ones(*obs_long.size())
            ).sample()) * x_long_std.view(-1) + x_long_mean.view(-1)

            prior_x_long_mean = torch.gather(
                Variable(torch.Tensor(self.means_2).unsqueeze(0).expand(len(obs_long), self.num_mixtures)),
                1,
                z_long.long().unsqueeze(-1)
            ).view(-1)
            prior_x_long_std = torch.gather(
                Variable(torch.Tensor(self.stds_2).unsqueeze(0).expand(len(obs_long), self.num_mixtures)),
                1,
                z_long.long().unsqueeze(-1)
            ).view(-1)

            log_prior_k_long = torch.gather(
                Variable(torch.log(torch.Tensor(self.num_clusters_probs)).unsqueeze(0).expand(len(obs_long), self.num_clusters_max)),
                1,
                k_long.long().unsqueeze(-1) - 1
            ).view(-1)
            log_prior_z_long = torch.gather(
                Variable(torch.log(torch.Tensor(self.mixture_probs)).unsqueeze(0).expand(len(obs_long), self.num_mixtures)),
                1,
                z_long.long().unsqueeze(-1)
            ).view(-1)
            log_prior_x_long = torch.distributions.Normal(prior_x_long_mean, prior_x_long_std).log_prob(x_long).view(-1)
            log_lik_long = torch.distributions.Normal(mean=x_long, std=self.obs_std).log_prob(obs_long).view(-1)

            log_q_k_long = log_q_k[k == 2]
            log_q_z_long = torch.log(torch.gather(z_long_prob, 1, z_long.long().unsqueeze(-1))).view(-1)
            log_q_x_long = torch.distributions.Normal(x_long_mean, x_long_std).log_prob(x_long.unsqueeze(-1)).view(-1)

            long_elbo = log_prior_k_long + log_prior_z_long + log_prior_x_long + log_lik_long - log_q_k_long - log_q_z_long - log_q_x_long

            control_variate_k_long = self.control_variate_k(
                torch.cat([obs_long.unsqueeze(-1), aux_k_long.unsqueeze(-1)], dim=1)
            ).squeeze(-1)
            control_variate_k_tilde_long = self.control_variate_k(
                torch.cat([obs_long.unsqueeze(-1), aux_k_tilde_long.unsqueeze(-1)], dim=1)
            ).squeeze(-1)
            control_variate_z_long = self.control_variate_z(
                torch.cat([obs_long.unsqueeze(-1), aux_z_long.unsqueeze(-1)], dim=1)
            ).squeeze(-1)
            control_variate_z_tilde_long = self.control_variate_z(
                torch.cat([obs_long.unsqueeze(-1), aux_z_tilde_long.unsqueeze(-1)], dim=1)
            ).squeeze(-1)

            loss = loss - torch.sum(
                long_elbo +
                (long_elbo - (control_variate_k_tilde_long + control_variate_z_tilde_long)).detach() * (log_q_k_long + log_q_z_long) +
                (control_variate_k_long + control_variate_z_long) -
                (control_variate_k_tilde_long + control_variate_z_tilde_long)
            ) / len(obs)

        if len(obs_short) > 0:
            # short traces
            x_short_mean, x_short_std = self.get_x_params_from_obs_k(obs_short, k_short)
            x_short = Variable(torch.distributions.Normal(
                mean=torch.zeros(*obs_short.size()),
                std=torch.ones(*obs_short.size())
            ).sample()) * x_short_std.view(-1) + x_short_mean.view(-1)

            log_prior_k_short = torch.gather(
                Variable(torch.log(torch.Tensor(self.num_clusters_probs)).unsqueeze(0).expand(len(obs_short), self.num_clusters_max)),
                1,
                k_short.long().unsqueeze(-1) - 1
            ).view(-1)
            log_prior_x_short = torch.distributions.Normal(self.mean_1, self.std_1).log_prob(x_short).view(-1)
            log_lik_short = torch.distributions.Normal(mean=x_short, std=self.obs_std).log_prob(obs_short).view(-1)

            log_q_k_short = log_q_k[k == 1]
            log_q_x_short = torch.distributions.Normal(x_short_mean, x_short_std).log_prob(x_short.unsqueeze(-1)).view(-1)

            short_elbo = log_prior_k_short + log_prior_x_short + log_lik_short - log_q_k_short - log_q_x_short

            control_variate_k_short = self.control_variate_k(
                torch.cat([obs_short.unsqueeze(-1), aux_k_short.unsqueeze(-1)], dim=1)
            ).squeeze(-1)
            control_variate_k_tilde_short = self.control_variate_k(
                torch.cat([obs_short.unsqueeze(-1), aux_k_tilde_short.unsqueeze(-1)], dim=1)
            ).squeeze(-1)
            loss = loss - torch.sum(
                short_elbo +
                (short_elbo - control_variate_k_tilde_short).detach() * log_q_k_short +
                control_variate_k_short -
                control_variate_k_tilde_short
            ) / len(obs)

        return loss


def train_vae(
    num_clusters_probs, mean_1, std_1, mixture_probs, means_2, stds_2, obs_std,
    num_iterations, num_traces, learning_rate,
    resume=False, resume_iteration=None, resume_mean_1_history=None,
    resume_loss_history=None, resume_vae_state_dict=None,
    resume_optimizer_state_dict=None, resume_control_variate_optimizer_state_dict=None
):
    mean_1_history = np.zeros([num_iterations])
    loss_history = np.zeros([num_iterations])
    init_mean_1 = np.random.normal(2, 0.1)
    vae = VAE(num_clusters_probs, init_mean_1, std_1, mixture_probs, means_2, stds_2, obs_std)
    num_parameters = sum([len(p) for p in vae.parameters()])
    optimizer = optim.Adam(vae.generative_network_params + vae.inference_network_params, lr=learning_rate)
    control_variate_optimizer = optim.Adam(vae.control_variate_params, lr=learning_rate)
    start_iteration = 0
    if resume:
        mean_1_history = resume_mean_1_history
        loss_history = resume_loss_history
        vae.load_state_dict(resume_vae_state_dict)
        optimizer.load_state_dict(resume_optimizer_state_dict)
        control_variate_optimizer.load_state_dict(resume_control_variate_optimizer_state_dict)
        start_iteration = resume_iteration

    for i in range(start_iteration, num_iterations):
        traces = generate_traces(num_traces, num_clusters_probs, mean_1, std_1, mixture_probs, means_2, stds_2, obs_std)

        # Optimizer
        optimizer.zero_grad()
        obs = Variable(torch.Tensor([trace[-1] for trace in traces]))
        loss = vae(obs)
        loss.backward(create_graph=True)

        two_loss_grad_detached = [2 * p.grad.detach() / num_parameters for p in vae.generative_network_params] + [p.grad.detach() for p in vae.inference_network_params]

        optimizer.step()

        # Control variate
        control_variate_optimizer.zero_grad()
        loss_grad = [p.grad for p in vae.generative_network_params] + [p.grad for p in vae.inference_network_params]
        torch.autograd.backward(loss_grad, two_loss_grad_detached)
        control_variate_optimizer.step()

        loss_history[i] = loss.data[0]
        mean_1_history[i] = vae.mean_1.data[0]
        if i % 10 == 0:
            print('iteration {}'.format(i))

        if i % 10 == 0:
            np.save('vae_relax_iteration.npy', i + 1)
            np.save('vae_relax_mean_1_history.npy', mean_1_history)
            np.save('vae_relax_loss_history.npy', loss_history)
            torch.save(vae.state_dict(), 'vae.pt')
            torch.save(optimizer.state_dict(), 'vae_optimizer.pt')
            print('VAE_Relax saved for iteration {}'.format(i))

    np.save('vae_relax_iteration.npy', i + 1)
    np.save('vae_relax_mean_1_history.npy', mean_1_history)
    np.save('vae_relax_loss_history.npy', loss_history)
    torch.save(vae.state_dict(), 'vae.pt')
    torch.save(optimizer.state_dict(), 'vae_optimizer.pt')
    print('VAE_Relax saved for iteration {}'.format(i))

    return loss_history, vae, mean_1_history


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

    # VAE
    num_iterations = 10000
    num_traces = 100
    learning_rate = 0.001

    if args.resume:
        filename = 'vae_relax_iteration.npy'
        vae_iteration = int(np.load(filename))
        print('Loaded from {}'.format(filename))

        filename = 'vae_relax_loss_history.npy'
        vae_loss_history = np.load(filename)
        print('Loaded from {}'.format(filename))

        filename = 'vae_relax_mean_1_history.npy'
        vae_mean_1_history = np.load(filename)
        print('Loaded from {}'.format(filename))

        filename = 'vae_relax.pt'
        vae_state_dict = torch.load(filename)
        print('Loaded from {}'.format(filename))

        filename = 'vae_relax_optimizer.pt'
        vae_optimizer_state_dict = torch.load(filename)
        print('Loaded from {}'.format(filename))

        vae_loss_history, vae, vae_mean_1_history = train_vae(
            num_clusters_probs, mean_1, std_1, mixture_probs, means_2, stds_2, obs_std,
            num_iterations, num_traces, learning_rate,
            resume=True,
            resume_iteration=vae_iteration,
            resume_mean_1_history=vae_mean_1_history,
            resume_loss_history=vae_loss_history,
            resume_vae_state_dict=vae_state_dict,
            resume_optimizer_state_dict=vae_optimizer_state_dict
        )
    else:
        vae_loss_history, vae, vae_mean_1_history = train_vae(
            num_clusters_probs, mean_1, std_1, mixture_probs, means_2, stds_2, obs_std,
            num_iterations, num_traces, learning_rate
        )

    # VAE Loss Plot
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(4, 3)
    ax.plot(vae_loss_history, color='black')
    ax.set_xlabel('Iteration')
    ax.set_ylabel("-ELBO")
    ax.set_title('VAE Relax Loss')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    filename = 'vae_relax_loss.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))

    # VAE Model Parameter Plot
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(4, 3)

    ax.plot(vae_mean_1_history, color='black')
    ax.axhline(mean_1, color='black', linestyle='dashed', label='true')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('$\mu_{1, 1}$')

    ax.set_title('VAE Relax Model Parameter')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    filename = 'vae_relax_model_param.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))

    # VAE Inference Plot
    num_test_obs = 5
    test_obs_min = -7
    test_obs_max = 7
    test_obss = np.linspace(test_obs_min, test_obs_max, num=num_test_obs)

    num_prior_samples = 1000
    num_posterior_samples = 10000
    num_inference_network_samples = 10000
    num_points = 100
    min_point = min([-10, test_obs_min])
    max_point = max([10, test_obs_max])

    bar_width = 0.1
    num_barplots = 3

    x_points = np.linspace(min_point, max_point, num_points)
    z_points = np.arange(len(mixture_probs))
    k_points = np.arange(len(num_clusters_probs)) + 1

    fig, axs = plt.subplots(3, num_test_obs, sharey='row')
    fig.set_size_inches(8, 3.25)

    for axs_ in axs:
        for ax in axs_:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    for test_obs_idx, test_obs in enumerate(test_obss):
        k_prior_pdf, z_prior_pdf, x_prior_pdf = get_prior_pdf(x_points, num_prior_samples, num_clusters_probs, mean_1, std_1, mixture_probs, means_2, stds_2, obs_std)
        k_posterior_pdf, z_posterior_pdf, x_posterior_pdf = get_posterior_pdf(x_points, num_posterior_samples, test_obs, num_clusters_probs, mean_1, std_1, mixture_probs, means_2, stds_2, obs_std)
        k_qp_pdf, z_qp_pdf, x_qp_pdf = vae.get_pdf(x_points, test_obs, num_inference_network_samples)

        i = 0
        axs[0][test_obs_idx].bar(k_points + 0.5 * bar_width * (2 * i + 1 - num_barplots), k_prior_pdf, width=bar_width, color='lightgray', edgecolor='lightgray', fill=True, label='prior')

        i = 1
        axs[0][test_obs_idx].bar(k_points + 0.5 * bar_width * (2 * i + 1 - num_barplots), k_posterior_pdf, width=bar_width, color='black', edgecolor='black', fill=True, label='posterior')

        i = 2
        axs[0][test_obs_idx].bar(k_points + 0.5 * bar_width * (2 * i + 1 - num_barplots), k_qp_pdf, width=bar_width, color='black', fill=False, linestyle='dashed', label='inference network')


        axs[0][test_obs_idx].set_xticks(k_points)
        axs[0][test_obs_idx].set_ylim([0, 1])
        axs[0][test_obs_idx].set_yticks([0, 1])
        axs[0][0].set_ylabel('k')

        i = 0
        axs[1][test_obs_idx].bar(z_points + 0.5 * bar_width * (2 * i + 1 - num_barplots), z_prior_pdf, width=bar_width, color='lightgray', edgecolor='lightgray', fill=True, label='prior')

        i = 1
        axs[1][test_obs_idx].bar(z_points + 0.5 * bar_width * (2 * i + 1 - num_barplots), z_posterior_pdf, width=bar_width, color='black', edgecolor='black', fill=True, label='posterior')

        i = 2
        axs[1][test_obs_idx].bar(z_points + 0.5 * bar_width * (2 * i + 1 - num_barplots), z_qp_pdf, width=bar_width, color='black', fill=False, linestyle='dashed', label='inference network')


        axs[1][test_obs_idx].set_xticks(z_points)
        axs[1][test_obs_idx].set_ylim([0, 1])
        axs[1][test_obs_idx].set_yticks([0, 1])
        axs[1][0].set_ylabel('z')

        axs[2][test_obs_idx].plot(x_points, x_prior_pdf, color='lightgray', label='prior')
        axs[2][test_obs_idx].plot(x_points, x_posterior_pdf, color='black', label='posterior')
        axs[2][test_obs_idx].plot(x_points, x_qp_pdf, color='black', linestyle='dashed', label='inference network')
        axs[2][test_obs_idx].scatter(x=test_obs, y=0, color='black', label='test obs', marker='x')

        axs[2][test_obs_idx].set_yticks([])
        axs[2][0].set_ylabel('x')

    axs[-1][test_obs_idx // 2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, fontsize='small')

    fig.tight_layout()

    filename = 'vae_relax_inference.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))


if __name__ == '__main__':
    main()
