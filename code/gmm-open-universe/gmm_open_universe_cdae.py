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


def isnan(x):
    if isinstance(x, Variable):
        return np.isnan(torch.sum(x).data[0])
    elif torch.is_tensor(x):
        return np.isnan(torch.sum(x))
    elif isinstance(x, nn.Module):
        for param in x.parameters():
            if isnan(param):
                return True
        return False


class InferenceNetwork(nn.Module):
    def __init__(self, num_clusters_max, num_mixtures):
        super(InferenceNetwork, self).__init__()
        self.num_clusters_max = num_clusters_max
        self.num_mixtures = num_mixtures
        self.obs_to_k_params = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, num_clusters_max),
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
            nn.Linear(8, num_mixtures),
            nn.Softmax(dim=1)
        )
        self.obs_k_z_to_x_mean = nn.Sequential(
            nn.Linear(1 + num_mixtures, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )
        self.obs_k_z_to_x_logstd = nn.Sequential(
            nn.Linear(1 + num_mixtures, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )

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
            if torch.sum(k_prob < 0).data[0] > 0:
                print('k_prob = {}'.format(k_prob))
            if torch.sum(k_prob).data[0] <= 0:
                print('k_prob = {}'.format(k_prob))
            try:
                k = torch.multinomial(k_prob, 1, True).view(-1) + 1
            except:
                print('k_prob = {}'.format(k_prob))
            trace.append(k.data[0])
            if k.data[0] == 1:
                x_mean, x_std = self.get_x_params_from_obs_k(obs_var, k)
                x = torch.distributions.Normal(x_mean, x_std).sample().view(-1)
                trace.append(x.data[0])
            else:
                z_prob = self.get_z_params_from_obs_k(obs_var, k)
                try:
                    z = torch.multinomial(z_prob, 1, True).view(-1)
                except:
                    print('z_prob = {}'.format(z_prob))
                trace.append(z.data[0])
                x_mean, x_std = self.get_x_params_from_obs_k_z(obs_var, k, z)
                x = torch.distributions.Normal(x_mean, x_std).sample().view(-1)
                trace.append(x.data[0])
            traces.append(trace)

        return traces

    def get_pdf(self, x_points, obs, num_samples):
        traces = self.get_traces(obs, num_samples)
        return get_pdf_from_traces(traces, range(1, 1 + self.num_clusters_max), range(self.num_mixtures), x_points)

    def forward(self, traces):
        k_short, x_short, obs_short, k_long, z_long, x_long, obs_long = self.sort_traces(traces)

        k_short_prob = self.get_k_params(obs_short)
        x_short_mean, x_short_std = self.get_x_params_from_obs_k(obs_short, k_short)
        log_q_k_short = torch.gather(torch.log(k_short_prob + epsilon), 1, k_short.long().unsqueeze(-1) - 1)
        log_q_x_short = torch.distributions.Normal(x_short_mean, x_short_std).log_prob(x_short.unsqueeze(-1))

        k_long_prob = self.get_k_params(obs_long)
        z_long_prob = self.get_z_params_from_obs_k(obs_long, k_long)
        x_long_mean, x_long_std = self.get_x_params_from_obs_k_z(obs_long, k_long, z_long)
        log_q_k_long = torch.gather(torch.log(k_long_prob + epsilon), 1, k_long.long().unsqueeze(-1) - 1)
        log_q_z_long = torch.gather(torch.log(z_long_prob + epsilon), 1, z_long.long().unsqueeze(-1))
        log_q_x_long = torch.distributions.Normal(x_long_mean, x_long_std).log_prob(x_long.unsqueeze(-1))
        return -(torch.sum(log_q_k_short + log_q_x_short) + torch.sum(log_q_k_long + log_q_z_long + log_q_x_long)) / len(traces)


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


def train_cdae(
    num_clusters_probs, mean_1, std_1, mixture_probs, means_2, stds_2, obs_std,
    num_iterations, num_theta_iterations, num_phi_iterations, num_traces, num_particles,
    learning_rate, resume=False, resume_iteration=None, resume_theta_loss_history=None, resume_phi_loss_history=None,
    resume_mean_1_history=None, resume_generative_network_state_dict=None,
    resume_inference_network_state_dict=None, resume_theta_optimizer_state_dict=None,
    resume_phi_optimizer_state_dict=None
):
    theta_loss_history = np.zeros([num_iterations, num_theta_iterations])
    phi_loss_history = np.zeros([num_iterations, num_phi_iterations])
    mean_1_history = np.zeros([num_iterations])
    init_mean_1 = np.random.normal(2, 0.1)
    init_means_2 = means_2
    generative_network = GenerativeNetwork(num_clusters_probs, init_mean_1, std_1, mixture_probs, init_means_2, stds_2, obs_std)
    inference_network = InferenceNetwork(len(num_clusters_probs), len(mixture_probs))
    theta_optimizer = optim.Adam(generative_network.parameters(), lr=learning_rate)
    phi_optimizer = optim.Adam(inference_network.parameters(), lr=learning_rate)
    start_iteration = 0
    if resume:
        theta_loss_history = resume_theta_loss_history
        phi_loss_history = resume_phi_loss_history
        mean_1_history = resume_mean_1_history
        generative_network.load_state_dict(resume_generative_network_state_dict)
        inference_network.load_state_dict(resume_inference_network_state_dict)
        theta_optimizer.load_state_dict(resume_theta_optimizer_state_dict)
        phi_optimizer.load_state_dict(resume_phi_optimizer_state_dict)
        start_iteration = resume_iteration

    # print('phi_optimizer.state_dict() = {}'.format(phi_optimizer.state_dict()))
    # print(inference_network.obs_to_k_params.modules())
    # print('obs_to_k_params parameters:')
    # for idx, (name, module) in enumerate(inference_network.obs_to_k_params.named_modules()):
    #     # print(idx, '->', (name, module))
    #     if isinstance(module, nn.Linear):
    #         print('{}\'s weight = {}'.format(name, module.weight.data.numpy()))
    #         print('{}\'s bias = {}'.format(name, module.bias.data.numpy()))
    # input()

    for i in range(start_iteration, num_iterations):
        for theta_idx in range(num_theta_iterations):
            traces = generate_traces(num_traces, num_clusters_probs, mean_1, std_1, mixture_probs, means_2, stds_2, obs_std, generate_obs=True)
            obs = [trace[-1] for trace in traces]
            theta_optimizer.zero_grad()
            loss = generative_network(obs, num_particles, inference_network)
            loss.backward()
            theta_optimizer.step()
            theta_loss_history[i, theta_idx] = loss.data[0]
        for phi_idx in range(num_phi_iterations):
            traces = generative_network.get_traces(num_traces)
            phi_optimizer.zero_grad()
            loss = inference_network(traces)
            loss.backward()
            phi_optimizer.step()
            phi_loss_history[i, phi_idx] = loss.data[0]
        mean_1_history[i] = generative_network.mean_1.data[0]
        if i % 10 == 0:
            print('iteration {}'.format(i))

        if i % 10 == 0:
            np.save('cdae_iteration.npy', i + 1)
            np.save('cdae_theta_loss_history.npy', theta_loss_history)
            np.save('cdae_phi_loss_history.npy', phi_loss_history)
            np.save('cdae_mean_1_history.npy', mean_1_history)
            torch.save(generative_network.state_dict(), 'cdae_generative_network.pt')
            torch.save(inference_network.state_dict(), 'cdae_inference_network.pt')
            torch.save(theta_optimizer.state_dict(), 'cdae_theta_optimizer.pt')
            torch.save(phi_optimizer.state_dict(), 'cdae_phi_optimizer.pt')
            print('CDAE saved for iteration {}'.format(i))

    np.save('cdae_iteration.npy', i + 1)
    np.save('cdae_theta_loss_history.npy', theta_loss_history)
    np.save('cdae_phi_loss_history.npy', phi_loss_history)
    np.save('cdae_mean_1_history.npy', mean_1_history)
    torch.save(generative_network.state_dict(), 'cdae_generative_network.pt')
    torch.save(inference_network.state_dict(), 'cdae_inference_network.pt')
    torch.save(theta_optimizer.state_dict(), 'cdae_theta_optimizer.pt')
    torch.save(phi_optimizer.state_dict(), 'cdae_phi_optimizer.pt')
    print('CDAE saved for iteration {}'.format(i))

    return theta_loss_history, phi_loss_history, generative_network, inference_network, mean_1_history


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

    # CDAE
    num_iterations = 10000
    num_theta_iterations = 1
    num_phi_iterations = 1
    num_traces = 100
    num_particles = 1
    learning_rate = 0.001

    if args.resume:
        filename = 'cdae_iteration.npy'
        cdae_iteration = int(np.load(filename))
        print('Loaded from {}'.format(filename))

        filename = 'cdae_theta_loss_history.npy'
        cdae_theta_loss_history = np.load(filename)
        print('Loaded from {}'.format(filename))

        filename = 'cdae_phi_loss_history.npy'
        cdae_phi_loss_history = np.load(filename)
        print('Loaded from {}'.format(filename))

        filename = 'cdae_mean_1_history.npy'
        cdae_mean_1_history = np.load(filename)
        print('Loaded from {}'.format(filename))

        filename = 'cdae_generative_network.pt'
        cdae_generative_network_state_dict = torch.load(filename)
        print('Loaded from {}'.format(filename))

        filename = 'cdae_inference_network.pt'
        cdae_inference_network_state_dict = torch.load(filename)
        print('Loaded from {}'.format(filename))

        filename = 'cdae_theta_optimizer.pt'
        cdae_theta_optimizer_state_dict = torch.load(filename)
        print('Loaded from {}'.format(filename))

        filename = 'cdae_phi_optimizer.pt'
        cdae_phi_optimizer_state_dict = torch.load(filename)
        print('Loaded from {}'.format(filename))

        cdae_theta_loss_history, cdae_phi_loss_history, cdae_generative_network, cdae_inference_network, cdae_mean_1_history = train_cdae(
            num_clusters_probs, mean_1, std_1, mixture_probs, means_2, stds_2, obs_std,
            num_iterations, num_theta_iterations, num_phi_iterations, num_traces, num_particles,
            learning_rate,
            resume=True,
            resume_iteration=cdae_iteration,
            resume_theta_loss_history=cdae_theta_loss_history,
            resume_phi_loss_history=cdae_phi_loss_history,
            resume_mean_1_history=cdae_mean_1_history,
            resume_generative_network_state_dict=cdae_generative_network_state_dict,
            resume_inference_network_state_dict=cdae_inference_network_state_dict,
            resume_theta_optimizer_state_dict=cdae_theta_optimizer_state_dict,
            resume_phi_optimizer_state_dict=cdae_phi_optimizer_state_dict
        )
    else:
        cdae_theta_loss_history, cdae_phi_loss_history, cdae_generative_network, cdae_inference_network, cdae_mean_1_history = train_cdae(
            num_clusters_probs, mean_1, std_1, mixture_probs, means_2, stds_2, obs_std,
            num_iterations, num_theta_iterations, num_phi_iterations, num_traces, num_particles,
            learning_rate
        )

    # CDAE Loss Plot
    fig, axs = plt.subplots(2, 1)
    fig.set_size_inches(4, 3)

    axs[0].plot(cdae_theta_loss_history.flatten(), color='black')
    axs[0].set_ylabel('$\\theta$ loss')
    axs[1].plot(cdae_phi_loss_history.flatten(), color='black')
    axs[1].set_ylabel('$\phi$ loss')

    axs[0].set_title('CDAE Loss')
    axs[1].set_xlabel('Iteration')
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    filename = 'cdae_loss.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))

    # CDAE Model Parameter Plot
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(4, 3)
    ax.plot(cdae_mean_1_history, color='black')
    ax.axhline(mean_1, color='black', linestyle='dashed', label='true')
    ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('$\mu_{1, 1}$')
    ax.set_title('CDAE Model Parameter')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    filename = 'cdae_model_param.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))

    # CDAE Inference Plot
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
        k_qp_pdf, z_qp_pdf, x_qp_pdf = cdae_inference_network.get_pdf(x_points, test_obs, num_inference_network_samples)

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

    filename = 'cdae_inference.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))


if __name__ == '__main__':
    main()
