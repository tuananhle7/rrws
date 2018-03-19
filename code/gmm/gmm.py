from collections import OrderedDict
from torch.autograd import Variable

import uuid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

iwae_optim = torch.optim.Adam
iwae_optim_params = {'lr': 1e-3}
rws_optim = torch.optim.Adam  # SGD
# rws_optim_params = {'lr': 1e-3, 'nesterov': True, 'momentum': 0.7}
rws_optim_params = {'lr': 1e-3}


class OnlineMeanStd():
    def __init__(self):
        self.count = 0
        self.means = None
        self.M2s = None

    def update(self, new_variables):
        if self.count == 0:
            self.count = 1
            self.means = []
            self.M2s = []
            for new_var in new_variables:
                self.means.append(new_var.data)
                self.M2s.append(new_var.data.new(new_var.size()).fill_(0))
        else:
            self.count = self.count + 1
            for new_var_idx, new_var in enumerate(new_variables):
                delta = new_var.data - self.means[new_var_idx]
                self.means[new_var_idx] = self.means[new_var_idx] + delta / self.count
                delta_2 = new_var.data - self.means[new_var_idx]
                self.M2s[new_var_idx] = self.M2s[new_var_idx] + delta * delta_2

    def means_stds(self):
        if self.count < 2:
            raise ArithmeticError('Need more than 1 value. Have {}'.format(self.count))
        else:
            stds = []
            for i in range(len(self.means)):
                stds.append(torch.sqrt(self.M2s[i] / self.count))
            return self.means, stds

    def avg_of_means_stds(self):
        means, stds = self.means_stds()
        num_parameters = np.sum([len(p) for p in means])
        return (
            np.sum([torch.sum(p) for p in means]) / num_parameters,
            np.sum([torch.sum(p) for p in stds]) / num_parameters
        )


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


def generate_obs(num_samples, generative_network):
    z, x = generative_network.sample(num_samples)
    return x.detach()


class GenerativeNetwork(nn.Module):
    def __init__(self, init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds):
        super(GenerativeNetwork, self).__init__()
        self.num_mixtures = len(init_mixture_probs_pre_softmax)
        self.mixture_probs_pre_softmax = nn.Parameter(torch.Tensor(init_mixture_probs_pre_softmax))
        self.mean_multiplier = Variable(torch.Tensor([init_mean_multiplier]))
        # self.means = Variable(torch.Tensor(means))
        self.log_stds = Variable(torch.Tensor(init_log_stds))

    def get_z_params(self):
        return F.softmax(self.mixture_probs_pre_softmax, dim=0)

    def get_x_params(self):
        return self.mean_multiplier * Variable(torch.arange(self.num_mixtures).type_as(self.mixture_probs_pre_softmax.data)), torch.exp(self.log_stds)

    def sample_z(self, num_samples):
        return torch.multinomial(
            self.get_z_params(),
            num_samples, replacement=True
        ).view(-1)

    def sample_x(self, z):
        num_samples = len(z)
        means, stds = self.get_x_params()
        return means[z] + stds[z] * Variable(means.data.new(num_samples).normal_())

    def sample(self, num_samples):
        z = self.sample_z(num_samples)
        x = self.sample_x(z)
        return z, x

    def z_logpdf(self, z):
        num_samples = len(z)
        return torch.gather(
            torch.log(self.get_z_params()).unsqueeze(0).expand(num_samples, self.num_mixtures),
            1,
            z.long().unsqueeze(-1)
        ).view(-1)

    def x_logpdf(self, x, z):
        means, stds = self.get_x_params()
        return torch.distributions.Normal(
            mean=means[z],
            std=stds[z]
        ).log_prob(x)

    def logpdf(self, z, x):
        return self.z_logpdf(z) + self.x_logpdf(x, z)

    def log_evidence(self, x):
        num_samples = len(x)
        means, stds = self.get_x_params()
        return logsumexp(
            torch.log(
                self.get_z_params().unsqueeze(0).expand(num_samples, - 1)
            ) + torch.distributions.Normal(
                mean=means.unsqueeze(0).expand(num_samples, -1),
                std=stds.unsqueeze(0).expand(num_samples, -1)
            ).log_prob(x.unsqueeze(-1).expand(-1, self.num_mixtures)),
            dim=1
        )


class InferenceNetwork(nn.Module):
    def __init__(self, num_mixtures):
        super(InferenceNetwork, self).__init__()
        self.num_mixtures = num_mixtures
        self.mlp = nn.Sequential(
            # nn.Linear(1, 16),
            # nn.ELU(),  # nn.Tanh(),
            # nn.Linear(16, 32),
            # nn.ELU(),  # nn.Tanh(),
            # nn.Linear(32, 16),
            # nn.ELU(),  # nn.Tanh(),
            # nn.Linear(16, self.num_mixtures),
            # nn.Softmax(dim=1)
            #
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, self.num_mixtures),
            nn.Softmax(dim=1)
        )

    def get_z_params(self, x):
        return self.mlp(x.unsqueeze(-1))

    def sample_z(self, x, relax=False):
        if relax:
            num_samples = len(x)
            z_prob = self.get_z_params(x)
            u_z = Variable(x.data.new(num_samples).uniform_())
            v_z = Variable(x.data.new(num_samples).uniform_())
            aux_z = reparam(u_z, z_prob[:, 1])
            z = heaviside(aux_z).detach()
            aux_z_tilde = conditional_reparam(v_z, z_prob[:, 1], z)
            return z, aux_z, aux_z_tilde
        else:
            return torch.multinomial(self.get_z_params(x), 1, replacement=True).view(-1).detach()

    def sample(self, x, relax=False):
        return self.sample_z(x, relax=relax)

    def z_logpdf(self, z, x):
        return torch.gather(
            torch.log(self.get_z_params(x)),
            1,
            z.long().unsqueeze(-1)
        ).view(-1)

    def logpdf(self, z, x):
        return self.z_logpdf(z, x)


class RelaxControlVariate(nn.Module):
    def __init__(self):
        super(RelaxControlVariate, self).__init__()
        self.control_variate_z = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.ReLU()
        )

    def forward(self, aux_z, aux_z_tilde, x):
        c_z = self.control_variate_z(
            torch.cat([x.unsqueeze(-1), aux_z.unsqueeze(-1)], dim=1)
        ).squeeze(-1)
        c_z_tilde = self.control_variate_z(
            torch.cat([x.unsqueeze(-1), aux_z_tilde.unsqueeze(-1)], dim=1)
        ).squeeze(-1)
        return c_z, c_z_tilde


class IWAE(nn.Module):
    def __init__(self, p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds):
        super(IWAE, self).__init__()
        self.num_mixtures = len(init_log_stds)
        self.generative_network = GenerativeNetwork(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds)
        self.inference_network = InferenceNetwork(self.num_mixtures)

    def elbo_reinforce(self, x, num_particles=1):
        x_expanded = x.unsqueeze(-1).expand(-1, num_particles)
        x_expanded_flattened = x_expanded.contiguous().view(-1)
        z_expanded_flattened = self.inference_network.sample(x_expanded_flattened)
        log_p = self.generative_network.logpdf(z_expanded_flattened, x_expanded_flattened).view(-1, num_particles)
        log_q = self.inference_network.logpdf(z_expanded_flattened, x_expanded_flattened).view(-1, num_particles)
        log_weight = log_p - log_q
        elbo = logsumexp(log_weight, dim=1) - np.log(num_particles)

        return elbo + elbo.detach() * torch.sum(log_q, dim=1), elbo

    def elbo_vimco(self, x, num_particles=2):
        x_expanded = x.unsqueeze(-1).expand(-1, num_particles)
        x_expanded_flattened = x_expanded.contiguous().view(-1)
        z_expanded_flattened = self.inference_network.sample(x_expanded_flattened)
        log_p = self.generative_network.logpdf(z_expanded_flattened, x_expanded_flattened).view(-1, num_particles)
        log_q = self.inference_network.logpdf(z_expanded_flattened, x_expanded_flattened).view(-1, num_particles)
        log_weight = log_p - log_q
        elbo = logsumexp(log_weight, dim=1) - np.log(num_particles)
        elbo_detached = elbo.detach()
        result = elbo
        for i in range(num_particles):
            log_weight_ = log_weight[:, list(set(range(num_particles)).difference(set([i])))]
            control_variate = logsumexp(
                torch.cat([log_weight_, torch.mean(log_weight_, dim=1, keepdim=True)], dim=1),
                dim=1
            )
            result = result + (elbo_detached - control_variate.detach()) * log_q[:, i]
        return result, elbo

    def forward(self, x, gradient_estimator, num_particles=2):
        if gradient_estimator == 'reinforce':
            result, elbo = self.elbo_reinforce(x, num_particles)
            return -torch.mean(result), torch.mean(elbo)
        elif gradient_estimator == 'vimco':
            result, elbo = self.elbo_vimco(x, num_particles)
            return -torch.mean(result), torch.mean(elbo)
        else:
            raise NotImplementedError


def train_iwae(
    p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds,
    true_p_mixture_probs, true_mean_multiplier, true_log_stds, test_x,
    num_iterations, num_samples, num_particles, num_mc_samples,
    gradient_estimator, learning_rate, logging_interval, saving_interval
):
    log_evidence_history = []
    elbo_history = []
    p_mixture_probs_ess_history = []
    p_mixture_probs_norm_history = []
    mean_multiplier_history = []
    p_grad_std_history = []
    q_grad_std_history = []

    reset_seed()
    iwae = IWAE(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds)
    optimizer = iwae_optim(iwae.parameters(), **iwae_optim_params)
    true_generative_network = GenerativeNetwork(np.log(true_p_mixture_probs), true_mean_multiplier, true_log_stds)
    if CUDA:
        iwae.cuda()
        true_generative_network.cuda()

    for i in range(num_iterations):
        x = generate_obs(num_samples, true_generative_network)

        optimizer.zero_grad()
        loss, elbo = iwae(x, gradient_estimator, num_particles)

        loss.backward()
        optimizer.step()

        if i % saving_interval == 0:
            filename = safe_fname('{}_{}'.format(gradient_estimator, i), 'pt')
            torch.save(
                OrderedDict((name, tensor.cpu()) for name, tensor in iwae.state_dict().items()),
                filename
            )
            print('Saved to {}'.format(filename))

        if i % logging_interval == 0:
            elbo_history.append(elbo.data[0])
            log_evidence_history.append(torch.mean(iwae.generative_network.log_evidence(test_x)).data[0])
            p_mixture_probs = iwae.generative_network.get_z_params().data.cpu().numpy()
            p_mixture_probs_ess_history.append(1 / np.sum(p_mixture_probs**2))
            p_mixture_probs_norm_history.append(np.linalg.norm(p_mixture_probs - true_p_mixture_probs))
            mean_multiplier_history.append(iwae.generative_network.mean_multiplier.data[0])

            p_online_mean_std = OnlineMeanStd()
            q_online_mean_std = OnlineMeanStd()
            for mc_sample_idx in range(num_mc_samples):
                iwae.zero_grad()
                loss, _ = iwae(test_x, gradient_estimator, num_particles)
                loss.backward()
                p_online_mean_std.update([p.grad for p in iwae.generative_network.parameters()])
                q_online_mean_std.update([p.grad for p in iwae.inference_network.parameters()])
            p_grad_std_history.append(p_online_mean_std.avg_of_means_stds()[1])
            q_grad_std_history.append(q_online_mean_std.avg_of_means_stds()[1])

            print('Iteration {}: elbo = {}, log_evidence = {}, ess = {}, norm = {}'.format(
                i, elbo_history[-1], log_evidence_history[-1], p_mixture_probs_ess_history[-1], p_mixture_probs_norm_history[-1]
            ))

    return np.array(log_evidence_history), np.array(elbo_history), np.array(p_mixture_probs_ess_history), np.array(p_mixture_probs_norm_history), np.array(mean_multiplier_history), np.array(p_grad_std_history), np.array(q_grad_std_history)


class RWS(nn.Module):
    def __init__(self, p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds):
        super(RWS, self).__init__()
        self.num_mixtures = len(init_log_stds)
        self.generative_network = GenerativeNetwork(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds)
        self.inference_network = InferenceNetwork(self.num_mixtures)

    def wake_theta(self, x, num_particles=1):
        x_expanded = x.unsqueeze(-1).expand(-1, num_particles)
        x_expanded_flattened = x_expanded.contiguous().view(-1)
        z_expanded_flattened = self.inference_network.sample(x_expanded_flattened)
        log_p = self.generative_network.logpdf(z_expanded_flattened, x_expanded_flattened).view(-1, num_particles)
        log_q = self.inference_network.logpdf(z_expanded_flattened, x_expanded_flattened).view(-1, num_particles)
        log_weight = log_p - log_q
        elbo = logsumexp(log_weight, dim=1) - np.log(num_particles)

        return -elbo

    def sleep_phi(self, num_samples):
        z, x = self.generative_network.sample(num_samples)
        return -self.inference_network.logpdf(z.detach(), x.detach())

    def wake_phi(self, x, num_particles=1):
        x_expanded = x.unsqueeze(-1).expand(-1, num_particles)
        x_expanded_flattened = x_expanded.contiguous().view(-1)
        z_expanded_flattened = self.inference_network.sample(x_expanded_flattened)
        log_p = self.generative_network.logpdf(z_expanded_flattened, x_expanded_flattened).view(-1, num_particles)
        log_q = self.inference_network.logpdf(z_expanded_flattened, x_expanded_flattened).view(-1, num_particles)
        log_weight = log_p - log_q
        normalized_weight = torch.exp(lognormexp(log_weight, dim=1))
        return -torch.sum(normalized_weight.detach() * log_q, dim=1)

    def forward(self, mode, x=None, num_particles=None, num_samples=None):
        if mode == 'wake_theta':
            return torch.mean(self.wake_theta(x, num_particles))
        elif mode == 'sleep_phi':
            return torch.mean(self.sleep_phi(num_samples))
        elif mode == 'wake_phi':
            return torch.mean(self.wake_phi(x, num_particles))
        else:
            raise NotImplementedError()


def train_rws(
    p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds,
    true_p_mixture_probs, true_mean_multiplier, true_log_stds, test_x,
    mode,
    num_iterations, num_samples, num_particles, num_mc_samples, learning_rate,
    logging_interval, saving_interval
):
    log_evidence_history = []
    p_mixture_probs_ess_history = []
    p_mixture_probs_norm_history = []
    mean_multiplier_history = []
    p_grad_std_history = []
    q_grad_std_history = []

    reset_seed()
    rws = RWS(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds)
    theta_optimizer = rws_optim(rws.generative_network.parameters(), **rws_optim_params)
    phi_optimizer = rws_optim(rws.inference_network.parameters(), **rws_optim_params)

    true_generative_network = GenerativeNetwork(np.log(true_p_mixture_probs), true_mean_multiplier, true_log_stds)
    if CUDA:
        rws.cuda()
        true_generative_network.cuda()

    for i in range(num_iterations):
        x = generate_obs(num_samples, true_generative_network)

        # Wake theta
        theta_optimizer.zero_grad()
        loss = rws('wake_theta', x=x, num_particles=num_particles)
        loss.backward()
        theta_optimizer.step()

        # Phi update
        if mode == 'ws':
            phi_optimizer.zero_grad()
            loss = rws('sleep_phi', num_samples=num_particles)
            loss.backward()
            phi_optimizer.step()
        elif mode == 'ww':
            phi_optimizer.zero_grad()
            loss = rws('wake_phi', x=x, num_particles=num_particles)
            loss.backward()
            phi_optimizer.step()
        elif mode == 'wsw':
            phi_optimizer.zero_grad()
            loss = 0.5 * rws('sleep_phi', num_samples=num_particles) + 0.5 * rws('wake_phi', x=x, num_particles=num_particles)
            loss.backward()
            phi_optimizer.step()
        elif mode == 'wswa':
            phi_optimizer.zero_grad()
            s_loss = rws('sleep_phi', num_samples=num_particles)
            w_loss = rws('wake_phi', x=x, num_particles=num_particles)
            d = torch.abs(s_loss - w_loss)
            alpha = 1 - d.neg().exp().data[0]
            loss = alpha * w_loss + (1 - alpha) * s_loss
            loss.backward()
            phi_optimizer.step()
        else:
            raise AttributeError('Mode must be one of ws, ww, wsw. Got: {}'.format(mode))

        if i % saving_interval == 0:
            filename = safe_fname('{}_{}'.format(gradient_estimator, i), 'pt')
            torch.save(
                OrderedDict((name, tensor.cpu()) for name, tensor in rws.state_dict().items()),
                filename
            )
            print('Saved to {}'.format(filename))

        if i % logging_interval == 0:
            log_evidence_history.append(torch.mean(rws.generative_network.log_evidence(test_x)).data[0])
            p_mixture_probs = rws.generative_network.get_z_params().data.cpu().numpy()
            p_mixture_probs_ess_history.append(1 / np.sum(p_mixture_probs**2))
            p_mixture_probs_norm_history.append(np.linalg.norm(p_mixture_probs - true_p_mixture_probs))
            mean_multiplier_history.append(rws.generative_network.mean_multiplier.data[0])

            p_online_mean_std = OnlineMeanStd()
            for mc_sample_idx in range(num_mc_samples):
                rws.generative_network.zero_grad()
                loss = rws('wake_theta', x=test_x, num_particles=num_particles)
                loss.backward()
                p_online_mean_std.update([p.grad for p in rws.generative_network.parameters()])
            p_grad_std_history.append(p_online_mean_std.avg_of_means_stds()[1])

            if mode == 'ws':
                q_online_mean_std = OnlineMeanStd()
                for mc_sample_idx in range(num_mc_samples):
                    rws.inference_network.zero_grad()
                    loss = rws('sleep_phi', num_samples=num_particles)
                    loss.backward()
                    q_online_mean_std.update([p.grad for p in rws.inference_network.parameters()])
                q_grad_std_history.append(q_online_mean_std.avg_of_means_stds()[1])
            elif mode == 'ww':
                q_online_mean_std = OnlineMeanStd()
                for mc_sample_idx in range(num_mc_samples):
                    rws.inference_network.zero_grad()
                    loss = rws('wake_phi', x=test_x, num_particles=num_particles)
                    loss.backward()
                    q_online_mean_std.update([p.grad for p in rws.inference_network.parameters()])
                q_grad_std_history.append(q_online_mean_std.avg_of_means_stds()[1])
            elif mode == 'wsw':
                q_online_mean_std = OnlineMeanStd()
                for mc_sample_idx in range(num_mc_samples):
                    rws.inference_network.zero_grad()
                    loss = 0.5 * rws('sleep_phi', num_samples=num_particles) + 0.5 * rws('wake_phi', x=test_x, num_particles=num_particles)
                    loss.backward()
                    q_online_mean_std.update([p.grad for p in rws.inference_network.parameters()])
                q_grad_std_history.append(q_online_mean_std.avg_of_means_stds()[1])
            elif mode == 'wswa':
                q_online_mean_std = OnlineMeanStd()
                for mc_sample_idx in range(num_mc_samples):
                    rws.inference_network.zero_grad()
                    s_loss = rws('sleep_phi', num_samples=num_particles)
                    w_loss = rws('wake_phi', x=x, num_particles=num_particles)
                    d = torch.abs(s_loss - w_loss)
                    alpha = 1 - d.neg().exp().data[0]
                    loss = alpha * w_loss + (1 - alpha) * s_loss
                    loss.backward()
                    q_online_mean_std.update([p.grad for p in rws.inference_network.parameters()])
                q_grad_std_history.append(q_online_mean_std.avg_of_means_stds()[1])
            else:
                raise AttributeError('Mode must be one of ws, ww, wsw. Got: {}'.format(mode))

            print('Iteration {}: log_evidence = {}, ess = {}, norm = {}'.format(
                i, log_evidence_history[-1], p_mixture_probs_ess_history[-1], p_mixture_probs_norm_history[-1]
            ))

    return np.array(log_evidence_history), np.array(p_mixture_probs_ess_history), np.array(p_mixture_probs_norm_history), np.array(mean_multiplier_history), np.array(p_grad_std_history), np.array(q_grad_std_history)


def main(args):
    num_mixtures = 20

    # uniform_p_mixture_probs = np.array([1 / num_mixtures for _ in range(num_mixtures)])
    # temp = np.exp(np.arange(num_mixtures) + 1e-3)
    temp = np.arange(num_mixtures) + 1e-3
    true_p_mixture_probs = temp / np.sum(temp)
    p_init_mixture_probs_pre_softmax = np.random.rand(num_mixtures) * 10

    true_mean_multiplier = 10
    # init_mean_multiplier = 2
    init_mean_multiplier = true_mean_multiplier

    true_log_stds = np.log(np.array([1 for _ in range(num_mixtures)]))
    init_log_stds = true_log_stds  # np.log(np.array([10 for _ in range(num_mixtures)]))
    num_particles = 5
    num_mc_samples = 10
    num_test_samples = 100
    num_samples = 100

    true_generative_network = GenerativeNetwork(np.log(true_p_mixture_probs), true_mean_multiplier, true_log_stds)
    if CUDA:
        true_generative_network.cuda()
    test_x = generate_obs(num_test_samples, true_generative_network)
    true_log_evidence = torch.mean(true_generative_network.log_evidence(test_x)).data[0]

    print('true_log_evidence = {}'.format(true_log_evidence))
    filename = 'true_log_evidence'
    np.save(safe_fname(filename, 'npy'), true_log_evidence)
    print('Saved to {}'.format(filename))

    true_ess = 1 / np.sum(true_p_mixture_probs**2)
    print('true_ess = {}'.format(true_ess))
    filename = 'true_ess'
    np.save(safe_fname(filename, 'npy'), true_ess)
    print('Saved to {}'.format(filename))

    learning_rate = 1e-3
    num_iterations = 30000
    logging_interval = 2000
    saving_interval = 10000

    filename = 'num_mixtures'
    np.save(safe_fname(filename, 'npy'), num_mixtures)
    print('Saved to {}'.format(filename))

    filename = 'logging_interval'
    np.save(safe_fname(filename, 'npy'), logging_interval)
    print('Saved to {}'.format(filename))

    filename = 'num_iterations'
    np.save(safe_fname(filename, 'npy'), num_iterations)
    print('Saved to {}'.format(filename))

    # IWAE

    ## Reinforce
    if args.all or args.reinforce:
        gradient_estimator = 'reinforce'
        iwae_reinforce_log_evidence_history, iwae_reinforce_elbo_history, iwae_reinforce_p_mixture_probs_ess_history, iwae_reinforce_p_mixture_probs_norm_history, iwae_reinforce_mean_multiplier_history, iwae_reinforce_p_grad_std_history, iwae_reinforce_q_grad_std_history = train_iwae(
            p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds,
            true_p_mixture_probs, true_mean_multiplier, true_log_stds, test_x,
            num_iterations, num_samples, num_particles, num_mc_samples,
            gradient_estimator, learning_rate,
            logging_interval, saving_interval
        )
        for [data, filename] in zip(
            [iwae_reinforce_log_evidence_history, iwae_reinforce_elbo_history, iwae_reinforce_p_mixture_probs_ess_history, iwae_reinforce_p_mixture_probs_norm_history, iwae_reinforce_mean_multiplier_history, iwae_reinforce_p_grad_std_history, iwae_reinforce_q_grad_std_history],
            ['iwae_reinforce_log_evidence_history', 'iwae_reinforce_elbo_history', 'iwae_reinforce_p_mixture_probs_ess_history', 'iwae_reinforce_p_mixture_probs_norm_history', 'iwae_reinforce_mean_multiplier_history', 'iwae_reinforce_p_grad_std_history', 'iwae_reinforce_q_grad_std_history']
        ):
            np.save(safe_fname(filename, 'npy'), data)
            print('Saved to {}'.format(filename))

    ## VIMCO
    if args.all or args.vimco:
        gradient_estimator = 'vimco'
        iwae_vimco_log_evidence_history, iwae_vimco_elbo_history, iwae_vimco_p_mixture_probs_ess_history, iwae_vimco_p_mixture_probs_norm_history, iwae_vimco_mean_multiplier_history, iwae_vimco_p_grad_std_history, iwae_vimco_q_grad_std_history = train_iwae(
            p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds,
            true_p_mixture_probs, true_mean_multiplier, true_log_stds, test_x,
            num_iterations, num_samples, num_particles, num_mc_samples,
            gradient_estimator, learning_rate,
            logging_interval, saving_interval
        )
        for [data, filename] in zip(
            [iwae_vimco_log_evidence_history, iwae_vimco_elbo_history, iwae_vimco_p_mixture_probs_ess_history, iwae_vimco_p_mixture_probs_norm_history, iwae_vimco_mean_multiplier_history, iwae_vimco_p_grad_std_history, iwae_vimco_q_grad_std_history],
            ['iwae_vimco_log_evidence_history', 'iwae_vimco_elbo_history', 'iwae_vimco_p_mixture_probs_ess_history', 'iwae_vimco_p_mixture_probs_norm_history', 'iwae_vimco_mean_multiplier_history', 'iwae_vimco_p_grad_std_history', 'iwae_vimco_q_grad_std_history']
        ):
            np.save(safe_fname(filename, 'npy'), data)
            print('Saved to {}'.format(filename))

    ## Relax

    # RWS

    ## WS
    if args.all or args.ws:
        mode = 'ws'
        ws_log_evidence_history, ws_p_mixture_probs_ess_history, ws_p_mixture_probs_norm_history, ws_mean_multiplier_history, ws_p_grad_std_history, ws_q_grad_std_history = train_rws(
            p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds,
            true_p_mixture_probs, true_mean_multiplier, true_log_stds, test_x,
            mode,
            num_iterations, num_samples, num_particles, num_mc_samples, learning_rate,
            logging_interval, saving_interval
        )
        for [data, filename] in zip(
            [ws_log_evidence_history, ws_p_mixture_probs_ess_history, ws_p_mixture_probs_norm_history, ws_mean_multiplier_history, ws_p_grad_std_history, ws_q_grad_std_history],
            ['ws_log_evidence_history', 'ws_p_mixture_probs_ess_history', 'ws_p_mixture_probs_norm_history', 'ws_mean_multiplier_history', 'ws_p_grad_std_history', 'ws_q_grad_std_history']
        ):
            np.save(safe_fname(filename, 'npy'), data)
            print('Saved to {}'.format(filename))

    ## WW
    if args.all or args.ww:
        mode = 'ww'
        ww_log_evidence_history, ww_p_mixture_probs_ess_history, ww_p_mixture_probs_norm_history, ww_mean_multiplier_history, ww_p_grad_std_history, ww_q_grad_std_history = train_rws(
            p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds,
            true_p_mixture_probs, true_mean_multiplier, true_log_stds, test_x,
            mode,
            num_iterations, num_samples, num_particles, num_mc_samples, learning_rate,
            logging_interval, saving_interval
        )
        for [data, filename] in zip(
            [ww_log_evidence_history, ww_p_mixture_probs_ess_history, ww_p_mixture_probs_norm_history, ww_mean_multiplier_history, ww_p_grad_std_history, ww_q_grad_std_history],
            ['ww_log_evidence_history', 'ww_p_mixture_probs_ess_history', 'ww_p_mixture_probs_norm_history', 'ww_mean_multiplier_history', 'ww_p_grad_std_history', 'ww_q_grad_std_history']
        ):
            np.save(safe_fname(filename, 'npy'), data)
            print('Saved to {}'.format(filename))

    ## WSW
    if args.all or args.wsw:
        mode = 'wsw'
        wsw_log_evidence_history, wsw_p_mixture_probs_ess_history, wsw_p_mixture_probs_norm_history, wsw_mean_multiplier_history, wsw_p_grad_std_history, wsw_q_grad_std_history = train_rws(
            p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds,
            true_p_mixture_probs, true_mean_multiplier, true_log_stds, test_x,
            mode,
            num_iterations, num_samples, num_particles, num_mc_samples, learning_rate,
            logging_interval, saving_interval
        )
        for [data, filename] in zip(
            [wsw_log_evidence_history, wsw_p_mixture_probs_ess_history, wsw_p_mixture_probs_norm_history, wsw_mean_multiplier_history, wsw_p_grad_std_history, wsw_q_grad_std_history],
            ['wsw_log_evidence_history', 'wsw_p_mixture_probs_ess_history', 'wsw_p_mixture_probs_norm_history', 'wsw_mean_multiplier_history', 'wsw_p_grad_std_history', 'wsw_q_grad_std_history']
        ):
            np.save(safe_fname(filename, 'npy'), data)
            print('Saved to {}'.format(filename))

    ## WSWA
    if args.all or args.wswa:
        mode = 'wswa'
        wswa_log_evidence_history, wswa_p_mixture_probs_ess_history, wswa_p_mixture_probs_norm_history, wswa_mean_multiplier_history, wswa_p_grad_std_history, wswa_q_grad_std_history = train_rws(
            p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds,
            true_p_mixture_probs, true_mean_multiplier, true_log_stds, test_x,
            mode,
            num_iterations, num_samples, num_particles, num_mc_samples, learning_rate,
            logging_interval, saving_interval
        )
        for [data, filename] in zip(
            [wswa_log_evidence_history, wswa_p_mixture_probs_ess_history, wswa_p_mixture_probs_norm_history, wswa_mean_multiplier_history, wswa_p_grad_std_history, wswa_q_grad_std_history],
            ['wswa_log_evidence_history', 'wswa_p_mixture_probs_ess_history', 'wswa_p_mixture_probs_norm_history', 'wswa_mean_multiplier_history', 'wswa_p_grad_std_history', 'wswa_q_grad_std_history']
        ):
            np.save(safe_fname(filename, 'npy'), data)
            print('Saved to {}'.format(filename))


# globals
CUDA = False
SEED = 1
UID = str(uuid.uuid4())[:8]


def safe_fname(fname, ext):
    return '{}_{:d}_{}.{}'.format(fname, SEED, UID, ext)


def reset_seed():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if CUDA:
        torch.cuda.manual_seed(SEED)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    import argparse

    parser = argparse.ArgumentParser(description='GMM')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA use')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--all', action='store_true', default=False)
    parser.add_argument('--reinforce', action='store_true', default=False)
    parser.add_argument('--vimco', action='store_true', default=False)
    parser.add_argument('--ws', action='store_true', default=False)
    parser.add_argument('--ww', action='store_true', default=False)
    parser.add_argument('--wsw', action='store_true', default=False)
    parser.add_argument('--wswa', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    CUDA = args.cuda
    SEED = args.seed

    reset_seed()

    print('CUDA:', CUDA)
    print('SEED:', SEED)
    print('UID:', UID)
    main(args)
