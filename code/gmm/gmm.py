from collections import OrderedDict
from torch.autograd import Variable

import sga
import itertools
import util
from util import logsumexp, lognormexp, OnlineMeanStd
import uuid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

iwae_theta_optim = torch.optim.Adam
iwae_phi_optim = torch.optim.Adam
iwae_theta_optim_params = {'lr': 1e-3}
iwae_phi_optim_params = {'lr': 1e-3}
rws_theta_optim = torch.optim.Adam
rws_phi_optim = torch.optim.Adam
rws_theta_optim_params = {'lr': 1e-3}
rws_phi_optim_params = {'lr': 1e-3}
sga_optim = torch.optim.Adam
sga_optim_params = {'lr': 1e-3}
softmax_multiplier = 0.5


class MixtureDistribution:
    def __init__(self, sample_lambdas, logpdf_lambdas, mixture_probs):
        self.sample_lambdas = sample_lambdas
        self.logpdf_lambdas = logpdf_lambdas
        self.mixture_probs = Variable(torch.Tensor(mixture_probs))
        self.num_mixtures = len(mixture_probs)

    def sample(self, num_samples):
        samples = torch.cat(
            [sample_lambda(num_samples).unsqueeze(1) for sample_lambda in self.sample_lambdas],
            dim=1
        )
        indices = torch.multinomial(
            self.mixture_probs,
            num_samples,
            replacement=True
        )
        return torch.gather(samples, dim=1, index=indices.unsqueeze(-1)).squeeze(-1)

    def logpdf(self, values):
        logpdfs = torch.cat(
            [logpdf_lambda(values).unsqueeze(1) for logpdf_lambda in self.logpdf_lambdas],
            dim=1
        )
        return logsumexp(logpdfs + torch.log(self.mixture_probs), dim=1)


def generate_obs(num_samples, generative_network):
    z, x = generative_network.sample(num_samples)
    return x.detach()


class GenerativeNetwork(nn.Module):
    def __init__(self, init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds):
        super(GenerativeNetwork, self).__init__()
        self.num_mixtures = len(init_mixture_probs_pre_softmax)
        self.mixture_probs_pre_softmax = nn.Parameter(torch.Tensor(init_mixture_probs_pre_softmax))
        self.mean_multiplier = Variable(torch.Tensor([init_mean_multiplier]).cuda()) if CUDA else Variable(torch.Tensor([init_mean_multiplier]))
        # self.means = Variable(torch.Tensor(means))
        self.log_stds = Variable(torch.Tensor(init_log_stds).cuda()) if CUDA else Variable(torch.Tensor(init_log_stds))

    def get_z_params(self):
        return F.softmax(self.mixture_probs_pre_softmax * softmax_multiplier, dim=0)

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
            loc=means[z],
            scale=stds[z]
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
                loc=means.unsqueeze(0).expand(num_samples, -1),
                scale=stds.unsqueeze(0).expand(num_samples, -1)
            ).log_prob(x.unsqueeze(-1).expand(-1, self.num_mixtures)),
            dim=1
        )

    def posterior(self, x):
        num_samples = len(x)
        z = Variable(torch.arange(self.num_mixtures).type_as(self.mixture_probs_pre_softmax.data).long())
        log_evidence = self.log_evidence(x)

        z_expanded = z.unsqueeze(0).expand(num_samples, self.num_mixtures)
        x_expanded = x.unsqueeze(-1).expand(num_samples, self.num_mixtures)
        log_evidence_expanded = log_evidence.unsqueeze(-1).expand(num_samples, self.num_mixtures)
        log_joint_expanded = self.logpdf(
            z_expanded.contiguous().view(-1),
            x_expanded.contiguous().view(-1)
        ).view(num_samples, self.num_mixtures)

        return torch.exp(log_joint_expanded - log_evidence_expanded)


class InferenceNetwork(nn.Module):
    def __init__(self, num_mixtures):
        super(InferenceNetwork, self).__init__()
        self.num_mixtures = num_mixtures
        self.mlp = nn.Sequential(
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
    gradient_estimator, logging_interval, saving_interval, seed=1
):
    log_evidence_history = []
    elbo_history = []
    posterior_norm_history = []
    true_posterior_norm_history = []
    p_mixture_probs_norm_history = []
    mean_multiplier_history = []
    p_grad_std_history = []
    q_grad_std_history = []
    q_grad_mean_history = []

    set_seed(seed)
    iwae = IWAE(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds)
    theta_optimizer = iwae_theta_optim(iwae.generative_network.parameters(), **iwae_theta_optim_params)
    phi_optimizer = iwae_phi_optim(iwae.inference_network.parameters(), **iwae_phi_optim_params)
    true_generative_network = GenerativeNetwork(np.log(true_p_mixture_probs) / softmax_multiplier, true_mean_multiplier, true_log_stds)
    if CUDA:
        iwae.cuda()
        true_generative_network.cuda()
    true_posterior = true_generative_network.posterior(test_x)

    for i in range(num_iterations):
        x = generate_obs(num_samples, true_generative_network)

        theta_optimizer.zero_grad()
        phi_optimizer.zero_grad()
        loss, elbo = iwae(x, gradient_estimator, num_particles)
        loss.backward()
        theta_optimizer.step()
        phi_optimizer.step()

        if i % saving_interval == 0:
            filename = safe_fname('{}_{}_{}'.format(gradient_estimator, i, seed), 'pt')
            torch.save(
                OrderedDict((name, tensor.cpu()) for name, tensor in iwae.state_dict().items()),
                filename
            )
            print('Saved to {}'.format(filename))

        if i % logging_interval == 0:
            elbo_history.append(elbo.item())
            log_evidence_history.append(torch.mean(iwae.generative_network.log_evidence(test_x)).item())

            q_posterior = iwae.inference_network.get_z_params(test_x)
            p_posterior = iwae.generative_network.posterior(test_x)
            posterior_norm_history.append(torch.mean(torch.norm(q_posterior - p_posterior, p=2, dim=1)).item())
            true_posterior_norm_history.append(torch.mean(torch.norm(q_posterior - true_posterior, p=2, dim=1)).item())

            p_mixture_probs = iwae.generative_network.get_z_params().data.cpu().numpy()
            p_mixture_probs_norm_history.append(np.linalg.norm(p_mixture_probs - true_p_mixture_probs))
            mean_multiplier_history.append(iwae.generative_network.mean_multiplier.item())

            p_online_mean_std = OnlineMeanStd()
            q_online_mean_std = OnlineMeanStd()
            for mc_sample_idx in range(num_mc_samples):
                iwae.zero_grad()
                loss, _ = iwae(test_x, gradient_estimator, num_particles)
                loss.backward()
                p_online_mean_std.update([p.grad for p in iwae.generative_network.parameters()])
                q_online_mean_std.update([p.grad for p in iwae.inference_network.parameters()])
            p_grad_std_history.append(p_online_mean_std.avg_of_means_stds()[1])

            q_grad_stats = q_online_mean_std.avg_of_means_stds()
            q_grad_std_history.append(q_grad_stats[1])
            q_grad_mean_history.append(q_grad_stats[0])

            print('Iteration {}: elbo = {}, log_evidence = {}, L2(p, true p) = {}, L2(q, current p) = {}'.format(
                i, elbo_history[-1], log_evidence_history[-1], p_mixture_probs_norm_history[-1], posterior_norm_history[-1]
            ))

    return tuple(map(np.array, [
        log_evidence_history, elbo_history, posterior_norm_history, true_posterior_norm_history, p_mixture_probs_norm_history, mean_multiplier_history, p_grad_std_history, q_grad_std_history, q_grad_mean_history
    ]))


class RWS(nn.Module):
    def __init__(self, p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds, true_generative_network=None):
        super(RWS, self).__init__()
        self.num_mixtures = len(init_log_stds)
        self.generative_network = GenerativeNetwork(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds)
        self.inference_network = InferenceNetwork(self.num_mixtures)
        self.true_generative_network = true_generative_network

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

    def wake_phi(self, x, num_particles=1, q_mixture_prob=1):
        num_samples = len(x)
        x_expanded = x.unsqueeze(-1).expand(-1, num_particles)
        x_expanded_flattened = x_expanded.contiguous().view(-1)
        # z_expanded_flattened = self.inference_network.sample(x_expanded_flattened)
        mixture_distribution = MixtureDistribution(
            [
                lambda _: self.inference_network.sample(x_expanded_flattened),
                lambda _: Variable(torch.multinomial(torch.ones(self.num_mixtures), num_particles * num_samples, replacement=True))
            ],
            [
                lambda values: self.inference_network.logpdf(values, x_expanded_flattened),
                lambda values: Variable(-torch.log(torch.Tensor([self.num_mixtures])).expand(num_particles * num_samples))
            ],
            [q_mixture_prob, 1 - q_mixture_prob]
        )
        z_expanded_flattened = mixture_distribution.sample(num_particles * num_samples)
        log_q_mixture = mixture_distribution.logpdf(z_expanded_flattened).view(-1, num_particles)
        log_p = self.generative_network.logpdf(z_expanded_flattened, x_expanded_flattened).view(-1, num_particles)
        log_p_true = 0 if self.true_generative_network is None else self.true_generative_network.log_evidence(x_expanded_flattened).view(-1, num_particles)
        log_q = self.inference_network.logpdf(z_expanded_flattened, x_expanded_flattened).view(-1, num_particles)
        # log_weight = log_p - log_q - log_p_true
        log_weight = log_p - log_q_mixture - log_p_true
        normalized_weight = torch.exp(lognormexp(log_weight, dim=1))
        return -torch.sum(normalized_weight.detach() * log_q, dim=1)

    def forward(self, mode, x=None, num_particles=None, num_samples=None, q_mixture_prob=1):
        if mode == 'wake_theta':
            return torch.mean(self.wake_theta(x, num_particles))
        elif mode == 'sleep_phi':
            return torch.mean(self.sleep_phi(num_samples))
        elif mode == 'wake_phi':
            return torch.mean(self.wake_phi(x, num_particles, q_mixture_prob=q_mixture_prob))
        else:
            raise NotImplementedError()


def train_rws(
    p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds,
    true_p_mixture_probs, true_mean_multiplier, true_log_stds, test_x, mode,
    num_iterations, num_samples, num_particles, num_mc_samples,
    logging_interval, saving_interval, q_mixture_prob=1, seed=1,
    train_sga=False
):
    log_evidence_history = []
    posterior_norm_history = []
    true_posterior_norm_history = []
    p_mixture_probs_norm_history = []
    mean_multiplier_history = []
    p_grad_std_history = []
    q_grad_std_history = []
    q_grad_mean_history = []

    true_generative_network = GenerativeNetwork(np.log(true_p_mixture_probs) / softmax_multiplier, true_mean_multiplier, true_log_stds)
    if CUDA:
        rws.cuda()
        true_generative_network.cuda()
    true_posterior = true_generative_network.posterior(test_x)

    set_seed(seed)
    rws = RWS(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds) # true_generative_network)
    if train_sga:
        num_theta_param_tensors = len(
            list(rws.generative_network.parameters()))
        num_phi_param_tensors = len(list(rws.inference_network.parameters()))
        params = [param for param in rws.generative_network.parameters()] + \
            [param for param in rws.inference_network.parameters()]
        optimizer = sga_optim(params, **sga_optim_params)
    else:
        theta_optimizer = rws_theta_optim(rws.generative_network.parameters(), **rws_theta_optim_params)
        phi_optimizer = rws_phi_optim(rws.inference_network.parameters(), **rws_phi_optim_params)

    for i in range(num_iterations):
        x = generate_obs(num_samples, true_generative_network)

        if train_sga:
            loss_theta = rws('wake_theta', x=x, num_particles=num_particles)
            if mode == 'ws':
                loss_phi = rws('sleep_phi', num_samples=num_samples) # TODO: change to num_particles
            elif mode == 'ww':
                loss_phi = rws('wake_phi', x=x, num_particles=num_particles,
                               q_mixture_prob=q_mixture_prob)

            losses = [loss_theta for _ in range(num_theta_param_tensors)] + \
                [loss_phi for _ in range(num_phi_param_tensors)]
            sga_grads = sga.sga(losses, params)
            sga.optimizer_step(optimizer, params, sga_grads)
        else:
            # Wake theta
            theta_optimizer.zero_grad()
            loss = rws('wake_theta', x=x, num_particles=num_particles)
            loss.backward()
            theta_optimizer.step()

            # Phi update
            if mode == 'ws':
                phi_optimizer.zero_grad()
                loss = rws('sleep_phi', num_samples=num_samples) # TODO: change to num_particles
                loss.backward()
                phi_optimizer.step()
            elif mode == 'ww':
                phi_optimizer.zero_grad()
                loss = rws('wake_phi', x=x, num_particles=num_particles,
                           q_mixture_prob=q_mixture_prob)
                loss.backward()
                phi_optimizer.step()
            elif mode == 'wsw':
                phi_optimizer.zero_grad()
                loss = 0.5 * rws('sleep_phi', num_samples=num_samples) + \
                    0.5 * rws('wake_phi', x=x, num_particles=num_particles,
                              q_mixture_prob=q_mixture_prob)
                loss.backward()
                phi_optimizer.step()
            elif mode == 'wswa':
                phi_optimizer.zero_grad()
                s_loss = rws('sleep_phi', num_samples=num_samples)
                w_loss = rws('wake_phi', x=x, num_particles=num_particles,
                             q_mixture_prob=q_mixture_prob)
                d = torch.abs(s_loss - w_loss)
                alpha = 1 - d.neg().exp().item()
                loss = alpha * w_loss + (1 - alpha) * s_loss
                loss.backward()
                phi_optimizer.step()
            else:
                raise AttributeError(
                    'Mode must be one of ws, ww, wsw. Got: {}'.format(mode))

        if i % saving_interval == 0:
            if mode == 'ww':
                filename = safe_fname('{}{}{}_{}_{}'.format(
                    mode, str(q_mixture_prob).replace('.', '-'),
                    '_sga' if train_sga else '', i, seed), 'pt')
            else:
                filename = safe_fname('{}{}_{}_{}'.format(
                    mode, '_sga' if train_sga else '', i, seed), 'pt')
            torch.save(
                OrderedDict((name, tensor.cpu())
                            for name, tensor in rws.state_dict().items()),
                filename
            )
            print('Saved to {}'.format(filename))

        if i % logging_interval == 0:
            log_evidence_history.append(torch.mean(rws.generative_network.log_evidence(test_x)).item())

            q_posterior = rws.inference_network.get_z_params(test_x)
            p_posterior = rws.generative_network.posterior(test_x)
            posterior_norm_history.append(torch.mean(torch.norm(q_posterior - p_posterior, p=2, dim=1)).item())
            true_posterior_norm_history.append(torch.mean(torch.norm(q_posterior - true_posterior, p=2, dim=1)).item())

            p_mixture_probs = rws.generative_network.get_z_params().data.cpu().numpy()
            p_mixture_probs_norm_history.append(np.linalg.norm(p_mixture_probs - true_p_mixture_probs))
            mean_multiplier_history.append(rws.generative_network.mean_multiplier.item())

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
                    loss = rws('sleep_phi', num_samples=num_samples)
                    loss.backward()
                    q_online_mean_std.update([p.grad for p in rws.inference_network.parameters()])
            elif mode == 'ww':
                q_online_mean_std = OnlineMeanStd()
                for mc_sample_idx in range(num_mc_samples):
                    rws.inference_network.zero_grad()
                    loss = rws('wake_phi', x=test_x, num_particles=num_particles)
                    loss.backward()
                    q_online_mean_std.update([p.grad for p in rws.inference_network.parameters()])
            elif mode == 'wsw':
                q_online_mean_std = OnlineMeanStd()
                for mc_sample_idx in range(num_mc_samples):
                    rws.inference_network.zero_grad()
                    loss = 0.5 * rws('sleep_phi', num_samples=num_samples) + 0.5 * rws('wake_phi', x=test_x, num_particles=num_particles)
                    loss.backward()
                    q_online_mean_std.update([p.grad for p in rws.inference_network.parameters()])
            elif mode == 'wswa':
                q_online_mean_std = OnlineMeanStd()
                for mc_sample_idx in range(num_mc_samples):
                    rws.inference_network.zero_grad()
                    s_loss = rws('sleep_phi', num_samples=num_samples)
                    w_loss = rws('wake_phi', x=x, num_particles=num_particles)
                    d = torch.abs(s_loss - w_loss)
                    alpha = 1 - d.neg().exp().item()
                    loss = alpha * w_loss + (1 - alpha) * s_loss
                    loss.backward()
                    q_online_mean_std.update([p.grad for p in rws.inference_network.parameters()])
            else:
                raise AttributeError('Mode must be one of ws, ww, wsw. Got: {}'.format(mode))

            q_grad_stats = q_online_mean_std.avg_of_means_stds()
            q_grad_std_history.append(q_grad_stats[1])
            q_grad_mean_history.append(q_grad_stats[0])

            print('Iteration {}: phi-loss = {}, log_evidence = {}, L2(p, true p) = {}, L2(q, current p) = {}'.format(
                i, loss.item(), log_evidence_history[-1], p_mixture_probs_norm_history[-1], posterior_norm_history[-1]
            ))

    return tuple(map(np.array, [
        log_evidence_history, posterior_norm_history, true_posterior_norm_history, p_mixture_probs_norm_history, mean_multiplier_history, p_grad_std_history, q_grad_std_history, q_grad_mean_history
    ]))


def main(args):
    num_iterations = args.num_iterations
    logging_interval = args.logging_interval
    saving_interval = args.saving_interval

    num_mixtures = args.num_mixtures

    temp = np.arange(num_mixtures) + 5
    true_p_mixture_probs = temp / np.sum(temp)
    # p_init_mixture_probs_pre_softmax = np.log(np.array(list(reversed(temp)))) # near
    p_init_mixture_probs_pre_softmax = np.array(list(reversed(
        2 * np.arange(num_mixtures))))  # far

    true_mean_multiplier = 10
    init_mean_multiplier = true_mean_multiplier

    true_log_stds = np.log(np.array([5 for _ in range(num_mixtures)]))
    init_log_stds = true_log_stds
    num_particles = args.num_particles
    num_mc_samples = 10
    num_test_samples = 100
    num_samples = 100

    true_generative_network = GenerativeNetwork(
        np.log(true_p_mixture_probs) / softmax_multiplier,
        true_mean_multiplier, true_log_stds)
    if CUDA:
        true_generative_network.cuda()
    test_x = generate_obs(num_test_samples, true_generative_network)
    true_log_evidence = torch.mean(true_generative_network.log_evidence(test_x)).item()

    util.save_np_arrays(
        [num_iterations, logging_interval, saving_interval, num_mixtures,
         num_particles, true_p_mixture_probs, true_mean_multiplier,
         true_log_stds, true_log_evidence],
        [safe_fname(filename, 'npy') for filename in
         ['num_iterations', 'logging_interval', 'saving_interval',
          'num_mixtures', 'num_particles', 'true_p_mixture_probs',
          'true_mean_multiplier', 'true_log_stds', 'true_log_evidence']]
    )
    for filename, data in zip(
        ['num_iterations', 'logging_interval', 'saving_interval', 'num_mixtures', 'num_particles', 'true_p_mixture_probs', 'true_mean_multiplier', 'true_log_stds', 'true_log_evidence'],
        [num_iterations, logging_interval, saving_interval, num_mixtures, num_particles, true_p_mixture_probs, true_mean_multiplier, true_log_stds, true_log_evidence],
    ):
        np.save(safe_fname(filename, 'npy'), data)
        print('Saved to {}'.format(filename))

    for seed in args.seeds:
        # IWAE
        iwae_filenames = [
            'log_evidence_history', 'elbo_history', 'posterior_norm_history',
            'true_posterior_norm_history', 'p_mixture_probs_norm_history',
            'mean_multiplier_history', 'p_grad_std_history',
            'q_grad_std_history', 'q_grad_mean_history']

        ## Reinforce
        if args.all or args.reinforce:
            gradient_estimator = 'reinforce'
            train_stats = list(train_iwae(
                p_init_mixture_probs_pre_softmax, init_mean_multiplier,
                init_log_stds, true_p_mixture_probs, true_mean_multiplier,
                true_log_stds, test_x, num_iterations, num_samples,
                num_particles, num_mc_samples, gradient_estimator,
                logging_interval, saving_interval, seed=seed
            ))
            util.save_np_arrays(
                train_stats,
                [safe_fname('iwae_reinforce_{}_{}'.format(filename, seed),
                            'npy') for filename in iwae_filanemes])

        ## VIMCO
        if args.all or args.vimco:
            gradient_estimator = 'vimco'
            train_stats = list(train_iwae(
                p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds,
                true_p_mixture_probs, true_mean_multiplier, true_log_stds, test_x,
                num_iterations, num_samples, num_particles, num_mc_samples,
                gradient_estimator,
                logging_interval, saving_interval, seed=seed
            ))
            util.save_np_arrays(
                train_stats,
                [safe_fname('iwae_vimco_{}_{}'.format(filename, seed), 'npy')
                 for filename in iwae_filenames])

        # RWS
        sga_prefix = '_sga' if args.sga else ''
        rws_filenames = [
            'log_evidence_history', 'posterior_norm_history',
            'true_posterior_norm_history', 'p_mixture_probs_norm_history',
            'mean_multiplier_history', 'p_grad_std_history',
            'q_grad_std_history', 'q_grad_mean_history']

        ## WS
        if args.all or args.ws:
            mode = 'ws'
            train_stats = list(train_rws(
                p_init_mixture_probs_pre_softmax, init_mean_multiplier,
                init_log_stds, true_p_mixture_probs, true_mean_multiplier,
                true_log_stds, test_x, mode, num_iterations, num_samples,
                num_particles, num_mc_samples, logging_interval,
                saving_interval, seed=seed, train_sga=args.sga
            ))
            util.save_np_arrays(train_stats, [
                safe_fname('{}{}_{}_{}'.format(mode, sga_prefix, filename, seed), 'npy')
                for filename in rws_filenames])

        ## WW
        if args.all or args.ww:
            mode = 'ww'
            for q_mixture_prob in args.ww_probs:
                train_stats = list(train_rws(
                    p_init_mixture_probs_pre_softmax, init_mean_multiplier,
                    init_log_stds, true_p_mixture_probs, true_mean_multiplier,
                    true_log_stds, test_x, mode, num_iterations, num_samples,
                    num_particles, num_mc_samples, logging_interval,
                    saving_interval, q_mixture_prob, seed=seed,
                    train_sga=args.sga
                ))
                util.save_np_arrays(train_stats, [
                    safe_fname('{}{}{}_{}_{}'.format(
                        mode,
                        str(q_mixture_prob).replace('.', '-'),
                        sga_prefix,
                        filename,
                        seed
                    ), 'npy') for filename in rws_filenames
                ])

        ## WSW
        if args.all or args.wsw:
            mode = 'wsw'
            train_stats = list(train_rws(
                p_init_mixture_probs_pre_softmax, init_mean_multiplier,
                init_log_stds, true_p_mixture_probs, true_mean_multiplier,
                true_log_stds, test_x, mode, num_iterations, num_samples,
                num_particles, num_mc_samples, logging_interval,
                saving_interval, seed=seed, train_sga=args.sga
            ))
            util.save_np_arrays(train_stats,
                [safe_fname('{}{}_{}_{}'.format(mode, sga_prefix, filename, seed), 'npy')
                 for filename in rws_filenames])

        ## WSWA
        if args.all or args.wswa:
            mode = 'wswa'
            train_stats = list(train_rws(
                p_init_mixture_probs_pre_softmax, init_mean_multiplier,
                init_log_stds, true_p_mixture_probs, true_mean_multiplier,
                true_log_stds, test_x, mode, num_iterations, num_samples,
                num_particles, num_mc_samples, logging_interval,
                saving_interval, seed=seed, train_sga=args.sga
            ))
            util.save_np_arrays(train_stats, [
                safe_fname('{}{}_{}_{}'.format(mode, sga_prefix, filename, seed), 'npy')
                for filename in rws_filenames])


# globals
CUDA = False
UID = str(uuid.uuid4())[:8]


def safe_fname(fname, ext):
    return '{}/{}_{}.{}'.format(util.WORKING_DIR, fname, UID, ext)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if CUDA:
        torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    import argparse

    parser = argparse.ArgumentParser(description='GMM')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA use')
    parser.add_argument('--num-mixtures', type=int, default=20)
    parser.add_argument('--num-iterations', type=int, default=100000)
    parser.add_argument('--logging-interval', type=int, default=1000)
    parser.add_argument('--saving-interval', type=int, default=1000)
    parser.add_argument('--num-particles', type=int, default=5)
    parser.add_argument('--seeds', nargs='*', type=int, default=[1])
    parser.add_argument('--all', action='store_true', default=False)
    parser.add_argument('--reinforce', action='store_true', default=False)
    parser.add_argument('--vimco', action='store_true', default=False)
    parser.add_argument('--ws', action='store_true', default=False)
    parser.add_argument('--ww', action='store_true', default=False)
    parser.add_argument('--ww-probs', nargs='*', type=float, default=[1.0])
    parser.add_argument('--wsw', action='store_true', default=False)
    parser.add_argument('--wswa', action='store_true', default=False)
    parser.add_argument('--sga', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    CUDA = args.cuda


    print('CUDA:', args.cuda)
    print('SEEDS:', args.seeds)
    print('UID:', UID)
    main(args)
