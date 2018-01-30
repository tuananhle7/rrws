from torch.autograd import Variable
from utils import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GenerativeNetworkL1(nn.Module):
    def __init__(self, train_observation_bias):
        super(GenerativeNetworkL1, self).__init__()
        self.latent_pre_nonlinearity_params = nn.Parameter(torch.zeros(200))
        self.lin1 = nn.Linear(200, 784)
        self.train_observation_bias = train_observation_bias

    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def get_latent_params(self):
        return torch.exp(-softplus(-self.latent_pre_nonlinearity_params))

    def get_latent_logalpha(self):
        return self.latent_pre_nonlinearity_params

    def get_latent_log_density(self, latent):
        latent_logalpha = self.get_latent_logalpha().unsqueeze(0).expand_as(latent)
        return torch.sum(
            bernoulli_logprob((latent + 1) / 2, latent_logalpha),
            dim=1
        )

    def get_observation_params(self, latent):
        return torch.exp(-softplus(-(self.lin1(latent) + self.train_observation_bias)))

    def get_observation_logalpha(self, latent):
        return self.lin1(latent) + self.train_observation_bias

    def get_observation_log_density(self, latent, observation):
        observation_logalpha = self.get_observation_logalpha(latent)
        return torch.sum(
            bernoulli_logprob(observation, observation_logalpha),
            dim=1
        )

    def get_joint_log_density(self, latent, observation):
        return self.get_latent_log_density(latent) + self.get_observation_log_density(latent, observation)

    def sample_latent(self, num_samples):
        latent_params = self.get_latent_params()
        return torch.distributions.Bernoulli(latent_params.unsqueeze(0).expand(num_samples, -1)).sample().float()

    def sample_observation(self, latent):
        observation_params = self.get_observation_params(latent)
        return torch.distributions.Bernoulli(observation_params).sample().float()

    def sample(self, num_samples):
        latent = self.sample_latent(num_samples)
        observation = self.sample_observation(latent)
        return latent, observation

    def elbo(self, observation, inference_network, num_particles=1):
        batch_size = observation.size(0)
        observation_expanded = observation.unsqueeze(1).expand(batch_size, num_particles, 784).contiguous().view(-1, 784)
        latent_expanded = inference_network.sample_latent(observation_expanded)
        log_q_latent_density = inference_network.get_latent_log_density(observation_expanded, latent_expanded)
        log_p_joint_density = self.get_joint_log_density(latent_expanded, observation_expanded)
        log_weights = log_p_joint_density.view(batch_size, num_particles) - log_q_latent_density.view(batch_size, num_particles)
        return logsumexp(log_weights, dim=1) - np.log(num_particles)

    def forward(self, observation, inference_network, num_theta_train_particles):
        elbo_ = self.elbo(observation, inference_network, num_theta_train_particles)
        return -torch.mean(elbo_), elbo_


class InferenceNetworkL1(nn.Module):
    def __init__(self, train_observation_mean):
        super(InferenceNetworkL1, self).__init__()
        self.lin1 = nn.Linear(784, 200)
        self.train_observation_mean = train_observation_mean

    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def get_latent_params(self, observation):
        return torch.exp(-softplus(-self.lin1(observation - self.train_observation_mean)))

    def get_latent_logalpha(self, observation):
        return self.lin1(observation - self.train_observation_mean)

    def get_latent_log_density(self, observation, latent):
        latent_logalpha = self.get_latent_logalpha(observation)
        return torch.sum(
            bernoulli_logprob((latent + 1) / 2, latent_logalpha),
            dim=1
        )

    def sample_latent(self, observation):
        latent_params = self.get_latent_params(observation)
        return torch.distributions.Bernoulli(latent_params).sample().float() * 2 - 1

    def sample_latent_aux(self, observation):
        latent_params = self.get_latent_params(observation)
        u = Variable(torch.rand(200).cuda() if self.is_cuda() else torch.rand(200))
        v = Variable(torch.rand(200).cuda() if self.is_cuda() else torch.rand(200))
        aux_latent = reparam(u, latent_params) # z
        latent = heaviside(aux_latent).float().detach() # b
        aux_latent_tilde = conditional_reparam(v, latent_params, latent) # z_tilde

        return latent * 2 - 1, aux_latent, aux_latent_tilde

    def forward(self, generated_latent, generated_observation):
        return -torch.mean(self.get_latent_log_density(generated_observation, generated_latent))


class RelaxControlVariateL1(nn.Module):
    def __init__(self, train_observation_mean):
        super(RelaxControlVariateL1, self).__init__()
        self.log_temperature = nn.Parameter(torch.log(torch.Tensor([0.5])))
        self.surrogate = nn.Sequential(
            nn.Linear(200 + 784, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )
        self.train_observation_mean = train_observation_mean

    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def forward(self, aux_latent, observation, generative_network, inference_network):
        relaxed_latent = continuous_relaxation(aux_latent, torch.exp(self.log_temperature)) * 2 - 1
        relaxed_elbo = generative_network.get_joint_log_density(relaxed_latent, observation) - \
            inference_network.get_latent_log_density(observation, relaxed_latent)

        return relaxed_elbo + self.surrogate(
            torch.cat([relaxed_latent, (observation - self.train_observation_mean)], dim=1)
        ).squeeze(-1)


class VAEL1(nn.Module):
    def __init__(self, estimator, train_observation_mean, train_observation_bias):
        super(VAEL1, self).__init__()
        self.inference_network = InferenceNetworkL1(train_observation_mean)
        self.generative_network = GenerativeNetworkL1(train_observation_bias)
        self.vae_params = list(self.inference_network.parameters()) + list(self.generative_network.parameters())
        self.estimator = estimator
        if self.estimator == 'relax':
            self.control_variate = RelaxControlVariateL1(train_observation_mean)

    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def forward(self, observation, num_particles=1):
        if num_particles > 1:
            raise NotImplementedError

        if self.estimator == 'reinforce':
            latent = self.inference_network.sample_latent(observation)
            log_q_latent_density = self.inference_network.get_latent_log_density(observation, latent)
            log_p_joint_density = self.generative_network.get_joint_log_density(latent, observation)
            elbo = log_p_joint_density - log_q_latent_density
            return -torch.mean(elbo + elbo.detach() * log_q_latent_density), elbo
        elif self.estimator == 'relax':
            latent, aux_latent, aux_latent_tilde = self.inference_network.sample_latent_aux(observation)
            c = self.control_variate(aux_latent, observation, self.generative_network, self.inference_network)
            c_tilde = self.control_variate(aux_latent_tilde, observation, self.generative_network, self.inference_network)
            log_q_latent_density = self.inference_network.get_latent_log_density(observation, latent)
            log_p_joint_density = self.generative_network.get_joint_log_density(latent, observation)
            elbo = log_p_joint_density - log_q_latent_density
            return -torch.mean(elbo + (elbo - c).detach() * log_q_latent_density + c - c_tilde), elbo
        elif self.estimator == 'nvil':
            raise NotImplementedError
        elif self.estimator == 'vimco':
            raise NotImplementedError
        elif self.estimator == 'rebar':
            raise NotImplementedError
