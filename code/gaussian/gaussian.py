from pynverse import inversefunc
from torch.autograd import Variable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import scipy.stats

SMALL_SIZE = 7
MEDIUM_SIZE = 9
BIGGER_SIZE = 11

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


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


def generate_data(true_prior_mean, true_prior_std,  true_obs_std, num_samples):
    obs_mean = np.random.normal(loc=true_prior_mean, scale=true_prior_std, size=num_samples)
    return np.random.normal(loc=obs_mean, scale=true_obs_std)


def get_proposal_params(prior_mean, prior_std, obs_std):
    posterior_var = 1 / (1 / prior_std**2 + 1 / obs_std**2)
    posterior_std = np.sqrt(posterior_var)
    multiplier = posterior_var / obs_std**2
    offset = posterior_var * prior_mean / prior_std**2

    return multiplier, offset, posterior_std


class GenerativeNetwork(nn.Module):
    def __init__(self, init_prior_mean, prior_std, init_obs_std):
        super(GenerativeNetwork, self).__init__()
        self.prior_mean = nn.Parameter(torch.Tensor([init_prior_mean]))
        self.prior_std = Variable(torch.Tensor([prior_std]))
        self.obs_std = nn.Parameter(torch.Tensor([init_obs_std]))

    def sample_mean(self, num_samples):
        return Variable(torch.Tensor(num_samples).normal_()) * self.prior_std + self.prior_mean

    def sample_obs(self, mean):
        num_samples = len(mean)
        return Variable(torch.Tensor(num_samples).normal_()) * self.obs_std + mean

    def sample(self, num_samples):
        mean = self.sample_mean(num_samples)
        obs = self.sample_obs(mean)
        return mean, obs

    def prior_logpdf(self, mean):
        num_samples = len(mean)
        return torch.distributions.Normal(
            self.prior_mean.expand(num_samples), self.prior_std.expand(num_samples)
        ).log_prob(mean)

    def obs_logpdf(self, mean, obs):
        num_samples = len(mean)
        return torch.distributions.Normal(
            mean, self.obs_std.expand(num_samples)
        ).log_prob(obs)

    def logpdf(self, mean, obs):
        return self.prior_logpdf(mean) + self.obs_logpdf(mean, obs)


class InferenceNetwork(nn.Module):
    def __init__(self, init_multiplier, init_offset, init_std):
        super(InferenceNetwork, self).__init__()
        self.multiplier = nn.Parameter(torch.Tensor([init_multiplier]))
        self.offset = nn.Parameter(torch.Tensor([init_offset]))
        self.std = nn.Parameter(torch.Tensor([init_std]))

    def get_params(self, obs):
        q_mean = self.multiplier.expand_as(obs) * obs + self.offset.expand_as(obs)
        q_std = self.std.expand_as(obs)
        return q_mean, q_std

    def sample_mean(self, obs):
        num_samples = len(obs)
        q_mean, q_std = self.get_params(obs)
        return Variable(torch.Tensor(num_samples).normal_()) * q_std + q_mean

    def logpdf(self, mean, obs):
        q_mean, q_std = self.get_params(obs)
        return torch.distributions.Normal(q_mean, q_std).log_prob(mean)


class IWAE(nn.Module):
    def __init__(self, init_prior_mean, prior_std, init_obs_std, init_multiplier, init_offset, init_std):
        super(IWAE, self).__init__()
        self.generative_network = GenerativeNetwork(init_prior_mean, prior_std, init_obs_std)
        self.inference_network = InferenceNetwork(init_multiplier, init_offset, init_std)

    def elbo(self, obs, num_particles=1):
        obs_expanded = obs.unsqueeze(-1).expand(-1, num_particles)
        obs_expanded_flattened = obs_expanded.contiguous().view(-1)
        mean_flattened = self.inference_network.sample_mean(obs_expanded_flattened)
        log_weight = (
            self.generative_network.logpdf(mean_flattened, obs_expanded_flattened) -
            self.inference_network.logpdf(mean_flattened, obs_expanded_flattened)
        ).view(-1, num_particles)

        return logsumexp(log_weight, dim=1) - np.log(num_particles)

    def forward(self, obs, num_particles=1):
        return torch.mean(-self.elbo(obs, num_particles))


def train_iwae(
    init_prior_mean, prior_std, init_obs_std, init_multiplier, init_offset, init_std,
    true_prior_mean, true_prior_std,  true_obs_std,
    num_iterations, learning_rate, num_samples, num_particles
):
    prior_mean_history = np.zeros([num_iterations])
    obs_std_history = np.zeros([num_iterations])
    multiplier_history = np.zeros([num_iterations])
    offset_history = np.zeros([num_iterations])
    std_history = np.zeros([num_iterations])
    loss_history = np.zeros([num_iterations])

    iwae = IWAE(init_prior_mean, prior_std, init_obs_std, init_multiplier, init_offset, init_std)
    optimizer = torch.optim.SGD(iwae.parameters(), lr=learning_rate)

    for i in range(num_iterations):
        obs = generate_data(true_prior_mean, true_prior_std, true_obs_std, num_samples)

        optimizer.zero_grad()
        loss = iwae(Variable(torch.from_numpy(obs).float()), num_particles)
        loss.backward()
        optimizer.step()

        prior_mean_history[i] = iwae.generative_network.prior_mean.data[0]
        obs_std_history[i] = iwae.generative_network.obs_std.data[0]
        multiplier_history[i] = iwae.inference_network.multiplier.data[0]
        offset_history[i] = iwae.inference_network.offset.data[0]
        std_history[i] = iwae.inference_network.std.data[0]
        loss_history[i] = loss.data[0]

        if i % 1000 == 0:
            print('Iteration {}'.format(i))

    return prior_mean_history, obs_std_history, multiplier_history, offset_history, std_history, loss_history


class RWS(nn.Module):
    def __init__(
        self, init_prior_mean, prior_std, init_obs_std, init_multiplier, init_offset, init_std,
        init_wake_phi_anneal_factor=None, wake_phi_ess_threshold=None
    ):
        super(RWS, self).__init__()
        self.generative_network = GenerativeNetwork(init_prior_mean, prior_std, init_obs_std)
        self.inference_network = InferenceNetwork(init_multiplier, init_offset, init_std)
        self.anneal_factor = init_wake_phi_anneal_factor
        self.ess_threshold = wake_phi_ess_threshold

    def ess(self, log_weight, anneal_factor, particles_dim=1):
        return torch.exp(
            2 * logsumexp(log_weight * anneal_factor, dim=particles_dim) -
            logsumexp(2 * log_weight * anneal_factor, dim=particles_dim)
        )

    def update_anneal_factor_greedy(self):
        self.anneal_factor = (1 + self.anneal_factor) / 2

    def update_anneal_factor_bs(self, log_weight):
        num_particles = log_weight.size(1)
        func = lambda a: torch.mean(self.ess(log_weight, float(a))).data[0]
        self.anneal_factor = float(inversefunc(func, y_values=self.ess_threshold * num_particles, domain=[0, 1]))

    def wake_theta(self, obs, num_particles=1):
        obs_expanded = obs.unsqueeze(-1).expand(-1, num_particles)
        obs_expanded_flattened = obs_expanded.contiguous().view(-1)
        mean_flattened = self.inference_network.sample_mean(obs_expanded_flattened)
        log_weight = (
            self.generative_network.logpdf(mean_flattened, obs_expanded_flattened) -
            self.inference_network.logpdf(mean_flattened, obs_expanded_flattened)
        ).view(-1, num_particles)

        return -(logsumexp(log_weight, dim=1) - np.log(num_particles))

    def sleep_phi(self, num_samples):
        mean, obs = self.generative_network.sample(num_samples)
        return -self.inference_network.logpdf(mean, obs)

    def wake_phi(self, obs, num_particles=1):
        obs_expanded = obs.unsqueeze(-1).expand(-1, num_particles)
        obs_expanded_flattened = obs_expanded.contiguous().view(-1)
        mean_flattened = self.inference_network.sample_mean(obs_expanded_flattened).detach()
        log_p = self.generative_network.logpdf(mean_flattened, obs_expanded_flattened)
        log_q = self.inference_network.logpdf(mean_flattened, obs_expanded_flattened)

        log_weight = (log_p - log_q).view(-1, num_particles)

        if self.anneal_factor is None:
            normalized_weight = torch.exp(lognormexp(log_weight, dim=1))
        else:
            if torch.mean(self.ess(log_weight, self.anneal_factor)).data[0] < self.ess_threshold * num_particles:
                self.update_anneal_factor_bs(log_weight)
            else:
                self.update_anneal_factor_greedy()
            normalized_weight = torch.exp(lognormexp(log_weight * self.anneal_factor, dim=1))

        return -torch.sum(normalized_weight.detach() * log_q.view(-1, num_particles), dim=1)

    def forward(self, mode, obs=None, num_particles=None, num_samples=None):
        if mode == 'wake_theta':
            return torch.mean(self.wake_theta(obs, num_particles))
        elif mode == 'sleep_phi':
            return torch.mean(self.sleep_phi(num_samples))
        elif mode == 'wake_phi':
            return torch.mean(self.wake_phi(obs, num_particles))
        else:
            raise NotImplementedError()


def train_rws(
    init_prior_mean, prior_std, init_obs_std, init_multiplier, init_offset, init_std,
    true_prior_mean, true_prior_std,  true_obs_std,
    sleep_phi, wake_phi, num_particles,
    theta_learning_rate, phi_learning_rate,
    num_iterations, anneal_wake_phi=False, init_anneal_factor=None, ess_threshold=None
):
    if (not sleep_phi) and (not wake_phi):
        raise AttributeError('Must have at least one of sleep_phi or wake_phi phases')
    if (not wake_phi) and anneal_wake_phi:
        raise AttributeError('Must have wake-phi phase in order to be able to anneal it')

    prior_mean_history = np.zeros([num_iterations])
    obs_std_history = np.zeros([num_iterations])
    multiplier_history = np.zeros([num_iterations])
    offset_history = np.zeros([num_iterations])
    std_history = np.zeros([num_iterations])
    wake_theta_loss_history = np.zeros([num_iterations])
    if sleep_phi:
        sleep_phi_loss_history = np.zeros([num_iterations])
    if wake_phi:
        wake_phi_loss_history = np.zeros([num_iterations])

    if anneal_wake_phi:
        anneal_factor_history = np.zeros([num_iterations])
        rws = RWS(
            init_prior_mean, prior_std, init_obs_std, init_multiplier, init_offset, init_std,
            init_anneal_factor, ess_threshold
        )
    else:
        rws = RWS(init_prior_mean, prior_std, init_obs_std, init_multiplier, init_offset, init_std)

    theta_optimizer = torch.optim.SGD(rws.generative_network.parameters(), lr=theta_learning_rate)
    phi_optimizer = torch.optim.SGD(rws.inference_network.parameters(), lr=phi_learning_rate)

    for i in range(num_iterations):
        obs = Variable(
            torch.from_numpy(generate_data(true_prior_mean, true_prior_std, true_obs_std, num_particles)).float()
        )

        # Wake theta
        theta_optimizer.zero_grad()
        loss = rws('wake_theta', obs=obs, num_particles=num_particles)
        loss.backward()
        theta_optimizer.step()
        wake_theta_loss_history[i] = loss.data[0]

        # Sleep phi
        if sleep_phi:
            phi_optimizer.zero_grad()
            loss = rws('sleep_phi', num_samples=num_particles)
            loss.backward()
            phi_optimizer.step()
            sleep_phi_loss_history[i] = loss.data[0]

        # Wake phi
        if wake_phi:
            phi_optimizer.zero_grad()
            loss = rws('wake_phi', obs=obs, num_particles=num_particles)
            loss.backward()
            phi_optimizer.step()
            wake_phi_loss_history[i] = loss.data[0]

            if anneal_wake_phi:
                anneal_factor_history[i] = rws.anneal_factor

        prior_mean_history[i] = rws.generative_network.prior_mean.data[0]
        obs_std_history[i] = rws.generative_network.obs_std.data[0]
        multiplier_history[i] = rws.inference_network.multiplier.data[0]
        offset_history[i] = rws.inference_network.offset.data[0]
        std_history[i] = rws.inference_network.std.data[0]

        if i % 1000 == 0:
            print('Iteration {}'.format(i))

    result = (prior_mean_history, obs_std_history, multiplier_history, offset_history, std_history, wake_theta_loss_history)

    if sleep_phi:
        result = result + (sleep_phi_loss_history,)
    else:
        result = result + (None,)

    if wake_phi:
        result = result + (wake_phi_loss_history,)
    else:
        result = result + (None,)

    if anneal_wake_phi:
        result = result + (anneal_factor_history,)
    else:
        result = result + (None,)

    return result


def main():
    true_prior_mean = 0
    true_prior_std = 1
    true_obs_std = 1

    init_prior_mean = 10
    prior_std = true_prior_std
    init_obs_std = 0.5
    init_multiplier = 2
    init_offset = 2
    init_std = 2

    num_iterations = 10000
    learning_rate = 0.01

    true_multiplier, true_offset, true_std = get_proposal_params(true_prior_mean, true_prior_std, true_obs_std)

    # IWAE
    iwae_num_samples = 10
    iwae_num_particles = 10

    iwae_prior_mean_history, iwae_obs_std_history, iwae_multiplier_history, iwae_offset_history, iwae_std_history, iwae_loss_history = train_iwae(
        init_prior_mean, prior_std, init_obs_std, init_multiplier, init_offset, init_std,
        true_prior_mean, true_prior_std,  true_obs_std,
        num_iterations, learning_rate, iwae_num_samples, iwae_num_particles
    )
    iwae_true_multiplier_history, iwae_true_offset_history, iwae_true_std_history = get_proposal_params(
        iwae_prior_mean_history, true_prior_std, iwae_obs_std_history
    )

    for [data, filename] in zip(
        [iwae_prior_mean_history, iwae_obs_std_history, iwae_multiplier_history, iwae_offset_history, iwae_std_history, iwae_loss_history, iwae_true_multiplier_history, iwae_true_offset_history, iwae_true_std_history],
        ['iwae_prior_mean_history.npy', 'iwae_obs_std_history.npy', 'iwae_multiplier_history.npy', 'iwae_offset_history.npy', 'iwae_std_history.npy', 'iwae_loss_history.npy', 'iwae_true_multiplier_history.npy', 'iwae_true_offset_history.npy', 'iwae_true_std_history.npy']
    ):
        np.save(filename, data)
        print('Saved to {}'.format(filename))


    # RWS
    rws_num_particles = 10
    theta_learning_rate = learning_rate
    phi_learning_rate = learning_rate

    sleep_phi = True
    wake_phi = False
    ws_prior_mean_history, ws_obs_std_history, ws_multiplier_history, ws_offset_history, ws_std_history, ws_wake_theta_loss_history, ws_sleep_phi_loss_history, _, _ = train_rws(
        init_prior_mean, prior_std, init_obs_std, init_multiplier, init_offset, init_std,
        true_prior_mean, true_prior_std,  true_obs_std,
        sleep_phi, wake_phi, rws_num_particles,
        theta_learning_rate, phi_learning_rate,
        num_iterations
    )
    ws_true_multiplier_history, ws_true_offset_history, ws_true_std_history = get_proposal_params(
        ws_prior_mean_history, true_prior_std, ws_obs_std_history
    )
    for [data, filename] in zip(
        [ws_prior_mean_history, ws_obs_std_history, ws_multiplier_history, ws_offset_history, ws_std_history, ws_wake_theta_loss_history, ws_sleep_phi_loss_history, ws_true_multiplier_history, ws_true_offset_history, ws_true_std_history],
        ['ws_prior_mean_history.npy', 'ws_obs_std_history.npy', 'ws_multiplier_history.npy', 'ws_offset_history.npy', 'ws_std_history.npy', 'ws_wake_theta_loss_history.npy', 'ws_sleep_phi_loss_history.npy', 'ws_true_multiplier_history.npy', 'ws_true_offset_history.npy', 'ws_true_std_history.npy']
    ):
        np.save(filename, data)
        print('Saved to {}'.format(filename))


    sleep_phi = False
    wake_phi = True
    ww_prior_mean_history, ww_obs_std_history, ww_multiplier_history, ww_offset_history, ww_std_history, ww_wake_theta_loss_history, _, ww_wake_phi_loss_history, _ = train_rws(
        init_prior_mean, prior_std, init_obs_std, init_multiplier, init_offset, init_std,
        true_prior_mean, true_prior_std,  true_obs_std,
        sleep_phi, wake_phi, rws_num_particles,
        theta_learning_rate, phi_learning_rate,
        num_iterations
    )
    ww_true_multiplier_history, ww_true_offset_history, ww_true_std_history = get_proposal_params(
        ww_prior_mean_history, true_prior_std, ww_obs_std_history
    )
    for [data, filename] in zip(
        [ww_prior_mean_history, ww_obs_std_history, ww_multiplier_history, ww_offset_history, ww_std_history, ww_wake_theta_loss_history, ww_wake_phi_loss_history, ww_true_multiplier_history, ww_true_offset_history, ww_true_std_history],
        ['ww_prior_mean_history.npy', 'ww_obs_std_history.npy', 'ww_multiplier_history.npy', 'ww_offset_history.npy', 'ww_std_history.npy', 'ww_wake_theta_loss_history.npy', 'ww_wake_phi_loss_history.npy', 'ww_true_multiplier_history.npy', 'ww_true_offset_history.npy', 'ww_true_std_history.npy']
    ):
        np.save(filename, data)
        print('Saved to {}'.format(filename))

    sleep_phi = True
    wake_phi = True
    wsw_prior_mean_history, wsw_obs_std_history, wsw_multiplier_history, wsw_offset_history, wsw_std_history, wsw_wake_theta_loss_history, wsw_sleep_phi_loss_history, wsw_wake_phi_loss_history, _ = train_rws(
        init_prior_mean, prior_std, init_obs_std, init_multiplier, init_offset, init_std,
        true_prior_mean, true_prior_std,  true_obs_std,
        sleep_phi, wake_phi, rws_num_particles,
        theta_learning_rate, phi_learning_rate,
        num_iterations
    )
    wsw_true_multiplier_history, wsw_true_offset_history, wsw_true_std_history = get_proposal_params(
        wsw_prior_mean_history, true_prior_std, wsw_obs_std_history
    )
    for [data, filename] in zip(
        [wsw_prior_mean_history, wsw_obs_std_history, wsw_multiplier_history, wsw_offset_history, wsw_std_history, wsw_wake_theta_loss_history, wsw_sleep_phi_loss_history, wsw_wake_phi_loss_history, wsw_true_multiplier_history, wsw_true_offset_history, wsw_true_std_history],
        ['wsw_prior_mean_history.npy', 'wsw_obs_std_history.npy', 'wsw_multiplier_history.npy', 'wsw_offset_history.npy', 'wsw_std_history.npy', 'wsw_wake_theta_loss_history.npy', 'wsw_sleep_phi_loss_history.npy', 'wsw_wake_phi_loss_history.npy', 'wsw_true_multiplier_history.npy', 'wsw_true_offset_history.npy', 'wsw_true_std_history.npy']
    ):
        np.save(filename, data)
        print('Saved to {}'.format(filename))

    sleep_phi = False
    wake_phi = True
    anneal_wake_phi = True
    init_anneal_factor = 0
    ess_threshold = 0.90

    waw_prior_mean_history, waw_obs_std_history, waw_multiplier_history, waw_offset_history, waw_std_history, waw_wake_theta_loss_history, _, waw_wake_phi_loss_history, waw_anneal_factor_history = train_rws(
        init_prior_mean, prior_std, init_obs_std, init_multiplier, init_offset, init_std,
        true_prior_mean, true_prior_std,  true_obs_std,
        sleep_phi, wake_phi, rws_num_particles,
        theta_learning_rate, phi_learning_rate,
        num_iterations, anneal_wake_phi, init_anneal_factor, ess_threshold
    )
    waw_true_multiplier_history, waw_true_offset_history, waw_true_std_history = get_proposal_params(
        waw_prior_mean_history, true_prior_std, waw_obs_std_history
    )
    for [data, filename] in zip(
        [waw_prior_mean_history, waw_obs_std_history, waw_multiplier_history, waw_offset_history, waw_std_history, waw_wake_theta_loss_history, waw_wake_phi_loss_history, waw_anneal_factor_history, waw_true_multiplier_history, waw_true_offset_history, waw_true_std_history],
        ['waw_prior_mean_history.npy', 'waw_obs_std_history.npy', 'waw_multiplier_history.npy', 'waw_offset_history.npy', 'waw_std_history.npy', 'waw_wake_theta_loss_history.npy', 'waw_wake_phi_loss_history.npy', 'waw_anneal_factor_history.npy', 'waw_true_multiplier_history.npy', 'waw_true_offset_history.npy', 'waw_true_std_history.npy']
    ):
        np.save(filename, data)
        print('Saved to {}'.format(filename))

    sleep_phi = True
    wake_phi = True
    anneal_wake_phi = True
    init_anneal_factor = 0
    ess_threshold = 0.90

    wsaw_prior_mean_history, wsaw_obs_std_history, wsaw_multiplier_history, wsaw_offset_history, wsaw_std_history, wsaw_wake_theta_loss_history, wsaw_sleep_phi_loss_history, wsaw_wake_phi_loss_history, wsaw_anneal_factor_history = train_rws(
        init_prior_mean, prior_std, init_obs_std, init_multiplier, init_offset, init_std,
        true_prior_mean, true_prior_std,  true_obs_std,
        sleep_phi, wake_phi, rws_num_particles,
        theta_learning_rate, phi_learning_rate,
        num_iterations, anneal_wake_phi, init_anneal_factor, ess_threshold
    )
    wsaw_true_multiplier_history, wsaw_true_offset_history, wsaw_true_std_history = get_proposal_params(
        wsaw_prior_mean_history, true_prior_std, wsaw_obs_std_history
    )
    for [data, filename] in zip(
        [wsaw_prior_mean_history, wsaw_obs_std_history, wsaw_multiplier_history, wsaw_offset_history, wsaw_std_history, wsaw_wake_theta_loss_history, wsaw_sleep_phi_loss_history, wsaw_wake_phi_loss_history, wsaw_anneal_factor_history, wsaw_true_multiplier_history, wsaw_true_offset_history, wsaw_true_std_history],
        ['wsaw_prior_mean_history.npy', 'wsaw_obs_std_history.npy', 'wsaw_multiplier_history.npy', 'wsaw_offset_history.npy', 'wsaw_std_history.npy', 'wsaw_wake_theta_loss_history.npy', 'wsaw_sleep_phi_loss_history.npy', 'wsaw_wake_phi_loss_history.npy', 'wsaw_anneal_factor_history.npy', 'wsaw_true_multiplier_history.npy', 'wsaw_true_offset_history.npy', 'wsaw_true_std_history.npy']
    ):
        np.save(filename, data)
        print('Saved to {}'.format(filename))

    # Plotting
    fig, axs = plt.subplots(5, 1, sharex=True)

    fig.set_size_inches(3.25, 4)

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # axs[0].set_title('Generative network parameters')
    axs[0].plot(iwae_prior_mean_history, color='black', linestyle=':', label='iwae-{}'.format(iwae_num_particles))
    axs[0].plot(ws_prior_mean_history, color='black', linestyle='-.', label='ws-{}'.format(rws_num_particles))
    axs[0].plot(ww_prior_mean_history, color='0.8', linestyle='-', label='ww-{}'.format(rws_num_particles))
    axs[0].plot(wsw_prior_mean_history, color='0.4', linestyle='-', label='wsw-{}'.format(rws_num_particles))
    axs[0].plot(waw_prior_mean_history, color='0.8', linestyle='--', label='waw-{}'.format(rws_num_particles))
    axs[0].plot(wsaw_prior_mean_history, color='0.4', linestyle='--', label='wsaw-{}'.format(rws_num_particles))
    axs[0].axhline(true_prior_mean, color='black', label='true')
    axs[0].set_ylabel('$\mu_0$')

    axs[1].plot(iwae_obs_std_history, color='black', linestyle=':', label='iwae-{}'.format(iwae_num_particles))
    axs[1].plot(ws_obs_std_history, color='black', linestyle='-.', label='ws-{}'.format(rws_num_particles))
    axs[1].plot(ww_obs_std_history, color='0.8', linestyle='-', label='ww-{}'.format(rws_num_particles))
    axs[1].plot(wsw_obs_std_history, color='0.4', linestyle='-', label='wsw-{}'.format(rws_num_particles))
    axs[1].plot(waw_obs_std_history, color='0.8', linestyle='--', label='waw-{}'.format(rws_num_particles))
    axs[1].plot(wsaw_obs_std_history, color='0.4', linestyle='--', label='wsaw-{}'.format(rws_num_particles))
    axs[1].axhline(true_obs_std, color='black', label='true')
    axs[1].set_ylabel('$\sigma$')
    # axs[1].set_xlabel('Iteration')

    # axs[2].set_title('Inference network parameters')
    axs[2].plot(iwae_multiplier_history, color='black', linestyle=':', label='iwae-{}'.format(iwae_num_particles))
    axs[2].plot(ws_multiplier_history, color='black', linestyle='-.', label='ws-{}'.format(rws_num_particles))
    axs[2].plot(ww_multiplier_history, color='0.8', linestyle='-', label='ww-{}'.format(rws_num_particles))
    axs[2].plot(wsw_multiplier_history, color='0.4', linestyle='-', label='wsw-{}'.format(rws_num_particles))
    axs[2].plot(waw_multiplier_history, color='0.8', linestyle='--', label='waw-{}'.format(rws_num_particles))
    axs[2].plot(wsaw_multiplier_history, color='0.4', linestyle='--', label='wsaw-{}'.format(rws_num_particles))
    axs[2].axhline(true_multiplier, color='black', label='true')
    axs[2].set_ylabel('$a$')

    axs[3].plot(iwae_offset_history, color='black', linestyle=':', label='iwae-{}'.format(iwae_num_particles))
    axs[3].plot(ws_offset_history, color='black', linestyle='-.', label='ws-{}'.format(rws_num_particles))
    axs[3].plot(ww_offset_history, color='0.8', linestyle='-', label='ww-{}'.format(rws_num_particles))
    axs[3].plot(wsw_offset_history, color='0.4', linestyle='-', label='wsw-{}'.format(rws_num_particles))
    axs[3].plot(waw_offset_history, color='0.8', linestyle='--', label='waw-{}'.format(rws_num_particles))
    axs[3].plot(wsaw_offset_history, color='0.4', linestyle='--', label='wsaw-{}'.format(rws_num_particles))
    axs[3].axhline(true_offset, color='black', label='true')
    axs[3].set_ylabel('$b$')

    axs[4].plot(iwae_std_history, color='black', linestyle=':', label='iwae-{}'.format(iwae_num_particles))
    axs[4].plot(ws_std_history, color='black', linestyle='-.', label='ws-{}'.format(rws_num_particles))
    axs[4].plot(ww_std_history, color='0.8', linestyle='-', label='ww-{}'.format(rws_num_particles))
    axs[4].plot(wsw_std_history, color='0.4', linestyle='-', label='wsw-{}'.format(rws_num_particles))
    axs[4].plot(waw_std_history, color='0.8', linestyle='--', label='waw-{}'.format(rws_num_particles))
    axs[4].plot(wsaw_std_history, color='0.4', linestyle='--', label='wsaw-{}'.format(rws_num_particles))
    axs[4].axhline(true_std, color='black', label='true')
    axs[4].set_ylabel('$c$')
    axs[4].set_xlabel('Iteration')

    axs[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.6), ncol=4)
    fig.tight_layout()

    filename = 'gaussian.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))


if __name__ == '__main__':
    main()
