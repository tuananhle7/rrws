from torch.autograd import Variable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

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


class GenerativeNetwork(nn.Module):
    def __init__(self, init_mixture_probs, means, stds):
        super(GenerativeNetwork, self).__init__()
        self.num_mixtures = len(init_mixture_probs)
        self.mixture_probs = nn.Parameter(torch.Tensor(init_mixture_probs))
        self.means = Variable(torch.Tensor(means))
        self.stds = Variable(torch.Tensor(stds))

    def sample_z(self, num_samples):
        return torch.multinomial(
            self.mixture_probs,
            num_samples, replacement=True
        )

    def sample_x(self, z):
        num_samples = len(z)
        return self.means[z] + self.stds[z] * Variable(torch.Tensor(num_samples).normal_())

    def sample(self, num_samples):
        z = self.sample_z(num_samples)
        x = self.sample_x(z)
        return z, x

    def z_logpdf(self, z):
        num_samples = len(z)
        return torch.gather(
            torch.log(self.mixture_probs).unsqueeze(0).expand(num_samples, self.num_mixtures),
            1,
            z.long().unsqueeze(-1)
        ).view(-1)

    def x_logpdf(self, x, z):
        return torch.distributions.Normal(
            mean=self.means[z],
            std=self.stds[z]
        ).log_prob(x)

    def logpdf(self, z, x):
        return self.z_logpdf(z) + self.x_logpdf(x, z)


class InferenceNetwork(nn.Module):
    def __init__(self, init_mixture_probs, means, stds):
        super(InferenceNetwork, self).__init__()
        self.num_mixtures = len(init_mixture_probs)
        self.mixture_probs = nn.Parameter(torch.Tensor(init_mixture_probs))

    def sample_z(self, x):
        num_samples = len(x)
        return torch.multinomial(
            self.mixture_probs,
            num_samples, replacement=True
        )

    def sample(self, x):
        return self.sample_z(x)

    def z_logpdf(self, z, x):
        num_samples = len(z)
        return torch.gather(
            torch.log(self.mixture_probs).unsqueeze(0).expand(num_samples, self.num_mixtures),
            1,
            z.long().unsqueeze(-1)
        ).view(-1)

    def logpdf(self, z, x):
        return self.z_logpdf(z, x)


class IWAE(nn.Module):
    def __init__(self, p_init_mixture_probs, q_init_mixture_probs, means, stds):
        super(IWAE, self).__init__()
        self.generative_network = GenerativeNetwork(p_init_mixture_probs, means, stds)
        self.inference_network = InferenceNetwork(q_init_mixture_probs, means, stds)

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


class RWS(nn.Module):
    def __init__(self, p_init_mixture_probs, q_init_mixture_probs, means, stds):
        super(RWS, self).__init__()
        self.generative_network = GenerativeNetwork(p_init_mixture_probs, means, stds)
        self.inference_network = InferenceNetwork(q_init_mixture_probs, means, stds)

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


def main():
    num_mixtures = 100
    p_init_mixture_probs = [1 / num_mixtures for _ in range(num_mixtures)]
    q_init_mixture_probs = [1 / num_mixtures for _ in range(num_mixtures)]
    means = [10 * i for i in range(num_mixtures)]
    stds = [1 for _ in range(num_mixtures)]
    num_particles_list = [10 * int(i) for i in np.arange(1, 11)]
    num_mc_samples = 1000
    iwae_reinforce_theta_grad = np.zeros([len(num_particles_list), num_mc_samples, num_mixtures])
    iwae_reinforce_phi_grad = np.zeros([len(num_particles_list), num_mc_samples, num_mixtures])

    iwae_vimco_theta_grad = np.zeros([len(num_particles_list), num_mc_samples, num_mixtures])
    iwae_vimco_phi_grad = np.zeros([len(num_particles_list), num_mc_samples, num_mixtures])

    wake_theta_grad = np.zeros([len(num_particles_list), num_mc_samples, num_mixtures])
    wake_phi_grad = np.zeros([len(num_particles_list), num_mc_samples, num_mixtures])
    sleep_phi_grad = np.zeros([len(num_particles_list), num_mc_samples, num_mixtures])

    x = Variable(torch.Tensor([2]))

    iwae = IWAE(p_init_mixture_probs, q_init_mixture_probs, means, stds)
    rws = RWS(p_init_mixture_probs, q_init_mixture_probs, means, stds)

    for num_particles_idx, num_particles in enumerate(num_particles_list):
        print('num_particles = {}'.format(num_particles))
        for mc_sample_idx in range(num_mc_samples):
            iwae.zero_grad()
            loss, elbo = iwae(x, 'reinforce', num_particles)
            loss.backward()
            iwae_reinforce_theta_grad[num_particles_idx, mc_sample_idx] = iwae.generative_network.mixture_probs.grad.data.numpy()
            iwae_reinforce_phi_grad[num_particles_idx, mc_sample_idx] = iwae.inference_network.mixture_probs.grad.data.numpy()

            iwae.zero_grad()
            loss, elbo = iwae(x, 'vimco', num_particles)
            loss.backward()
            iwae_vimco_theta_grad[num_particles_idx, mc_sample_idx] = iwae.generative_network.mixture_probs.grad.data.numpy()
            iwae_vimco_phi_grad[num_particles_idx, mc_sample_idx] = iwae.inference_network.mixture_probs.grad.data.numpy()

            rws.generative_network.zero_grad()
            loss = rws('wake_theta', x=x, num_particles=num_particles)
            loss.backward()
            wake_theta_grad[num_particles_idx, mc_sample_idx] = rws.generative_network.mixture_probs.grad.data.numpy()

            rws.inference_network.zero_grad()
            loss = rws('wake_phi', x=x, num_particles=num_particles)
            loss.backward()
            wake_phi_grad[num_particles_idx, mc_sample_idx] = rws.inference_network.mixture_probs.grad.data.numpy()

            rws.inference_network.zero_grad()
            loss = rws('sleep_phi', x=x, num_particles=num_particles, num_samples=1)
            loss.backward()
            sleep_phi_grad[num_particles_idx, mc_sample_idx] = rws.inference_network.mixture_probs.grad.data.numpy()

    iwae_reinforce_theta_grad_std = np.mean(np.std(iwae_reinforce_theta_grad, axis=1), axis=1)
    iwae_reinforce_phi_grad_std = np.mean(np.std(iwae_reinforce_phi_grad, axis=1), axis=1)

    iwae_vimco_theta_grad_std = np.mean(np.std(iwae_vimco_theta_grad, axis=1), axis=1)
    iwae_vimco_phi_grad_std = np.mean(np.std(iwae_vimco_phi_grad, axis=1), axis=1)

    wake_theta_grad_std = np.mean(np.std(wake_theta_grad, axis=1), axis=1)
    wake_phi_grad_std = np.mean(np.std(wake_phi_grad, axis=1), axis=1)
    sleep_phi_grad_std = np.mean(np.std(sleep_phi_grad, axis=1), axis=1)

    # Plotting
    fig, axs = plt.subplots(2, 1, sharex=True)
    fig.set_size_inches(3.25, 3)

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axs[-1].set_xlabel('number of particles')

    axs[0].plot(num_particles_list, iwae_reinforce_theta_grad_std, linestyle='-', color='black', label='reinforce')
    axs[0].plot(num_particles_list, iwae_vimco_theta_grad_std, linestyle='--', color='black', label='vimco')
    axs[0].plot(num_particles_list, wake_theta_grad_std, linestyle='-.',color='black', label='wake-theta')
    axs[0].set_ylabel('average std. of $\\theta$\n gradient estimator')
    axs[0].legend()

    axs[1].plot(num_particles_list, iwae_reinforce_phi_grad_std, linestyle='-', color='black', label='reinforce')
    axs[1].plot(num_particles_list, iwae_vimco_phi_grad_std, linestyle='--', color='black', label='vimco')
    axs[1].plot(num_particles_list, wake_phi_grad_std, linestyle='-.', color='black', label='wake-phi')
    axs[1].plot(num_particles_list, sleep_phi_grad_std, linestyle=':', color='black', label='sleep-phi')

    axs[1].set_yscale('log')
    axs[1].set_ylim(1)
    axs[1].legend(ncol=2, loc='center left', bbox_to_anchor=(0, 0.35))
    axs[1].set_ylabel('average std. of $\phi$\n gradient estimator')

    fig.tight_layout()
    filename = 'gmm_gradient.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))


if __name__ == '__main__':
    main()
