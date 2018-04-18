from util import *

import copy
import dgm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import uuid


class Prior(dgm.model.InitialNetwork):
    def __init__(self, init_mixture_probs_pre_softmax, softmax_multiplier):
        super(Prior, self).__init__()
        self.num_mixtures = len(init_mixture_probs_pre_softmax)
        self.mixture_probs_pre_softmax = nn.Parameter(
            torch.Tensor(init_mixture_probs_pre_softmax)
        )
        self.softmax_multiplier = softmax_multiplier

    def probs(self):
        return F.softmax(
            self.mixture_probs_pre_softmax * self.softmax_multiplier, dim=0
        )

    def initial(self):
        return torch.distributions.OneHotCategorical(probs=self.probs())


class Likelihood(dgm.model.EmissionDistribution):
    def __init__(self, mean_multiplier, stds):
        self.num_mixtures = len(stds)
        self.means = mean_multiplier * torch.arange(self.num_mixtures)
        self.stds = torch.Tensor(stds)

    def emission(self, latent=None, time=None):
        return torch.distributions.Normal(
            loc=torch.sum(self.means * latent, dim=-1),
            scale=torch.sum(self.stds * latent, dim=-1)
        )


class InferenceNetwork(dgm.model.ProposalNetwork):
    def __init__(self, num_mixtures, temperature):
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
        self.temperature = temperature

    def probs(self, observation):
        return self.mlp(observation.unsqueeze(-1))

    def proposal(self, previous_latent=None, time=None, observations=None):
        return torch.distributions.RelaxedOneHotCategorical(
            temperature=torch.Tensor([self.temperature]),
            probs=self.probs(observations[0])
        )


class TrainingStats(object):
    def __init__(
        self,
        temperature_schedule,
        saving_interval,
        logging_interval,
        true_prior,
        likelihood,
        num_test_data,
        num_mc_samples,
        num_particles,
        uid,
        seed
    ):
        self.temperature_schedule = temperature_schedule
        self.saving_interval = saving_interval
        self.logging_interval = logging_interval
        self.true_prior = true_prior
        self.likelihood = likelihood
        self.num_test_data = num_test_data
        self.num_mc_samples = num_mc_samples
        self.num_particles = num_particles
        self.uid = uid
        self.seed = seed

        self.iteration_idx_history = []
        self.prior_l2_history = []
        self.posterior_l2_history = []
        self.true_posterior_l2_history = []
        self.temperature_history = []
        self.inference_network_grad_phi_std_history = []

        dataloader = dgm.train.get_synthetic_dataloader(
            true_prior, None, likelihood, 1, num_test_data
        )
        self.test_observations = next(iter(dataloader))
        self.true_posterior = get_posterior(
            true_prior, likelihood, self.test_observations[0]
        ).detach().numpy()

    def __call__(
        self,
        epoch_idx,
        epoch_iteration_idx,
        autoencoder,
        observations
    ):
        if epoch_iteration_idx % self.saving_interval == 0:
            self.iteration_idx_history.append(epoch_iteration_idx)
            self.prior_l2_history.append(np.linalg.norm(
                autoencoder.initial.probs().detach().numpy() -
                self.true_prior.probs().detach().numpy()
            ))
            posterior = get_posterior(
                autoencoder.initial, self.likelihood, self.test_observations[0]
            ).detach().numpy()
            self.posterior_l2_history.append(np.mean(np.linalg.norm(
                autoencoder.proposal.probs(
                    self.test_observations[0]
                ).detach().numpy() - posterior,
                axis=1
            )))
            self.true_posterior_l2_history.append(np.mean(np.linalg.norm(
                autoencoder.proposal.probs(
                    self.test_observations[0]
                ).detach().numpy() - self.true_posterior,
                axis=1
            )))
            filename = '{}/concrete_{}_{}_{}.pt'.format(
                WORKING_DIR, epoch_iteration_idx, self.seed, self.uid
            )
            torch.save(autoencoder.state_dict(), filename)
            autoencoder.proposal.temperature = self.temperature_schedule(
                epoch_idx, epoch_iteration_idx
            )
            self.temperature_history.append(autoencoder.proposal.temperature)
            inference_network_grad_phi_std = OnlineMeanStd()
            for mc_sample_idx in range(self.num_mc_samples):
                autoencoder.zero_grad()
                loss = -torch.mean(autoencoder.forward(
                    observations,
                    self.num_particles,
                    dgm.autoencoder.AutoencoderAlgorithm.IWAE
                ))
                loss.backward()
                inference_network_grad_phi_std.update(
                    [p.grad for p in autoencoder.proposal.parameters()]
                )
            self.inference_network_grad_phi_std_history.append(
                inference_network_grad_phi_std.avg_of_means_stds()[1]
            )
        if epoch_iteration_idx % self.logging_interval == 0:
            print('Iteration: {}'.format(epoch_iteration_idx))


def get_log_evidence(prior, likelihood, observation):
    num_samples = len(observation)
    num_mixtures = prior.num_mixtures
    return dgm.math.logsumexp(
        torch.log(
            prior.probs().unsqueeze(0).expand(num_samples, -1)
        ) + torch.distributions.Normal(
            loc=likelihood.means.unsqueeze(0).expand(num_samples, -1),
            scale=likelihood.stds.unsqueeze(0).expand(num_samples, -1)
        ).log_prob(
            observation.unsqueeze(-1).expand(-1, num_mixtures)
        ),
        dim=1
    )


# https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/4
def one_hot(x, length):
    return torch.zeros(len(x), length).scatter_(1, x.unsqueeze(-1), 1)


def get_posterior(prior, likelihood, observation):
    num_samples = len(observation)
    z = one_hot(torch.arange(prior.num_mixtures).long(), prior.num_mixtures)
    log_evidence = get_log_evidence(prior, likelihood, observation)

    z_expanded = z.unsqueeze(0).expand(
        num_samples, prior.num_mixtures, prior.num_mixtures
    )
    observation_expanded = observation.unsqueeze(-1).expand(
        num_samples, prior.num_mixtures
    )
    log_evidence_expanded = log_evidence.unsqueeze(-1).expand(
        num_samples, prior.num_mixtures
    )
    log_joint_expanded = prior.initial().log_prob(z_expanded) + \
        likelihood.emission(latent=z_expanded).log_prob(observation_expanded)

    return torch.exp(log_joint_expanded - log_evidence_expanded)


class TemperatureSchedule():
    def __init__(self, init_temperature, end_temperature, num_iterations):
        self.init_temperature = init_temperature
        self.end_temperature = end_temperature
        self.num_iterations = num_iterations

    def __call__(self, epoch_idx, epoch_iteration_idx):
        return self.init_temperature + epoch_iteration_idx * \
            (self.end_temperature - self.init_temperature) / \
            self.num_iterations


def main(args, uid):
    num_mixtures = args.num_mixtures
    init_temperature = 3
    end_temperature = 1
    num_particles = args.num_particles
    batch_size = 100
    num_iterations = args.num_iterations
    saving_interval = args.saving_interval
    logging_interval = args.logging_interval
    num_mc_samples = 10
    mean_multiplier = 10
    softmax_multiplier = 0.5
    num_test_data = 100

    temp = np.arange(num_mixtures) + 5
    true_mixture_probs = temp / np.sum(temp)
    init_mixture_probs_pre_softmax = np.array(
        list(reversed(2 * np.arange(num_mixtures)))
    )
    stds = np.array([5 for _ in range(num_mixtures)])
    true_prior = Prior(
        init_mixture_probs_pre_softmax=np.log(
            true_mixture_probs
        ) / softmax_multiplier,
        softmax_multiplier=softmax_multiplier
    )
    prior = Prior(
        init_mixture_probs_pre_softmax=init_mixture_probs_pre_softmax,
        softmax_multiplier=softmax_multiplier
    )
    likelihood = Likelihood(mean_multiplier, stds)
    for name, data in zip(
        ['num_mixtures', 'init_temperature', 'end_temperature',
         'num_particles', 'batch_size', 'num_iterations', 'saving_interval',
         'logging_interval', 'num_mc_samples', 'mean_multiplier',
         'softmax_multiplier', 'num_test_data', 'true_mixture_probs'],
        [num_mixtures, init_temperature, end_temperature, num_particles,
         batch_size, num_iterations, saving_interval, logging_interval,
         num_mc_samples, mean_multiplier, softmax_multiplier, num_test_data,
         true_mixture_probs],
    ):
        filename = '{}/{}_{}.npy'.format(WORKING_DIR, name, uid)
        np.save(filename, data)
        print('Saved to {}'.format(filename))

    for seed in args.seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if args.cuda:
            torch.cuda.manual_seed(seed)

        inference_network = InferenceNetwork(
            num_mixtures, temperature=init_temperature
        )
        autoencoder = dgm.autoencoder.AutoEncoder(
            prior, None, likelihood, inference_network
        )
        training_stats = TrainingStats(
            TemperatureSchedule(
                init_temperature, end_temperature, num_iterations
            ),
            saving_interval=saving_interval,
            logging_interval=logging_interval,
            true_prior=true_prior,
            likelihood=likelihood,
            num_test_data=num_test_data,
            num_mc_samples=num_mc_samples,
            num_particles=num_particles,
            uid=uid,
            seed=seed
        )
        dgm.train.train_autoencoder(
            autoencoder,
            dgm.train.get_synthetic_dataloader(
                true_prior, None, likelihood, 1, batch_size
            ),
            autoencoder_algorithm=dgm.autoencoder.AutoencoderAlgorithm.IWAE,
            num_epochs=1,
            num_iterations_per_epoch=num_iterations,
            num_particles=num_particles,
            callback=training_stats
        )
        for data, name in zip(
            [
                training_stats.prior_l2_history,
                training_stats.posterior_l2_history,
                training_stats.true_posterior_l2_history,
                training_stats.inference_network_grad_phi_std_history
            ],
            [
                'prior_l2_history',
                'posterior_l2_history',
                'true_posterior_l2_history',
                'inference_network_grad_phi_std_history'
            ]
        ):
            filename = '{}/concrete_{}_{}_{}.npy'.format(
                WORKING_DIR, name, seed, uid
            )
            np.save(filename, data)
            print('Saved to {}'.format(filename))


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    import argparse

    parser = argparse.ArgumentParser(description='GMM')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA use')
    parser.add_argument('--num-mixtures', type=int, default=20)
    parser.add_argument('--num-particles', type=int, default=2)
    parser.add_argument('--num-iterations', type=int, default=100000)
    parser.add_argument('--saving-interval', type=int, default=1000)
    parser.add_argument('--logging-interval', type=int, default=1000)
    parser.add_argument('--seeds', nargs='*', type=int, default=[1])
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    uid = str(uuid.uuid4())[:8]

    print('args: {}'.format(args))
    print('uid: {}'.format(uid))
    main(args, uid)
