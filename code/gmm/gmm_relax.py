from util import *
import itertools
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import uuid


epsilon = 1e-6


class Prior(nn.Module):
    def __init__(self, init_mixture_probs_pre_softmax, softmax_multiplier=0.5):
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

    def forward(self):
        return torch.distributions.OneHotCategorical(probs=self.probs())


class Likelihood():
    def __init__(self, mean_multiplier, stds):
        self.num_mixtures = len(stds)
        self.means = mean_multiplier * torch.arange(self.num_mixtures)
        self.stds = torch.Tensor(stds)

    def __call__(self, latent=None):
        return torch.distributions.Normal(
            loc=torch.sum(self.means * latent, dim=-1),
            scale=torch.sum(self.stds * latent, dim=-1)
        )


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

    def probs(self, observation):
        return self.mlp(observation.unsqueeze(-1))

    def forward(self, observation):
        return torch.distributions.OneHotCategorical(
            probs=self.probs(observation)
        )


class ControlVariate(nn.Module):
    def __init__(self, num_mixtures):
        super(ControlVariate, self).__init__()
        self.num_mixtures = num_mixtures
        self.mlp = nn.Sequential(
            nn.Linear(num_mixtures + 1, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

    def forward(self, aux, observation):
        return self.mlp(
            torch.cat([observation.unsqueeze(-1), aux], dim=1)
        ).squeeze(-1)


# This implements Appendix C in the REBAR paper
def sample_relax(inference_network, control_variate,
                 observation, num_particles):
    batch_size = len(observation)
    num_mixtures = inference_network.num_mixtures
    probs = inference_network.probs(observation)
    probs_expanded = probs.unsqueeze(1).expand(
        batch_size, num_particles, num_mixtures
    ).contiguous().view(batch_size * num_particles, num_mixtures)

    # latent_aux
    u = torch.distributions.Uniform(0 + epsilon, 1 - epsilon).sample(
        sample_shape=(batch_size * num_particles, num_mixtures)
    )
    latent_aux = torch.log(probs_expanded) - torch.log(-torch.log(u))

    # latent
    latent = torch.zeros(batch_size * num_particles, num_mixtures)
    k = torch.argmax(latent_aux, dim=1)
    arange = torch.arange(batch_size * num_particles).long()
    latent[arange, k] = 1

    # latent_aux_tilde
    v = torch.distributions.Uniform(0 + epsilon, 1 - epsilon).sample(
        sample_shape=(batch_size * num_particles, num_mixtures)
    )
    latent_aux_tilde = torch.zeros(batch_size * num_particles, num_mixtures)
    latent_aux_tilde[latent.byte()] = -torch.log(-torch.log(v[latent.byte()]))
    latent_aux_tilde[1 - latent.byte()] = -torch.log(
        -torch.log(v[1 - latent.byte()]) / probs_expanded[1 - latent.byte()] -
        torch.log(v[latent.byte()])
        .unsqueeze(-1).expand(-1, num_mixtures - 1).contiguous().view(-1)
    )
    return [x.view(batch_size, num_particles, num_mixtures)
            for x in [latent, latent_aux, latent_aux_tilde]]


def elbo_relax(prior, likelihood, inference_network, control_variate,
               observation, num_particles):
    batch_size = len(observation)
    num_mixtures = prior.num_mixtures
    latent, latent_aux, latent_aux_tilde = sample_relax(
        inference_network, control_variate, observation, num_particles
    )
    observation_expanded = observation.unsqueeze(-1).expand(-1, num_particles)
    log_inf = inference_network(observation).log_prob(latent.t()).t()
    logweight = prior().log_prob(latent) + \
        likelihood(latent).log_prob(observation_expanded) - \
        log_inf
    log_evidence = logsumexp(logweight, dim=1) - np.log(num_particles)
    c, c_tilde = [
        torch.mean(control_variate(
            aux.view(-1, num_mixtures),
            observation_expanded.contiguous().view(-1)
        ).contiguous().view(batch_size, num_particles), dim=1)
        for aux in [latent_aux, latent_aux_tilde]
    ]
    return (log_evidence - c_tilde).detach() * torch.sum(log_inf, dim=1) + \
        c - c_tilde + log_evidence


def train_relax(prior, likelihood, inference_network, control_variate,
                num_particles, dataloader, num_iterations, callback):
    num_inf_params = sum([inf_param.nelement()
                          for inf_param in inference_network.parameters()])
    iwae_optimizer = torch.optim.Adam(
        itertools.chain(prior.parameters(), inference_network.parameters())
    )
    control_variate_optimizer = torch.optim.Adam(control_variate.parameters())

    for iteration_idx, observation in enumerate(dataloader):
        if iteration_idx == num_iterations:
            break

        iwae_optimizer.zero_grad()
        control_variate_optimizer.zero_grad()
        loss = -torch.mean(elbo_relax(
            prior, likelihood, inference_network, control_variate, observation,
            num_particles
        ))
        if torch.isnan(loss):
            import pdb; pdb.set_trace()
        loss.backward(create_graph=True)
        iwae_optimizer.step()

        control_variate_optimizer.zero_grad()
        torch.autograd.backward(
            [2 * inf_param.grad / num_inf_params
             for inf_param in inference_network.parameters()],
            [inf_param.grad.detach()
             for inf_param in inference_network.parameters()]
        )
        control_variate_optimizer.step()
        callback(iteration_idx, prior, likelihood, inference_network,
                 control_variate, observation)


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, prior, likelihood, batch_size):
        self.prior = prior
        self.likelihood = likelihood
        self.batch_size = batch_size

    def __getitem__(self, index):
        latent = self.prior().sample(sample_shape=(self.batch_size,)).detach()
        return self.likelihood(latent).sample().detach()

    def __len__(self):
        return sys.maxsize  # effectively infinite


def get_synthetic_dataloader(prior, likelihood, batch_size):
    return torch.utils.data.DataLoader(
        SyntheticDataset(prior, likelihood, batch_size),
        batch_size=1,
        collate_fn=lambda x: x[0]
    )


def get_log_evidence(prior, likelihood, observation):
    num_samples = len(observation)
    num_mixtures = prior.num_mixtures
    return logsumexp(
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
    log_joint_expanded = prior().log_prob(z_expanded) + \
        likelihood(latent=z_expanded).log_prob(observation_expanded)

    return torch.exp(log_joint_expanded - log_evidence_expanded)


class TrainingStats(object):
    def __init__(self, saving_interval, logging_interval, true_prior,
                 likelihood, num_test_data, num_mc_samples, num_particles, uid,
                 seed):
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
        self.inference_network_grad_phi_std_history = []

        self.test_observation = next(iter(get_synthetic_dataloader(
            true_prior, likelihood, num_test_data
        )))
        self.true_posterior = get_posterior(
            true_prior, likelihood, self.test_observation
        ).detach().numpy()

    def __call__(self, iteration_idx, prior, likelihood, inference_network,
                 control_variate, observation):
        if iteration_idx % self.saving_interval == 0:
            self.iteration_idx_history.append(iteration_idx)
            self.prior_l2_history.append(np.linalg.norm(
                prior.probs().detach().numpy() -
                self.true_prior.probs().detach().numpy()
            ))
            posterior = get_posterior(prior, likelihood,
                                      self.test_observation).detach().numpy()
            self.posterior_l2_history.append(np.mean(np.linalg.norm(
                inference_network.probs(
                    self.test_observation
                ).detach().numpy() - posterior,
                axis=1
            )))
            self.true_posterior_l2_history.append(np.mean(np.linalg.norm(
                inference_network.probs(
                    self.test_observation
                ).detach().numpy() - self.true_posterior,
                axis=1
            )))
            filename = '{}/relax_{}_{}_{}.pt'.format(
                WORKING_DIR, iteration_idx, self.seed, self.uid
            )
            torch.save({'prior': prior.state_dict(),
                        'inference_network': inference_network.state_dict(),
                        'control_variate': control_variate.state_dict()},
                       filename)
            inference_network_grad_phi_std = OnlineMeanStd()
            for mc_sample_idx in range(self.num_mc_samples):
                prior.zero_grad()
                inference_network.zero_grad()
                control_variate.zero_grad()
                loss = -torch.mean(elbo_relax(
                    prior, likelihood, inference_network, control_variate,
                    observation, self.num_particles
                ))
                loss.backward()
                inference_network_grad_phi_std.update(
                    [p.grad for p in inference_network.parameters()]
                )
                prior.zero_grad()
                inference_network.zero_grad()
                control_variate.zero_grad()
            self.inference_network_grad_phi_std_history.append(
                inference_network_grad_phi_std.avg_of_means_stds()[1]
            )
        if iteration_idx % self.logging_interval == 0:
            print('Iteration: {}'.format(iteration_idx))


def main(args, uid):
    num_mixtures = args.num_mixtures
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
        ['num_mixtures',
         'num_particles', 'batch_size', 'num_iterations', 'saving_interval',
         'logging_interval', 'num_mc_samples', 'mean_multiplier',
         'softmax_multiplier', 'num_test_data', 'true_mixture_probs'],
        [num_mixtures, num_particles,
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

        inference_network = InferenceNetwork(num_mixtures)
        control_variate = ControlVariate(num_mixtures)
        training_stats = TrainingStats(
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
        train_relax(prior, likelihood, inference_network, control_variate,
                    num_particles, get_synthetic_dataloader(
                        true_prior, likelihood, batch_size
                    ), num_iterations, training_stats)
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
            filename = '{}/relax_{}_{}_{}.npy'.format(
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
    parser.add_argument('--logging-interval', type=int, default=50000)
    parser.add_argument('--seeds', nargs='*', type=int, default=[1])
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    uid = str(uuid.uuid4())[:8]

    print('args: {}'.format(args))
    print('uid: {}'.format(uid))
    main(args, uid)
