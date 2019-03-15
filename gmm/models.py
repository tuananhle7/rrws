import torch
import torch.nn as nn
import torch.nn.functional as F


class GenerativeModel(nn.Module):
    def __init__(self, init_mixture_logits, softmax_multiplier=0.5,
                 device=torch.device('cpu')):
        super(GenerativeModel, self).__init__()
        self.num_mixtures = len(init_mixture_logits)
        self.mixture_logits = nn.Parameter(torch.tensor(
            init_mixture_logits, device=device, dtype=torch.float))
        self.locs = 10 * torch.arange(self.num_mixtures, device=device,
                                      dtype=torch.float)
        self.scale = torch.tensor(5, device=device, dtype=torch.float)
        self.logit_multiplier = 0.5
        self.device = device

    def get_latent_params(self):
        return F.softmax(self.mixture_logits * self.logit_multiplier, dim=0)

    def get_latent_dist(self):
        """Returns: distribution with batch shape [] and event shape
            [num_mixtures].
        """
        return torch.distributions.OneHotCategorical(
            logits=self.mixture_logits * self.logit_multiplier)

    def get_obs_dist(self, latent):
        """Args:
            latent: tensor of shape [dim0, ..., dimN, num_mixtures]

        Returns: distribution with batch shape [dim0, ..., dimN] and
            event shape []
        """
        return torch.distributions.Normal(
            loc=torch.sum(self.locs * latent, dim=-1), scale=self.scale)

    def get_log_prob(self, latent, obs):
        """Log of joint probability.

        Args:
            latent: tensor of shape [dim1, ..., dimN, batch_size, num_mixtures]
            obs: tensor of shape [batch_size]

        Returns: tensor of shape [dim1, ..., dimN, batch_size]
        """

        latent_log_prob = self.get_latent_dist().log_prob(latent)
        obs_log_prob = self.get_obs_dist(latent).log_prob(obs)
        return latent_log_prob + obs_log_prob

    def sample_latent_and_obs(self, num_samples=1):
        """Args:
            num_samples: int

        Returns:
            latent: tensor of shape [num_samples, num_mixtures]
            obs: tensor of shape [num_samples]
        """

        latent_dist = self.get_latent_dist()
        latent = latent_dist.sample((num_samples,))
        obs_dist = self.get_obs_dist(latent)
        obs = obs_dist.sample()

        return latent, obs

    def sample_obs(self, num_samples=1):
        """Args:
            num_samples: int

        Returns:
            obs: tensor of shape [num_samples]
        """

        return self.sample_latent_and_obs(num_samples)[1]

    def get_log_evidence(self, obs):
        """Args:
            obs: tensor of shape [batch_size]

        Returns: tensor of shape [batch_size]
        """
        batch_size = len(obs)
        return torch.logsumexp(
            torch.log(
                self.get_latent_params().unsqueeze(0).expand(batch_size, -1)
            ) + torch.distributions.Normal(
                loc=self.locs.unsqueeze(0).expand(batch_size, -1),
                scale=self.scale
            ).log_prob(obs.unsqueeze(-1).expand(-1, self.num_mixtures)),
            dim=1)

    def get_posterior_probs(self, obs):
        """Args:
            obs: tensor of shape [batch_size]

        Returns: tensor of shape [batch_size, num_mixtures]
        """
        batch_size = len(obs)
        latent = torch.eye(
            self.num_mixtures, dtype=torch.float, device=self.device
        ).unsqueeze(1).expand(self.num_mixtures, batch_size, self.num_mixtures)
        log_evidence = self.get_log_evidence(obs)
        log_joint = self.get_log_prob(latent, obs)
        log_posterior = log_joint - log_evidence
        return torch.exp(log_posterior).t()


class InferenceNetwork(nn.Module):
    def __init__(self, num_mixtures, relaxed_one_hot=False, temperature=None,
                 device=torch.device('cpu')):
        super(InferenceNetwork, self).__init__()
        self.num_mixtures = num_mixtures
        self.mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, self.num_mixtures),
            nn.Softmax(dim=1))
        self.relaxed_one_hot = relaxed_one_hot
        if temperature is not None:
            self.temperature = torch.tensor(temperature, device=device,
                                            dtype=torch.float)
        self.device = device

    def set_temperature(self, temperature):
        self.temperature = torch.tensor(temperature, device=self.device,
                                        dtype=torch.float)

    def get_latent_params(self, obs):
        """Args:
            obs: tensor of shape [batch_size]

        Returns: tensor of shape [batch_size, num_mixtures]
        """
        return self.mlp(obs.unsqueeze(-1))

    def get_latent_dist(self, obs):
        """Args:
            obs: tensor of shape [batch_size]

        Returns: distribution with batch shape [batch_size] and event shape
            [num_mixtures]
        """
        probs = self.get_latent_params(obs)
        if self.relaxed_one_hot:
            return torch.distributions.RelaxedOneHotCategorical(
                self.temperature, probs=probs)
        else:
            return torch.distributions.OneHotCategorical(probs=probs)

    def sample_from_latent_dist(self, latent_dist, num_samples, reparam=False):
        """Samples from q(latent | obs)

        Args:
            latent_dist: distribution with batch shape [batch_size] and event
                shape [num_mixtures]
            num_samples: int
            reparam: reparameterize sampling from q (only applicable if z is
                Concrete)

        Returns:
            latent: tensor of shape [num_samples, batch_size, num_mixtures]
        """
        if reparam:
            return latent_dist.rsample((num_samples,))
        else:
            return latent_dist.sample((num_samples,))

    def get_log_prob_from_latent_dist(self, latent_dist, latent):
        """Log q(latent | obs).

        Args:
            latent_dist: distribution with batch shape [batch_size] and event
                shape [num_mixtures]
            latent: tensor of shape [dim1, ..., dimN, batch_size, num_mixtures]

        Returns: tensor of shape [dim1, ..., dimN, batch_size]
        """

        return latent_dist.log_prob(latent)

    def get_log_prob(self, latent, obs):
        """Log q(latent | obs).

        Args:
            latent: tensor of shape [dim1, ..., dimN, batch_size, num_mixtures]
            obs: tensor of shape [batch_size]

        Returns: tensor of shape [dim1, ..., dimN, batch_size]
        """
        return self.get_log_prob_from_latent_dist(
            self.get_latent_dist(obs), latent)


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

    def forward(self, aux, obs):
        """Args:
            aux: tensor of shape [batch_size, num_particles, num_mixtures]
            obs: tensor of shape [batch_size]

        Returns: tensor of shape [batch_size]
        """
        batch_size, num_particles, num_mixtures = aux.shape
        obs_expanded = obs.unsqueeze(-1).expand(-1, num_particles)
        return torch.mean(self.mlp(torch.cat([
            obs_expanded.contiguous().view(-1).unsqueeze(-1),
            aux.view(-1, num_mixtures)], dim=1)).squeeze(-1).view(
                batch_size, num_particles),
            dim=1)
