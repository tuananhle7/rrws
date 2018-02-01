from torch.autograd import Variable

import aesmc.model
import aesmc.random_variable as rv
import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot(i, num_bins):
    if isinstance(i, Variable):
        return Variable(
            torch.zeros(len(i), num_bins)
        ).scatter_(1, i.long().unsqueeze(-1), 1)
    else:
        return torch.zeros(len(i), num_bins).scatter_(
            1, i.long().unsqueeze(-1), 1
        )


class HMMInitialDistribution(aesmc.model.InitialDistribution):
    def __init__(self, initial_probs):
        self.num_states = len(initial_probs)
        self.initial_probs = Variable(torch.Tensor(initial_probs))

    def initial(self):
        return rv.StateRandomVariable(random_variables={
            'x': rv.Categorical(
                self.initial_probs.unsqueeze(0).unsqueeze(0).expand(
                    self.batch_size, self.num_particles, -1
                )
            )
        })


class HMMTransitionNetwork(aesmc.model.TransitionNetwork):
    def __init__(self, num_states):
        super(HMMTransitionNetwork, self).__init__()
        self.num_states = num_states
        self.pre_softmax = nn.Parameter(torch.eye(num_states))

    def get_transition_matrix(self):
        return F.softmax(self.pre_softmax, dim=1)

    def transition(self, previous_latent_state=None, time=None):
        previous_x = previous_latent_state.x
        batch_size, num_particles = previous_x.size()
        previous_x_expanded = previous_x.unsqueeze(-1).unsqueeze(-1).expand(
            batch_size, num_particles, 1, self.num_states
        )

        transition_matrix = self.get_transition_matrix()
        transition_matrix_expanded = transition_matrix.unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_particles, self.num_states, self.num_states
        )
        current_x_probs = torch.gather(
            transition_matrix_expanded,
            dim=2,
            index=previous_x_expanded.long()
        ).squeeze(2)
        return rv.StateRandomVariable(random_variables={
            'x': rv.Categorical(current_x_probs)
        })


class HMMEmissionDistribution(aesmc.model.EmissionDistribution):
    def __init__(self, obs_means, obs_vars):
        self.obs_means = Variable(torch.Tensor(obs_means))
        self.obs_vars = Variable(torch.Tensor(obs_vars))
        self.num_states = len(obs_means)

    def emission(self, latent_state=None, time=None):
        batch_size, num_particles = latent_state.x.size()
        latent_x_expanded = latent_state.x.long().unsqueeze(-1)
        obs_means_expanded = self.obs_means.unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_particles, self.num_states
        )
        obs_vars_expanded = self.obs_vars.unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_particles, self.num_states
        )
        return rv.StateRandomVariable(random_variables={
            'y': rv.Normal(
                mean=torch.gather(
                    obs_means_expanded,
                    dim=2,
                    index=latent_x_expanded
                ).squeeze(2),
                variance=torch.gather(
                    obs_vars_expanded,
                    dim=2,
                    index=latent_x_expanded
                ).squeeze(2)
            )
        })


class HMMProposalNetwork(aesmc.model.ProposalNetwork):
    def __init__(self, num_states):
        super(HMMProposalNetwork, self).__init__()
        self.num_states = num_states
        self.initial_mlp = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, num_states),
            nn.Softmax(dim=1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(1 + num_states, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, num_states),
            nn.Softmax(dim=1)
        )

    def proposal(
        self, previous_latent_state=None, time=None, observation_states=None
    ):
        y = observation_states[time].y
        batch_size, num_particles = y.size()
        if time == 0:
            probabilities = self.initial_mlp(y.contiguous().view(-1, 1)).view(
                batch_size, num_particles, self.num_states
            )
        else:
            previous_x = previous_latent_state.x
            previous_x_one_hots = one_hot(previous_x.view(-1), self.num_states)
            probabilities = self.mlp(torch.cat(
                [y.contiguous().view(-1, 1), previous_x_one_hots], dim=1
            )).view(batch_size, num_particles, self.num_states)
        return rv.StateRandomVariable(random_variables={
            'x': rv.Categorical(probabilities)
        })
