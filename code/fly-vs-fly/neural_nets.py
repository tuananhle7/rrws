from torch.autograd import Variable
from util import *

import aesmc.model
import aesmc.random_variable as rv
import aesmc.state as st

import torch
import torch.nn as nn


def one_hot(i, num_bins):
    if isinstance(i, Variable):
        return Variable(
            torch.zeros(len(i), num_bins)
        ).scatter_(1, i.long().unsqueeze(-1), 1)
    else:
        return torch.zeros(len(i), num_bins).scatter_(
            1, i.long().unsqueeze(-1), 1
        )


class FlyInitialDistribution(aesmc.model.InitialDistribution):
    def __init__(self, num_flies, num_actions, mental_state_dim):
        self.num_flies = num_flies
        self.num_actions = num_actions
        self.mental_state_dim = mental_state_dim

        self.action_probabilities = Variable(torch.Tensor([1 / self.num_actions]).expand(self.num_actions))
        self.position_mean = Variable(torch.zeros(2))
        self.position_var = Variable(torch.ones(2)) # learn these values (either by hand or otherwise)
        self.mental_state_mean = Variable(torch.zeros(mental_state_dim))
        self.mental_state_var = Variable(torch.ones(mental_state_dim))

    def initial(self):
        initial_state_random_variable = rv.StateRandomVariable()
        for f in range(self.num_flies):
            # Actions
            initial_state_random_variable.set_random_variable_(
                'action_{}'.format(f),
                rv.Categorical(self.action_probabilities.unsqueeze(0).unsqueeze(0).expand(
                    self.batch_size, self.num_particles, self.num_actions
                ))
            )

            # Positions
            initial_state_random_variable.set_random_variable_(
                'position_{}'.format(f),
                rv.MultivariateIndependentNormal(
                    self.position_mean.unsqueeze(0).unsqueeze(0).expand(
                        self.batch_size, self.num_particles, 2
                    ),
                    self.position_var.unsqueeze(0).unsqueeze(0).expand(
                        self.batch_size, self.num_particles, 2
                    )
                )
            )

            # Mental states
            initial_state_random_variable.set_random_variable_(
                'mental_state_{}'.format(f),
                rv.MultivariateIndependentNormal(
                    self.mental_state_mean.unsqueeze(0).unsqueeze(0).expand(
                        self.batch_size, self.num_particles, 2
                    ),
                    self.mental_state_var.unsqueeze(0).unsqueeze(0).expand(
                        self.batch_size, self.num_particles, 2
                    )
                )
            )


class FlyTransitionNetwork(aesmc.model.TransitionNetwork):
    def __init__(self, num_flies, num_actions, mental_state_dim):
        super(FlyTransitionNetwork, self).__init__()
        self.num_flies = num_flies
        self.num_actions = num_actions
        self.mental_state_dim = mental_state_dim

        # Action neural nets
        hidden_layer_dim = 200
        action_neural_nets = []
        for _ in range(self.num_flies):
            action_neural_nets.append(
                nn.Sequential(
                    nn.Linear(self.mental_state_dim, hidden_layer_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_layer_dim, hidden_layer_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_layer_dim, self.num_actions),
                    nn.Softmax(dim=1)
                )
            )

        # Position neural nets
        hidden_layer_dim = 200
        position_neural_nets = []
        for _ in range(self.num_flies):
            position_neural_nets.append(
                nn.Sequential(
                    nn.Linear(self.mental_state_dim, hidden_layer_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_layer_dim, hidden_layer_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_layer_dim, 2)
                )
            )
        self.position_var = Variable(torch.ones(2)) # learn these values (either by hand or otherwise)

        # Mental state neural nets
        hidden_layer_dim = 200
        mental_state_neural_nets = []
        for _ in range(self.num_flies):
            mental_state_neural_nets.append(
                nn.Sequential(
                    nn.Linear(self.mental_state_dim + 2, hidden_layer_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_layer_dim, hidden_layer_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_layer_dim, self.mental_state_dim)
                )
            )
        self.mental_state_var = Variable(torch.ones(mental_state_dim))

    def transition(self, previous_latent_state=None, time=None):
        latent_state_random_variable = rv.StateRandomVariable()
        for f in range(self.num_flies):
            previous_action = previous_latent_state._values['action_{}'.format(f)]
            previous_position = previous_latent_state._values['position_{}'.format(f)]
            previous_mental_state = previous_latent_state._values['mental_state_{}'.format(f)]

            # Action
            latent_state_random_variable.set_random_variable_(
                'action_{}'.format(f),
                rv.Categorical(select_evaluate_scatter(
                    previous_mental_state.view(-1, self.mental_state_dim),
                    previous_action.view(-1),
                    self.action_neural_nets,
                    out_dim=self.num_actions
                ).view(self.batch_size, self.num_particles, self.num_actions))
            )

            # Position
            latent_state_random_variable.set_random_variable_(
                'position_{}'.format(f),
                rv.MultivariateIndependentNormal(
                    mean=select_evaluate_scatter(
                        previous_mental_state.view(-1, self.mental_state_dim),
                        previous_action.view(-1),
                        self.position_neural_nets,
                        out_dim=2
                    ).view(self.batch_size, self.num_particles, 2),
                    variance=self.position_var.unsqueeze(0).unsqueeze(0).expand(
                        self.batch_size, self.num_particles, 2
                    )
                )
            )

            # Mental state
            latent_state_random_variable.set_random_variable_(
                'mental_state_{}'.format(f),
                rv.MultivariateIndependentNormal(
                    mean=select_evaluate_scatter(
                        torch.cat(
                            [
                                previous_mental_state.view(-1, self.mental_state_dim),
                                previous_position.view(-1, 2)
                            ], dim=1
                        ),
                        previous_action.view(-1),
                        self.mental_state_neural_nets,
                        out_dim=self.mental_state_dim
                    ).view(self.batch_size, self.num_particles, self.mental_state_dim),
                    variance=self.mental_state_var.unsqueeze(0).unsqueeze(0).expand(
                        self.batch_size, self.num_particles, self.mental_state_dim
                    )
                )
            )
        return latent_state_random_variable


class FlyEmissionNetwork(aesmc.model.EmissionNetwork):
    def __init__(self, num_flies, mental_state_dim, num_encoding_bins):
        self.num_flies = num_flies
        self.mental_state_dim = mental_state_dim
        self.num_encoding_bins = num_encoding_bins
        self.position_sensor_var = Variable(torch.ones(2))
        self.encoding_neural_net = nn.Sequential(
            nn.Linear(self.mental_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_encoding_bins),
            nn.Sigmoid()
        )

    def emission(self, latent_state=None, time=None):
        observation_state_random_variable = rv.StateRandomVariable()
        for f in range(self.num_flies):
            position = latent_state._values['position_{}'.format(f)]
            mental_state = latent_state._values['mental_state_{}'.format(f)]

            # Position
            observation_state_random_variable.set_random_variable_(
                'observed_position_{}'.format(f),
                rv.MultivariateIndependentNormal(
                    mean=position,
                    variance=self.position_sensor_var.unsqueeze(0).unsqueeze(0).expand(
                        self.batch_size, self.num_particles, 2
                    )
                )
            )

            # Encoding
            observation_state_random_variable.set_random_variable_(
                'observed_encoding_{}'.format(f),
                rv.MultivariateIndependentPseudobernoulli(
                    self.encoding_neural_net(mental_state.view(-1, self.mental_state_dim)).view(
                        self.batch_size, self.num_particles, self.num_encoding_bins
                    )
                )
            )

        return observation_state_random_variable


class FlyProposalNetwork(aesmc.model.ProposalNetwork):
    def __init__(self, num_flies, num_actions, mental_state_dim, num_encoding_bins):
        self.num_flies = num_flies
        self.num_actions = num_actions
        self.mental_state_dim = mental_state_dim
        self.num_encoding_bins = num_encoding_bins
        self.position_var = Variable(torch.ones(2))

        # Neural nets for time = 0
        hidden_layer_dim = 200
        self.initial_action_mlp = nn.Sequential(
            nn.Linear(2 + self.num_encoding_bins, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, self.num_actions),
            nn.Softmax(dim=1)
        )
        self.initial_mental_state_mean_mlp = nn.Sequential(
            nn.Linear(2 + self.num_encoding_bins, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, hidden_layer_dim)
        )
        self.initial_mental_state_var_mlp = nn.Sequential(
            nn.Linear(2 + self.num_encoding_bins, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, hidden_layer_dim),
            nn.Softplus()
        )

        # Neural nets for time > 0
        self.action_mlp = nn.Sequential(
            nn.Linear(self.num_actions + 2 + self.mental_state_dim + 2 + self.num_encoding_bins, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, self.num_actions),
            nn.Softmax(dim=1)
        )
        self.mental_state_mean_mlp = nn.Sequential(
            nn.Linear(self.num_actions + 2 + self.mental_state_dim + 2 + self.num_encoding_bins, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, self.mental_state_dim)
        )
        self.mental_state_var_mlp = nn.Sequential(
            nn.Linear(self.num_actions + 2 + self.mental_state_dim + 2 + self.num_encoding_bins, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, self.mental_state_dim),
            nn.Softplus()
        )

    def proposal(
        self, previous_latent_state=None, time=None, observation_states=None
    ):
        latent_state_random_variable = rv.StateRandomVariable()
        for f in range(self.num_flies):
            observed_position = observation_states[time]._values['observed_position_{}'.format(f)]
            observed_encoding = observation_states[time]._values['observed_encoding_{}'.format(f)]

            if time == 0:
                # Action
                latent_state_random_variable.set_random_variable_(
                    'action_{}'.format(f),
                    rv.Categorical(self.initial_action_mlp(
                        torch.cat([
                            observation_position.view(-1, 2),
                            observation_encoding.view(-1, self.num_encoding_bins)
                        ], dim=1)
                    ).view(self.batch_size, self.num_particles, self.num_actions))
                )

                # Position
                latent_state_random_variable.set_random_variable_(
                    'position_{}'.format(f),
                    rv.MultivariateIndependentNormal(
                        mean=observed_position, # TODO: should we do something else here?
                        variance=self.position_var.unsqueeze(0).unsqueeze(0).expand(
                            self.batch_size, self.num_particles, 2
                        )
                    )
                )

                # Mental State
                latent_state_random_variable.set_random_variable_(
                    'mental_state_{}'.format(f),
                    rv.MultivariateIndependentNormal(
                        mean=self.initial_mental_state_mean_mlp(
                            torch.cat([
                                observation_position.view(-1, 2),
                                observation_encoding.view(-1, self.num_encoding_bins)
                            ], dim=1)
                        ).view(self.batch_size, self.num_particles, -1),
                        variance=self.initial_mental_state_var_mlp(
                            torch.cat([
                                observation_position.view(-1, 2),
                                observation_encoding.view(-1, self.num_encoding_bins)
                            ], dim=1)
                        ).view(self.batch_size, self.num_particles, -1)
                    )
                )
            else:
                previous_action = previous_latent_state._values['action_{}'.format(f)]
                previous_position = previous_latent_state._values['position_{}'.format(f)]
                previous_mental_state = previous_latent_state._values['mental_state_{}'.format(f)]

                # Action
                latent_state_random_variable.set_random_variable_(
                    'action_{}'.format(f),
                    rv.Categorical(self.action_mlp(
                        torch.cat([
                            one_hot(previous_action.view(-1), self.num_actions),
                            previous_position.view(-1, 2),
                            previous_mental_state.view(-1, self.mental_state_dim),
                            observation_position.view(-1, 2),
                            observation_encoding.view(-1, self.num_encoding_bins)
                        ], dim=1)
                    ).view(self.batch_size, self.num_particles, self.num_actions))
                )

                # Position
                latent_state_random_variable.set_random_variable_(
                    'position_{}'.format(f),
                    rv.MultivariateIndependentNormal(
                        mean=observed_position, # TODO: should we do something else here?
                        variance=self.position_var.unsqueeze(0).unsqueeze(0).expand(
                            self.batch_size, self.num_particles, 2
                        )
                    )
                )

                # Mental State
                latent_state_random_variable.set_random_variable_(
                    'mental_state_{}'.format(f),
                    rv.MultivariateIndependentNormal(
                        mean=self.mental_state_mean_mlp(
                            torch.cat([
                                one_hot(previous_action.view(-1), self.num_actions),
                                previous_position.view(-1, 2),
                                previous_mental_state.view(-1, self.mental_state_dim),
                                observation_position.view(-1, 2),
                                observation_encoding.view(-1, self.num_encoding_bins)
                            ], dim=1)
                        ).view(self.batch_size, self.num_particles, -1),
                        variance=self.mental_state_var_mlp(
                            torch.cat([
                                one_hot(previous_action.view(-1), self.num_actions),
                                previous_position.view(-1, 2),
                                previous_mental_state.view(-1, self.mental_state_dim),
                                observation_position.view(-1, 2),
                                observation_encoding.view(-1, self.num_encoding_bins)
                            ], dim=1)
                        ).view(self.batch_size, self.num_particles, -1)
                    )
                )
        return latent_state_random_variable
