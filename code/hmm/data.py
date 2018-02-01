from util import *
import torch.utils.data


class HMMDataset(torch.utils.data.Dataset):
    def __init__(self, num_timesteps, filename, fake_num_data=1024):
        self.initial_probs, self.transition_probs, self.obs_means, self.obs_vars = read_model(filename)
        self.fake_num_data = fake_num_data
        self.num_timesteps = num_timesteps

    def __len__(self):
        return self.fake_num_data

    def __getitem__(self, idx):
        _, observations = generate_from_prior(
            1, self.num_timesteps, self.initial_probs, self.transition_probs, self.obs_means, self.obs_vars
        )
        observation_states = []
        for time in range(self.num_timesteps):
            observation_states.append(torch.Tensor([observations[0][time]]))
        return observation_states
