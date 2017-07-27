import torch.nn as nn
import torch.utils.data


class GenerativeModel():
    """
    An abstract class representing a generative model.
    """

    def sample(self, batch_size):
        """
        Returns sample from the generative model.

        input:
            batch_size: int

        output:
            latents: list of Tensor [batch_size, ...]
            observes: list of Tensor [batch_size, ...]
        """

        raise NotImplementedError


class GenerativeModelDataset(torch.utils.data.Dataset):
    def __init__(self, generative_model, infinite_data=False, num_data=None):
        """
        Initializes GenerativeModelDataset. If infinite_data is True, generates data on the fly,
        otherwise generates data once at the start.

        input:
            generative_model: instance of GenerativeModel
            infinite_data: bool. If True, supply fake_num_data and data_generator, otherwise supply
            data (optional, default: False)
            num_data: number. In the case of infinite_data, this forms as a fake num_data in order
            to have "epochs".
        """
        assert(type(infinite_data) is bool)
        assert(type(num_data) is int)

        self.infinite_data = infinite_data
        if infinite_data:
            self.num_data = num_data
            self.generative_model = generative_model
        else:
            self.num_data = num_data
            _, self.observes = generative_model.sample(num_data)

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        if self.infinite_data:
            return list(map(
                lambda observe: torch.squeeze(observe, dim=0),
                self.generative_model.sample(batch_size=1)[1]
            ))
        else:
            return list(map(
                lambda observe: observe[0],
                self.observes
            ))


class GenerativeNetwork(nn.Module):
    """
    An abstract class representing a generative network.
    """

    def __init__(self):
        super(GenerativeNetwork, self).__init__()

    def sample(self, batch_size):
        """
        Returns sample from the generative network.

        input:
            batch_size: int

        output:
            latents: list of Tensor [batch_size, ...]
            observes: list of Tensor [batch_size, ...]
        """

        raise NotImplementedError

    def forward(self, *latents_and_observes):
        """
        Returns log p_{\theta}(latents, observes)

        input:
            *latents_and_observes: list of Variable [batch_size, ...]

        output: Variable [batch_size]
        """

        raise NotImplementedError


class InferenceNetwork(nn.Module):
    """
    An abstract class representing an inference network.
    """

    def __init__(self):
        super(InferenceNetwork, self).__init__()

    def sample(self, *observes):
        """
        Returns samples from the inference network.

        input:
            observes: list of Tensor [batch_size, ...]

        output:
            latents: list of Tensor [batch_size, ...]
        """

        raise NotImplementedError

    def forward(self, *latents_and_observes):
        """
        Returns log q_{\phi}(latents | observes)

        input:
            *latents_and_observes: list of Variable [batch_size, ...]

        output: Variable [batch_size]
        """

        raise NotImplementedError
