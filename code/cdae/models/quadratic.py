import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data
import numpy as np

from .. import distributions as dists
from .. import util as util

##########################
# True generative model
##########################


class GenerativeModel():
    """
    The true generative model $p$ over latents $k, x$ and observed variables $y$ is:
        \begin{align}
            k &\sim \mathrm{Categorical}([2, 5], [0.5, 0.5]) \\\
            x &\sim \mathrm{Normal}(0, 1) \\\
            y &\sim \mathrm{Normal}(f(k, x) ,1)
        \end{align}
        where $f: \mathbb R^2 \to \mathbb R$ is defined as
        \begin{align}
            f(k, x) = a(k + x)^2 + b(k + x) + c.
        \end{align}
    """

    def __init__(self, a=1, b=0, c=0):
        """
        Initialize generative model with a, b, c.
        """
        self.a = a
        self.b = b
        self.c = c

    def f(self, k, x):
        """
        input:
            k: Tensor [dim_1, ..., dim_N]
            x: Tensor [dim_1, ..., dim_N]

        output:
            f(k, x): Tensor [dim_1, ..., dim_N]
        """
        return self.a * (k + x)**2 + self.b * (k + x) + self.c

    def sample(self, batch_size):
        """
        Returns sample from the generative model.

        input:
            batch_size: int

        output:
            k: Tensor [batch_size, 1]
            x: Tensor [batch_size, 1]
            y: Tensor [batch_size, 1]
        """

        k = dists.categorical_sample(
            categories=torch.Tensor([2, 5]).unsqueeze(-1).unsqueeze(-1).expand(2, batch_size, 1),
            probabilities=torch.Tensor([0.5, 0.5])
            .unsqueeze(-1).unsqueeze(-1).expand(2, batch_size, 1)
        )
        x = dists.normal_sample(
            mean=torch.zeros(batch_size, 1),
            var=torch.ones(batch_size, 1)
        )

        mean = self.f(k, x)
        var = torch.ones(mean.size())
        y = dists.normal_sample(
            mean=mean,
            var=var
        )

        return k, x, y

    def importance_sample(self, y, num_particles, resample=True, num_resample=None):
        """
        Importance sampling.

        input:
            y: float
            num_particles: float
            resample (optional): bool, resample particles at the end
            num_resample (optional): int, how many times to resample (if not given, num_resample
            will be set to num_particles)

        output:
            if resample==True:
                k: Tensor [num_resample]
                x: Tensor [num_resample]
            else:
                k: Tensor [num_particles]
                x: Tensor [num_particles]
                unnormalized_weights: Tensor [num_particles]
        """

        y_expanded = torch.Tensor([y]).expand(num_particles)
        k, x, _ = self.sample(num_particles)
        k = k.squeeze(1)
        x = x.squeeze(1)

        unnormalized_weights = torch.exp(dists.normal_logpdf(
            y_expanded,
            self.f(k, x),
            torch.ones(num_particles)
        ))

        if resample:
            k_and_x = torch.cat([k.unsqueeze(0), x.unsqueeze(0)])
            normalized_weights = unnormalized_weights / torch.sum(unnormalized_weights)
            resampled_particle_indices = dists.categorical_sample(
                torch.arange(0, num_particles)
                .long().unsqueeze(-1).expand(num_particles, num_resample),
                normalized_weights.unsqueeze(-1).expand(num_particles, num_resample)
            )
            k_and_x_resampled = torch.gather(
                k_and_x,
                dim=1,
                index=resampled_particle_indices.unsqueeze(0).expand(2, num_resample).long()
            )
            return k_and_x_resampled[0], k_and_x_resampled[1]
        else:
            return k, x, unnormalized_weights


##########################
# Generative network
##########################


class GenerativeNetwork(nn.Module):
    """
    Assume that we actually don't know the form of $f$ and that we want to learn it, i.e. the true
    model $p$ from a dataset $(y^{(n)})_{n = 1}^N$.

    Let's model the family of functions $f$ under consideration as a neural network parameterized
    by generative weights $\theta$ such that it maps from $\mathbb R^2$ to $\mathbb R$.
    """
    def __init__(self):
        '''
        Initialize generative network.
        '''
        super(GenerativeNetwork, self).__init__()
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.c = nn.Parameter(torch.randn(1))

    def f_approx(self, k, x):
        '''
        Returns output of current approximation of f.

        input:
            k: Variable [batch_size, 1]
            x: Variable [batch_size, 1]

        output: Variable [batch_size, 1]
        '''

        a_expanded = self.a.unsqueeze(0).expand_as(k)
        b_expanded = self.b.unsqueeze(0).expand_as(k)
        c_expanded = self.c.unsqueeze(0).expand_as(k)

        return a_expanded * (k + x)**2 + b_expanded * (k + x) + c_expanded

    def forward(self, k, x, y):
        '''
        Returns log p_{\theta}(k, x, y)

        input:
            k: Variable [batch_size, 1]
            x: Variable [batch_size, 1]
            y: Variable [batch_size, 1]

        output: Variable [batch_size, 1]
        '''

        batch_size = k.size(0)

        logpdf_k = dists.categorical_logpdf(
            k,
            categories=Variable(
                torch.Tensor([2, 5]).unsqueeze(-1).unsqueeze(-1).expand(2, batch_size, 1)
            ),
            probabilities=Variable(
                torch.Tensor([0.5, 0.5]).unsqueeze(-1).unsqueeze(-1).expand(2, batch_size, 1)
            )
        )

        logpdf_x = dists.normal_logpdf(
            x,
            Variable(torch.zeros(x.size())), Variable(torch.ones(x.size()))
        )

        mean = self.f_approx(k, x)
        var = Variable(torch.ones(mean.size()))
        logpdf_y = dists.normal_logpdf(y, mean, var)

        return logpdf_k + logpdf_x + logpdf_y

    def sample(self, batch_size):
        '''
        Returns sample from the generative model.

        input:
            batch_size: int

        output:
            k: Tensor [batch_size, 1]
            x: Tensor [batch_size, 1]
            y: Tensor [batch_size, 1]
        '''

        k = dists.categorical_sample(
            categories=torch.Tensor([2, 5]).unsqueeze(-1).unsqueeze(-1).expand(2, batch_size, 1),
            probabilities=torch.Tensor([0.5, 0.5])
            .unsqueeze(-1).unsqueeze(-1).expand(2, batch_size, 1)
        )
        x = dists.normal_sample(
            mean=torch.zeros(batch_size, 1),
            var=torch.ones(batch_size, 1)
        )

        mean = self.f_approx(Variable(k, volatile=True), Variable(x, volatile=True)).data
        var = torch.ones(mean.size())
        y = dists.normal_sample(
            mean=mean,
            var=var
        )

        return k, x, y


##########################
# Inference network
##########################


class InferenceNetwork(nn.Module):
    """We seek to learn an inference network $q_{\phi}(k, x \lvert y)$ parameterized by $\phi$

    which, given $y$ maps to the parameters of the distribution over $(k, x)$, ideally close to the
    posterior under the true model, $p(k, x \lvert y)$.

    Let
    \begin{align}
        q_{\phi}(k, x \lvert y) &= q_{\phi}(k \lvert y) q_{\phi}(x \lvert k, y) \\\
        q_{\phi}(k \lvert y) &= \mathrm{Categorical}([2, 5], [\phi_1, \phi_2]) \\\
        q_{\phi}(x \lvert k, y) &= \mathrm{Normal}(\phi_3, \phi_4)
    \end{align}
    where $\phi = [\phi_1, \dotsc, \phi_4]$ is the output of the inference network.
    """
    def __init__(self):
        '''
        Initialize inference network.
        '''

        super(InferenceNetwork, self).__init__()
        self.k_lin1 = nn.Linear(1, 16)
        self.k_lin2 = nn.Linear(16, 2)

        self.x_mean_lin1 = nn.Linear(2, 16)
        self.x_mean_lin2 = nn.Linear(16, 1)

        self.x_var_lin1 = nn.Linear(2, 16)
        self.x_var_lin2 = nn.Linear(16, 1)

        init.xavier_uniform(self.k_lin1.weight, gain=init.calculate_gain('relu'))
        init.xavier_uniform(self.k_lin2.weight)
        init.xavier_uniform(self.x_mean_lin1.weight, gain=init.calculate_gain('relu'))
        init.xavier_uniform(self.x_mean_lin2.weight)
        init.xavier_uniform(self.x_var_lin1.weight, gain=init.calculate_gain('relu'))
        init.xavier_uniform(self.x_var_lin2.weight)

    def get_q_k_params(self, y):
        '''
        Returns parameters \phi_1, \phi_2.

        input:
            y: Variable [batch_size, 1]

        output: Variable [batch_size, 2]
        '''

        ret = self.k_lin1(y)
        ret = F.relu(ret)
        ret = self.k_lin2(ret)
        ret = F.softmax(ret) + util.epsilon
        ret = ret / torch.sum(ret, dim=1).expand_as(ret)

        return ret

    def get_q_x_params(self, k, y):
        '''
        Returns parameters \phi_3, \phi_4.

        input:
            k: Variable [batch_size, 1]
            y: Variable [batch_size, 1]

        output:
            mean: Variable [batch_size, 1]
            var: Variable [batch_size, 1]
        '''

        mean = self.x_mean_lin1(torch.cat([k, y], dim=1))
        mean = F.relu(mean)
        mean = self.x_mean_lin2(mean)

        var = self.x_var_lin1(torch.cat([k, y], dim=1))
        var = F.relu(var)
        var = self.x_var_lin2(var)
        var = F.softplus(var) + util.epsilon

        return mean, var

    def forward(self, k, x, y):
        '''
        Returns log q_{\phi}(k, x | y)

        input:
            k: Variable [batch_size, 1]
            x: Variable [batch_size, 1]
            y: Variable [batch_size, 1]

        output: Variable [batch_size, 1]
        '''
        batch_size, _ = k.size()

        probabilities = self.get_q_k_params(y)
        logpdf_k = dists.categorical_logpdf(
            k,
            categories=Variable(
                torch.Tensor([2, 5]).unsqueeze(-1).unsqueeze(-1).expand(2, batch_size, 1)
            ),
            probabilities=torch.t(probabilities).unsqueeze(-1)
        )

        mean, var = self.get_q_x_params(k, y)
        logpdf_x = dists.normal_logpdf(
            x,
            mean=mean,
            var=var
        )

        return logpdf_k + logpdf_x

    def sample(self, y):
        '''
        Returns samples from q_{\phi}(k, x | y)

        input:
            y: Tensor [batch_size, 1]

        output:
            k: Tensor [batch_size, 1]
            x: Tensor [batch_size, 1]
        '''

        batch_size = y.size(0)

        probabilities = self.get_q_k_params(Variable(y, volatile=True)).data
        k = dists.categorical_sample(
            categories=torch.Tensor([2, 5]).unsqueeze(-1).unsqueeze(-1).expand(2, batch_size, 1),
            probabilities=torch.t(probabilities).unsqueeze(-1)
        )

        mean, var = self.get_q_x_params(Variable(k, volatile=True), Variable(y, volatile=True))
        x = dists.normal_sample(
            mean=mean.data,
            var=var.data
        )

        return k, x


##########################
# Dataset
##########################


class Dataset(torch.utils.data.Dataset):
    def __init__(self, infinite_data=None, num_data=None, data_generator=None):
        '''
        Initializes quadratic Dataset. If infinite_data is True, generates data on the fly,
        otherwise generates data once at the start.

        input:
            infinite_data: bool. If True, supply fake_num_data and data_generator, otherwise supply
            data
            num_data: number. In the case of infinite_data, this forms as a fake num_data in order
            to be able to talk about "epochs".
            data_generator: function that generates a sample from the true generative model
        '''
        assert(type(infinite_data) is bool)
        assert(type(num_data) is int)
        assert(callable(data_generator))

        self.infinite_data = infinite_data
        if infinite_data:
            self.num_data = num_data
            self.data_generator = lambda: np.float32(data_generator())
        else:
            self.num_data = num_data
            self.data = np.array([np.float32(data_generator()) for i in range(num_data)])

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        if self.infinite_data:
            return self.data_generator()
        else:
            return self.data[index]
