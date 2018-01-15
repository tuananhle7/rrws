import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from . import model
from .. import distributions as dists
from .. import util as util
plt.style.use('seaborn-whitegrid')


##########################
# True generative model
##########################


class QuadraticGenerativeModel(model.GenerativeModel):
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
            [k, x]: [Tensor [batch_size, 1], Tensor [batch_size, 1]]
            [y]: [Tensor [batch_size, 1]]
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

        return [k, x], [y]

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
        [k, x], _ = self.sample(num_particles)
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


class QuadraticGenerativeNetworkSmall(model.GenerativeNetwork):
    """
    Assume that we actually don't know the form of $f$ and that we want to learn it, i.e. the true
    model $p$ from a dataset $(y^{(n)})_{n = 1}^N$.

    Let's model the family of functions $f$ under consideration as a neural network parameterized
    by generative weights $\theta$ such that it maps from $\mathbb R^2$ to $\mathbb R$.
    """
    def __init__(self):
        """
        Initialize generative network.
        """
        super(QuadraticGenerativeNetworkSmall, self).__init__()
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.c = nn.Parameter(torch.randn(1))

    def f_approx(self, k, x):
        """
        Returns output of current approximation of f.

        input:
            k: Variable [batch_size, 1]
            x: Variable [batch_size, 1]

        output: Variable [batch_size, 1]
        """

        a_expanded = self.a.unsqueeze(0).expand_as(k)
        b_expanded = self.b.unsqueeze(0).expand_as(k)
        c_expanded = self.c.unsqueeze(0).expand_as(k)

        return a_expanded * (k + x)**2 + b_expanded * (k + x) + c_expanded

    def forward(self, k, x, y):
        """
        Returns log p_{\theta}(k, x, y)

        input:
            k: Variable [batch_size, 1]
            x: Variable [batch_size, 1]
            y: Variable [batch_size, 1]

        output: Variable [batch_size]
        """

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

        return (logpdf_k + logpdf_x + logpdf_y).squeeze(1)

    def sample(self, batch_size):
        """
        Returns sample from the generative network.

        input:
            batch_size: int

        output:
            [k, x]: [Tensor [batch_size, 1], Tensor [batch_size, 1]]
            [y]: [Tensor [batch_size, 1]]
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

        mean = self.f_approx(Variable(k, volatile=True), Variable(x, volatile=True)).data
        var = torch.ones(mean.size())
        y = dists.normal_sample(
            mean=mean,
            var=var
        )

        return [k, x], [y]


class QuadraticGenerativeNetworkLarge(model.GenerativeNetwork):
    """
    Assume that we actually don't know the form of $f$ and that we want to learn it, i.e. the true
    model $p$ from a dataset $(y^{(n)})_{n = 1}^N$.

    Let's model the family of functions $f$ under consideration as a neural network parameterized
    by generative weights $\theta$ such that it maps from $\mathbb R^2$ to $\mathbb R$.
    """
    def __init__(self):
        """
        Initialize generative network.
        """

        super(QuadraticGenerativeNetworkLarge, self).__init__()
        self.lin1 = nn.Linear(2, 16)
        self.lin2 = nn.Linear(16, 1)

        init.xavier_uniform(self.lin1.weight, gain=init.calculate_gain('relu'))
        init.xavier_uniform(self.lin2.weight)

    def f_approx(self, k, x):
        """
        Returns output of current approximation of f.

        input:
            k: Variable [batch_size, 1]
            x: Variable [batch_size, 1]

        output: Variable [batch_size, 1]
        """

        ret = self.lin1(torch.cat([k, x], dim=1))
        ret = F.relu(ret)
        ret = self.lin2(ret)

        return ret

    def forward(self, k, x, y):
        """
        Returns log p_{\theta}(k, x, y)

        input:
            k: Variable [batch_size, 1]
            x: Variable [batch_size, 1]
            y: Variable [batch_size, 1]

        output: Variable [batch_size]
        """

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

        return (logpdf_k + logpdf_x + logpdf_y).squeeze(1)

    def sample(self, batch_size):
        """
        Returns sample from the generative network.

        input:
            batch_size: int

        output:
            [k, x]: [Tensor [batch_size, 1], Tensor [batch_size, 1]]
            [y]: [Tensor [batch_size, 1]]
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

        mean = self.f_approx(Variable(k, volatile=True), Variable(x, volatile=True)).data
        var = torch.ones(mean.size())
        y = dists.normal_sample(
            mean=mean,
            var=var
        )

        return [k, x], [y]


##########################
# Inference network
##########################


class QuadraticInferenceNetworkForwardDependence(model.InferenceNetwork):
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
        """
        Initialize inference network.
        """

        super(QuadraticInferenceNetworkForwardDependence, self).__init__()
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
        """
        Returns parameters \phi_1, \phi_2.

        input:
            y: Variable [batch_size, 1]

        output: Variable [batch_size, 2]
        """

        ret = self.k_lin1(y)
        ret = F.relu(ret)
        ret = self.k_lin2(ret)
        ret = F.softmax(ret)
        ret = ret / torch.sum(ret, dim=1).expand_as(ret)

        return ret

    def get_q_x_params(self, k, y):
        """
        Returns parameters \phi_3, \phi_4.

        input:
            k: Variable [batch_size, 1]
            y: Variable [batch_size, 1]

        output:
            mean: Variable [batch_size, 1]
            var: Variable [batch_size, 1]
        """

        mean = self.x_mean_lin1(torch.cat([k, y], dim=1))
        mean = F.relu(mean)
        mean = self.x_mean_lin2(mean)

        var = self.x_var_lin1(torch.cat([k, y], dim=1))
        var = F.relu(var)
        var = self.x_var_lin2(var)
        var = F.softplus(var)

        return mean, var

    def forward(self, k, x, y):
        """
        Returns log q_{\phi}(k, x | y)

        input:
            k: Variable [batch_size, 1]
            x: Variable [batch_size, 1]
            y: Variable [batch_size, 1]

        output: Variable [batch_size]
        """
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

        return (logpdf_k + logpdf_x).squeeze(1)

    def sample(self, y):
        """
        Returns samples from q_{\phi}(k, x | y)

        input:
            y: Tensor [batch_size, 1]

        output:
            [k, x]: [Tensor [batch_size, 1], Tensor [batch_size, 1]]
        """

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

        return [k, x]


class QuadraticInferenceNetworkReverseDependence(model.InferenceNetwork):
    """We seek to learn an inference network $q_{\phi}(k, x \lvert y)$ parameterized by $\phi$

    which, given $y$ maps to the parameters of the distribution over $(k, x)$, ideally close to the
    posterior under the true model, $p(k, x \lvert y)$.

    Let
    \begin{align}
        q_{\phi}(k, x \lvert y) &= q_{\phi}(k \lvert y) q_{\phi}(x \lvert k, y) \\\
        q_{\phi}(x \lvert y) &= \mathrm{Normal}(\phi_1, \phi_2) \\\
        q_{\phi}(k \lvert x, y) &= \mathrm{Categorical}([2, 5], [\phi_3, \phi_4])
    \end{align}
    where $\phi = [\phi_1, \dotsc, \phi_4]$ is the output of the inference network.
    """
    def __init__(self):
        """
        Initialize inference network.
        """

        super(QuadraticInferenceNetworkReverseDependence, self).__init__()
        self.x_mean_lin1 = nn.Linear(1, 16)
        self.x_mean_lin2 = nn.Linear(16, 1)

        self.x_var_lin1 = nn.Linear(1, 16)
        self.x_var_lin2 = nn.Linear(16, 1)

        self.k_lin1 = nn.Linear(2, 16)
        self.k_lin2 = nn.Linear(16, 2)

        init.xavier_uniform(self.k_lin1.weight, gain=init.calculate_gain('relu'))
        init.xavier_uniform(self.k_lin2.weight)
        init.xavier_uniform(self.x_mean_lin1.weight, gain=init.calculate_gain('relu'))
        init.xavier_uniform(self.x_mean_lin2.weight)
        init.xavier_uniform(self.x_var_lin1.weight, gain=init.calculate_gain('relu'))
        init.xavier_uniform(self.x_var_lin2.weight)

    def get_q_x_params(self, y):
        """
        Returns parameters \phi_1, \phi_2.

        input:
            y: Variable [batch_size, 1]

        output:
            mean: Variable [batch_size, 1]
            var: Variable [batch_size, 1]
        """

        mean = self.x_mean_lin1(y)
        mean = F.relu(mean)
        mean = self.x_mean_lin2(mean)

        var = self.x_var_lin1(y)
        var = F.relu(var)
        var = self.x_var_lin2(var)
        var = F.softplus(var)

        return mean, var

    def get_q_k_params(self, x, y):
        """
        Returns parameters \phi_1, \phi_2.

        input:
            x: Variable [batch_size, 1]
            y: Variable [batch_size, 1]

        output: Variable [batch_size, 2]
        """

        ret = self.k_lin1(torch.cat([x, y], dim=1))
        ret = F.relu(ret)
        ret = self.k_lin2(ret)
        ret = F.softmax(ret)
        ret = ret / torch.sum(ret, dim=1).expand_as(ret)

        return ret

    def forward(self, k, x, y):
        """
        Returns log q_{\phi}(k, x | y)

        input:
            k: Variable [batch_size, 1]
            x: Variable [batch_size, 1]
            y: Variable [batch_size, 1]

        output: Variable [batch_size]
        """
        batch_size, _ = k.size()

        mean, var = self.get_q_x_params(y)
        logpdf_x = dists.normal_logpdf(
            x,
            mean=mean,
            var=var
        )

        probabilities = self.get_q_k_params(x, y)
        logpdf_k = dists.categorical_logpdf(
            k,
            categories=Variable(
                torch.Tensor([2, 5]).unsqueeze(-1).unsqueeze(-1).expand(2, batch_size, 1)
            ),
            probabilities=torch.t(probabilities).unsqueeze(-1)
        )

        return (logpdf_k + logpdf_x).squeeze(1)

    def sample(self, y):
        """
        Returns samples from q_{\phi}(k, x | y)

        input:
            y: Tensor [batch_size, 1]

        output:
            [k, x]: [Tensor [batch_size, 1], Tensor [batch_size, 1]]
        """

        batch_size = y.size(0)

        mean, var = self.get_q_x_params(Variable(y, volatile=True))
        x = dists.normal_sample(
            mean=mean.data,
            var=var.data
        )

        probabilities = self.get_q_k_params(
            Variable(x, volatile=True),
            Variable(y, volatile=True)
        ).data
        k = dists.categorical_sample(
            categories=torch.Tensor([2, 5]).unsqueeze(-1).unsqueeze(-1).expand(2, batch_size, 1),
            probabilities=torch.t(probabilities).unsqueeze(-1)
        )

        return [k, x]


class QuadraticInferenceNetworkIndependent(model.InferenceNetwork):
    """We seek to learn an inference network $q_{\phi}(k, x \lvert y)$ parameterized by $\phi$

    which, given $y$ maps to the parameters of the distribution over $(k, x)$, ideally close to the
    posterior under the true model, $p(k, x \lvert y)$.

    Let
    \begin{align}
        q_{\phi}(k, x \lvert y) &= q_{\phi}(k \lvert y) q_{\phi}(x \lvert k, y) \\\
        q_{\phi}(k \lvert y) &= \mathrm{Categorical}([2, 5], [\phi_1, \phi_2]) \\\
        q_{\phi}(x \lvert y) &= \mathrm{Normal}(\phi_3, \phi_4)
    \end{align}
    where $\phi = [\phi_1, \dotsc, \phi_4]$ is the output of the inference network.
    """
    def __init__(self):
        """
        Initialize inference network.
        """

        super(QuadraticInferenceNetworkIndependent, self).__init__()
        self.k_lin1 = nn.Linear(1, 16)
        self.k_lin2 = nn.Linear(16, 2)

        self.x_mean_lin1 = nn.Linear(1, 16)
        self.x_mean_lin2 = nn.Linear(16, 1)

        self.x_var_lin1 = nn.Linear(1, 16)
        self.x_var_lin2 = nn.Linear(16, 1)

        init.xavier_uniform(self.k_lin1.weight, gain=init.calculate_gain('relu'))
        init.xavier_uniform(self.k_lin2.weight)
        init.xavier_uniform(self.x_mean_lin1.weight, gain=init.calculate_gain('relu'))
        init.xavier_uniform(self.x_mean_lin2.weight)
        init.xavier_uniform(self.x_var_lin1.weight, gain=init.calculate_gain('relu'))
        init.xavier_uniform(self.x_var_lin2.weight)

    def get_q_k_params(self, y):
        """
        Returns parameters \phi_1, \phi_2.

        input:
            y: Variable [batch_size, 1]

        output: Variable [batch_size, 2]
        """

        ret = self.k_lin1(y)
        ret = F.relu(ret)
        ret = self.k_lin2(ret)
        ret = F.softmax(ret)
        ret = ret / torch.sum(ret, dim=1).expand_as(ret)

        return ret

    def get_q_x_params(self, y):
        """
        Returns parameters \phi_3, \phi_4.

        input:
            y: Variable [batch_size, 1]

        output:
            mean: Variable [batch_size, 1]
            var: Variable [batch_size, 1]
        """

        mean = self.x_mean_lin1(y)
        mean = F.relu(mean)
        mean = self.x_mean_lin2(mean)

        var = self.x_var_lin1(y)
        var = F.relu(var)
        var = self.x_var_lin2(var)
        var = F.softplus(var)

        return mean, var

    def forward(self, k, x, y):
        """
        Returns log q_{\phi}(k, x | y)

        input:
            k: Variable [batch_size, 1]
            x: Variable [batch_size, 1]
            y: Variable [batch_size, 1]

        output: Variable [batch_size]
        """
        batch_size, _ = k.size()

        probabilities = self.get_q_k_params(y)
        logpdf_k = dists.categorical_logpdf(
            k,
            categories=Variable(
                torch.Tensor([2, 5]).unsqueeze(-1).unsqueeze(-1).expand(2, batch_size, 1)
            ),
            probabilities=torch.t(probabilities).unsqueeze(-1)
        )

        mean, var = self.get_q_x_params(y)
        logpdf_x = dists.normal_logpdf(
            x,
            mean=mean,
            var=var
        )

        return (logpdf_k + logpdf_x).squeeze(1)

    def sample(self, y):
        """
        Returns samples from q_{\phi}(k, x | y)

        input:
            y: Tensor [batch_size, 1]

        output:
            [k, x]: [Tensor [batch_size, 1], Tensor [batch_size, 1]]
        """

        batch_size = y.size(0)

        probabilities = self.get_q_k_params(Variable(y, volatile=True)).data
        k = dists.categorical_sample(
            categories=torch.Tensor([2, 5]).unsqueeze(-1).unsqueeze(-1).expand(2, batch_size, 1),
            probabilities=torch.t(probabilities).unsqueeze(-1)
        )

        mean, var = self.get_q_x_params(Variable(y, volatile=True))
        x = dists.normal_sample(
            mean=mean.data,
            var=var.data
        )

        return [k, x]


##########################
# Plotting scripts
##########################


def plot_quadratic_generative_comparison(
    quadratic_generative_model,
    quadratic_generative_network,
    num_data,
    filename
):
    fig, ax = plt.subplots(ncols=1, nrows=1)

    _, [y] = quadratic_generative_model.sample(num_data)
    y = y.squeeze(1).numpy()
    sns.kdeplot(y, ax=ax, label='true model $p(y)$')

    _, [y] = quadratic_generative_network.sample(num_data)
    y = y.squeeze(1).numpy()
    sns.kdeplot(y, ax=ax, label='learned model $p_{\\theta}(y)$')

    ax.set_ylabel('KDE')
    ax.set_xlabel('$y$')
    ax.set_title('Testing the Generative Model')

    fig.savefig(filename, bbox_inches='tight')


def plot_quadratic_comparison(
    quadratic_generative_model,
    quadratic_generative_network,
    filename
):
    fig, ax = plt.subplots(ncols=1, nrows=1)
    true_quadratic = quadratic_generative_model.f(
        torch.zeros(100),
        torch.linspace(-10, 10, 100)
    ).numpy()
    learned_quadratic = quadratic_generative_network.f_approx(
        Variable(torch.zeros(100).unsqueeze(-1)),
        Variable(torch.linspace(-10, 10, 100).unsqueeze(-1))
    ).data.numpy()

    ax.plot(np.linspace(-10, 10, 100), true_quadratic, label='true quadratic')
    ax.plot(np.linspace(-10, 10, 100), learned_quadratic, label='learned quadratic')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$ax^2 + bx + c$')
    ax.set_title('Quadratic')
    ax.legend()

    fig.savefig(filename, bbox_inches='tight')


def plot_quadratic_inference_comparison(
    quadratic_generative_model,
    quadratic_inference_network,
    num_inference_network_samples,
    num_importance_particles,
    y_test,
    filename
):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.set_size_inches(10, 10)
    fig.suptitle('Testing inference\n$y = {}$'.format(y_test))
    bar_width = 0.8

    ##############
    ##############
    ##############

    y = torch.Tensor([[y_test]]).expand(num_inference_network_samples, 1)
    [k, x] = quadratic_inference_network.sample(y)

    values = [2, 5]
    normalized_weights = [
        torch.sum(k == 2) / len(k),
        torch.sum(k == 5) / len(k),
    ]
    bar1 = ax[0][0].bar(
        np.array(values) - bar_width / 2,
        normalized_weights,
        width=bar_width,
        tick_label=values,
        color=colors[0],
        label='$q_{\phi}(k | y)$'
    )
    sns.kdeplot(x.view(-1).numpy(), ax=ax[0][1], label='$q_{\phi}(x | y)$', color=colors[0])

    if len(x[k == 2]) != 0:
        sns.kdeplot(
            x[k == 2].view(-1).numpy(),
            ax=ax[1][0],
            label='$q_{\phi}(x | y, k = 2)$',
            color=colors[0]
        )
    else:
        ax[1][0].axhline(0, label='$q_{\phi}(x | y, k = 2)$', color=colors[0])

    if len(x[k == 5]) != 0:
        sns.kdeplot(
            x[k == 5].view(-1).numpy(),
            ax=ax[1][1],
            label='$q_{\phi}(x | y, k = 5)$',
            color=colors[0]
        )
    else:
        ax[1][1].axhline(0, label='$q_{\phi}(x | y, k = 5)$', color=colors[0])

    ##############
    ##############
    ##############

    k, x = quadratic_generative_model.importance_sample(
        y_test,
        num_importance_particles,
        resample=True,
        num_resample=num_importance_particles
    )

    values = [2, 5]
    normalized_weights = [
        sum(k == 2) / len(k),
        sum(k == 5) / len(k),
    ]
    bar2 = ax[0][0].bar(
        np.array(values) + bar_width / 2,
        normalized_weights,
        tick_label=values,
        label='$p(k | y)$',
        color=colors[1]
    )

    sns.distplot(x.numpy(), hist=False, ax=ax[0][1], label='$p(x | y)$', color=colors[1])

    if normalized_weights[0] != 0:
        sns.distplot(
            x[k == 2].numpy(),
            hist=False,
            ax=ax[1][0],
            label='$p(x | y, k = 2)$',
            color=colors[1]
        )
    else:
        ax[1][0].axhline(0, label='$p(x | y, k = 2)$', color=colors[1])

    if normalized_weights[1] != 0:
        sns.distplot(
            x[k == 5].numpy(),
            hist=False,
            ax=ax[1][1],
            label='$p(x | y, k = 5)$',
            color=colors[1]
        )
    else:
        ax[1][1].axhline(0, label='$p(x | y, k = 5)$', color=colors[1])

    ##############
    ##############
    ##############

    ax[0][0].set_xticks(np.array([2, 5]))
    ax[0][0].xaxis.grid(False)
    ax[0][0].set_ylim([0, 1])
    ax[0][0].set_title('Histogram of $k | y$')
    ax[0][0].set_xlabel('$k$')
    ax[0][0].legend()

    ax[0][1].set_title('Kernel density estimate of $x | y$')
    ax[0][1].set_xlabel('$x$')

    ax[1][0].set_title('Kernel density estimate of $x | y, k = 2$')
    ax[1][0].set_xlabel('$x$')

    ax[1][1].set_title('Kernel density estimate of $x | y, k = 5$')
    ax[1][1].set_xlabel('$x$')
    ax[1][1].legend()

    fig.savefig(filename, bbox_inches='tight')
