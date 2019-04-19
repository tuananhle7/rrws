import datetime
import os
import pickle
import uuid
import torch
import numpy as np
import models


def lognormexp(values, dim=0):
    """Exponentiates, normalizes and takes log of a tensor.

    Args:
        values: tensor [dim_1, ..., dim_N]
        dim: n

    Returns:
        result: tensor [dim_1, ..., dim_N]
            where result[i_1, ..., i_N] =
                                 exp(values[i_1, ..., i_N])
            log( ------------------------------------------------------------ )
                    sum_{j = 1}^{dim_n} exp(values[i_1, ..., j, ..., i_N])
    """

    log_denominator = torch.logsumexp(values, dim=dim, keepdim=True)
    # log_numerator = values
    return values - log_denominator


def exponentiate_and_normalize(values, dim=0):
    """Exponentiates and normalizes a tensor.

    Args:
        values: tensor [dim_1, ..., dim_N]
        dim: n

    Returns:
        result: tensor [dim_1, ..., dim_N]
            where result[i_1, ..., i_N] =
                            exp(values[i_1, ..., i_N])
            ------------------------------------------------------------
             sum_{j = 1}^{dim_n} exp(values[i_1, ..., j, ..., i_N])
    """

    return torch.exp(lognormexp(values, dim=dim))


def range_except(end, i):
    """Outputs an increasing list from 0 to (end - 1) except i.
    Args:
        end: int
        i: int

    Returns: list of length (end - 1)
    """

    result = list(set(range(end)))
    return result[:i] + result[(i + 1):]


def get_yyyymmdd():
    return str(datetime.date.today()).replace('-', '')


def get_hhmmss():
    return datetime.datetime.now().strftime('%H:%M:%S')


def print_with_time(str):
    print(get_yyyymmdd() + ' ' + get_hhmmss() + ' ' + str)


# https://stackoverflow.com/questions/4529815/saving-an-object-data-persistence
def save_object(obj, path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(path, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    print_with_time('Saved to {}'.format(path))


def load_object(path):
    with open(path, 'rb') as input_:
        obj = pickle.load(input_)
    return obj


def get_stats_path(model_folder='.'):
    return os.path.join(model_folder, 'stats.pkl')


def get_uuid():
    return str(uuid.uuid4())[:8]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_args_path(model_folder='.'):
    return os.path.join(model_folder, 'args.pkl')


def get_model_folder(rootdir='./models/'):
    return os.path.join(rootdir, get_yyyymmdd() + '_' + get_uuid())


def save_models(generative_model, inference_network, model_folder='.',
                iteration=None):
    if iteration is None:
        suffix = ''
    else:
        suffix = iteration
    generative_model_path = os.path.join(model_folder,
                                         'gen{}.pt'.format(suffix))
    inference_network_path = os.path.join(model_folder,
                                          'inf{}.pt'.format(suffix))
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(generative_model.state_dict(), generative_model_path)
    print_with_time('Saved to {}'.format(generative_model_path))
    torch.save(inference_network.state_dict(), inference_network_path)
    print_with_time('Saved to {}'.format(inference_network_path))


def load_models(model_folder='.', iteration=None):
    """Returns: generative_model, inference network
    """
    if iteration is None:
        suffix = ''
    else:
        suffix = iteration
    generative_model_path = os.path.join(model_folder,
                                         'gen{}.pt'.format(suffix))
    inference_network_path = os.path.join(model_folder,
                                          'inf{}.pt'.format(suffix))
    if os.path.exists(generative_model_path):
        args = load_object(get_args_path(model_folder))

        generative_model = models.GenerativeModel(
            args.init_mixture_logits,
            softmax_multiplier=args.softmax_multiplier, device=args.device
        ).to(device=args.device)
        inference_network = models.InferenceNetwork(
            args.num_mixtures, args.relaxed_one_hot, args.temperature,
            args.device).to(device=args.device)
        generative_model.load_state_dict(torch.load(generative_model_path))
        print_with_time('Loaded from {}'.format(generative_model_path))
        inference_network.load_state_dict(torch.load(inference_network_path))
        print_with_time('Loaded from {}'.format(inference_network_path))

        return generative_model, inference_network
    else:
        return None, None


def save_control_variate(control_variate, model_folder='.', iteration=None):
    if iteration is None:
        suffix = ''
    else:
        suffix = iteration
    path = os.path.join(model_folder, 'c{}.pt'.format(suffix))
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(control_variate.state_dict(), path)
    print_with_time('Saved to {}'.format(path))


def load_control_variate(model_folder='.', iteration=None):
    if iteration is None:
        suffix = ''
    else:
        suffix = iteration
    path = os.path.join(model_folder, 'c{}.pt'.format(suffix))
    args = load_object(get_args_path(model_folder))
    control_variate = models.ControlVariate(args.num_mixtures)
    control_variate.load_state_dict(torch.load(path))
    print_with_time('Loaded from {}'.format(path))


def init_models(args):
    """Returns: generative_model, inference_network, true_generative_model"""

    generative_model = models.GenerativeModel(
        args.init_mixture_logits, softmax_multiplier=args.softmax_multiplier,
        device=args.device).to(device=args.device)
    inference_network = models.InferenceNetwork(
        args.num_mixtures, args.relaxed_one_hot, args.temperature,
        args.device).to(device=args.device)
    true_generative_model = models.GenerativeModel(
        args.true_mixture_logits, softmax_multiplier=args.softmax_multiplier,
        device=args.device).to(device=args.device)

    return generative_model, inference_network, true_generative_model


def sample_relax(inference_network, control_variate, obs, num_particles,
                 epsilon=1e-6):
    """This implements Appendix C in the REBAR paper.

    Args:
        inference_network:
        control_variate:
        obs: tensor of shape [batch_size]
        num_particles: int

    Returns:
        latent, latent_aux, latent_aux_tilde: tensors of shape
            [batch_size, num_particles, num_mixtures]
    """
    batch_size = len(obs)
    num_mixtures = inference_network.num_mixtures
    probs = inference_network.get_latent_params(obs)
    probs_expanded = probs.unsqueeze(1).expand(
        batch_size, num_particles, num_mixtures).contiguous().view(
        batch_size * num_particles, num_mixtures)

    # latent_aux
    u = torch.distributions.Uniform(0 + epsilon, 1 - epsilon).sample(
        sample_shape=(batch_size * num_particles, num_mixtures))
    latent_aux = torch.log(probs_expanded) - torch.log(-torch.log(u))

    # latent
    latent = torch.zeros(batch_size * num_particles, num_mixtures)
    k = torch.argmax(latent_aux, dim=1)
    arange = torch.arange(batch_size * num_particles).long()
    latent[arange, k] = 1

    # latent_aux_tilde
    v = torch.distributions.Uniform(0 + epsilon, 1 - epsilon).sample(
        sample_shape=(batch_size * num_particles, num_mixtures))
    latent_aux_tilde = torch.zeros(batch_size * num_particles, num_mixtures)
    latent_aux_tilde[latent.byte()] = -torch.log(-torch.log(v[latent.byte()]))
    latent_aux_tilde[1 - latent.byte()] = -torch.log(
        -torch.log(v[1 - latent.byte()]) / probs_expanded[1 - latent.byte()] -
        torch.log(v[latent.byte()])
        .unsqueeze(-1).expand(-1, num_mixtures - 1).contiguous().view(-1))
    return [x.view(batch_size, num_particles, num_mixtures)
            for x in [latent, latent_aux, latent_aux_tilde]]


def logaddexp(a, b):
    """Returns log(exp(a) + exp(b))."""

    return torch.logsumexp(torch.cat([a.unsqueeze(0), b.unsqueeze(0)]), dim=0)


def get_p_error(true_generative_model, generative_model):
    return torch.norm(generative_model.get_latent_params() -
                      true_generative_model.get_latent_params()).item()


def get_q_error(true_generative_model, inference_network, obs):
    p_probs = true_generative_model.get_posterior_probs(obs)
    q_probs = inference_network.get_latent_params(obs)
    return torch.mean(torch.norm(p_probs - q_probs, p=2, dim=1)).item()


class OnlineMeanStd():
    def __init__(self):
        self.count = 0
        self.means = None
        self.M2s = None

    def update(self, new_variables):
        if self.count == 0:
            self.count = 1
            self.means = []
            self.M2s = []
            for new_var in new_variables:
                self.means.append(new_var.data)
                self.M2s.append(new_var.data.new(new_var.size()).fill_(0))
        else:
            self.count = self.count + 1
            for new_var_idx, new_var in enumerate(new_variables):
                delta = new_var.data - self.means[new_var_idx]
                self.means[new_var_idx] = self.means[new_var_idx] + delta / \
                    self.count
                delta_2 = new_var.data - self.means[new_var_idx]
                self.M2s[new_var_idx] = self.M2s[new_var_idx] + delta * delta_2

    def means_stds(self):
        if self.count < 2:
            raise ArithmeticError('Need more than 1 value. Have {}'.format(
                self.count))
        else:
            stds = []
            for i in range(len(self.means)):
                stds.append(torch.sqrt(self.M2s[i] / self.count))
            return self.means, stds

    def avg_of_means_stds(self):
        means, stds = self.means_stds()
        num_parameters = np.sum([len(p) for p in means])
        return (np.sum([torch.sum(p) for p in means]) / num_parameters,
                np.sum([torch.sum(p) for p in stds]) / num_parameters)


def args_match(model_folder, **kwargs):
    """Do training args match kwargs?"""

    args_filename = get_args_path(model_folder)
    if os.path.exists(args_filename):
        args = load_object(args_filename)
        for k, v in kwargs.items():
            if args.__dict__[k] != v:
                return False
        return True
    else:
        return False


def list_subdirs(rootdir):
    for file in os.listdir(rootdir):
        path = os.path.join(rootdir, file)
        if os.path.isdir(path):
            yield(path)


def list_model_folders_args_match(rootdir='./models/', **kwargs):
    """Return a list of model folders whose training args
    match kwargs.
    """

    result = []
    for model_folder in list_subdirs(rootdir):
        if args_match(model_folder, **kwargs):
            result.append(model_folder)
    return result


def get_most_recent_model_folder_args_match(**kwargs):
    model_folders = list_model_folders_args_match(**kwargs)
    if len(model_folders) > 0:
        return model_folders[np.argmax(
            [os.stat(x).st_mtime for x in model_folders])]


class OnlineMeanStd():
    def __init__(self):
        self.count = 0
        self.means = None
        self.M2s = None

    def update(self, new_variables):
        if self.count == 0:
            self.count = 1
            self.means = []
            self.M2s = []
            for new_var in new_variables:
                self.means.append(new_var.data)
                self.M2s.append(new_var.data.new(new_var.size()).fill_(0))
        else:
            self.count = self.count + 1
            for new_var_idx, new_var in enumerate(new_variables):
                delta = new_var.data - self.means[new_var_idx]
                self.means[new_var_idx] = self.means[new_var_idx] + delta / \
                    self.count
                delta_2 = new_var.data - self.means[new_var_idx]
                self.M2s[new_var_idx] = self.M2s[new_var_idx] + delta * delta_2

    def means_stds(self):
        if self.count < 2:
            raise ArithmeticError('Need more than 1 value. Have {}'.format(
                self.count))
        else:
            stds = []
            for i in range(len(self.means)):
                stds.append(torch.sqrt(self.M2s[i] / self.count))
            return self.means, stds

    def avg_of_means_stds(self):
        means, stds = self.means_stds()
        num_parameters = np.sum([len(p) for p in means])
        return (np.sum([torch.sum(p) for p in means]) / num_parameters,
                np.sum([torch.sum(p) for p in stds]) / num_parameters)
