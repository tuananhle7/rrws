import torch
import itertools
import json
import os
import models
import string
import Levenshtein
import losses
import pickle
import uuid
import datetime
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns


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


def production_probs_to_tensor(production_probs):
    """Convert production_probs in list to tensor.

    Args:
        production_probs: dict whose keys are non-terminals and values are
            probabilities of productions represented as list of shape
            [num_productions]

    Returns: same as production_probs but values are tensors instead of
    lists.
    """
    return {k: torch.tensor(v, dtype=torch.float)
            for k, v in production_probs.items()}


def one_hot(indices, num_bins):
    """Returns one hot vector given indices.

    Args:
        indices: tensors
        num_bins: number of bins

    Returns: matrix where ith row corresponds to a one
        hot version of indices[i].
    """
    return torch.zeros(len(indices), num_bins).scatter_(
        1, indices.long().unsqueeze(-1), 1)


def get_sample_address_embedding(non_terminal, non_terminals):
    """Returns an embedding of the sample address of a production.

    Args:
        non_terminal: string
        non_terminals: set of non_terminal symbols

    Returns: one-hot vector
    """
    num_bins = len(non_terminals)
    i = sorted(non_terminals).index(non_terminal)
    return one_hot(torch.tensor([i]), num_bins)[0]


def get_root(tree):
    """Returns root of a tree.

    Args: list of lists or string
    Returns: string
    """
    if isinstance(tree, list):
        return tree[0]
    else:
        return tree


def get_production_index(non_terminal, production, productions):
    """Args:
        non_terminal: string
        production: list of strings
        productions: dict where key is a non-terminal and value is a list of
            productions

    Returns: int
    """
    return productions[non_terminal].index(production)


def word_to_one_hot(word, terminals):
    """Convert word to its one-hot representation.

    Args:
        word: string
        terminals: set of terminal strings

    Returns: one hot tensor of shape [len(terminals)] or zeros if word is not
        in terminals
    """
    num_bins = len(terminals)
    try:
        i = sorted(terminals).index(word)
        return one_hot(torch.tensor([i]), num_bins)[0]
    except ValueError:
        return torch.zeros((num_bins,))


def sentence_to_one_hots(sentence, terminals):
    """Convert sentence to one-hots.

    Args:
        sentence: list of strings
        terminals: set of terminal strings

    Returns: matrix where ith row corresponds to a one-hot of ith word, shape
        [num_words, len(terminals)]
    """
    return torch.cat([word_to_one_hot(word, terminals).unsqueeze(0)
                      for word in sentence])


def get_leaves(tree):
    """Return leaves of a tree.

    Args: list of lists or string
    Returns: list of strings
    """
    if isinstance(tree, list):
        return list(itertools.chain.from_iterable(
            [get_leaves(subtree) for subtree in tree[1:]]))
    else:
        return [tree]


def read_pcfg(pcfg_path):
    with open(pcfg_path) as json_data:
        data = json.load(json_data)

    grammar = {
        'terminals': set(data['terminals']),
        'non_terminals': set(data['non_terminals']),
        'productions': data['productions'],
        'start_symbol': data['start_symbol'],
        'name': data['name']
    }
    true_production_probs = production_probs_to_tensor(
        data['production_probs'])

    return grammar, true_production_probs


def save_models(generative_model, inference_network, pcfg_path,
                model_folder='.'):
    generative_model_path = os.path.join(model_folder, 'gen.pt')
    inference_network_path = os.path.join(model_folder, 'inf.pt')
    pcfg_path_path = os.path.join(model_folder, 'pcfg_path.txt')

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(generative_model.state_dict(), generative_model_path)
    print_with_time('Saved to {}'.format(generative_model_path))
    torch.save(inference_network.state_dict(), inference_network_path)
    print_with_time('Saved to {}'.format(inference_network_path))
    with open(pcfg_path_path, 'w') as f:
        f.write(pcfg_path)
    print_with_time('Saved to {}'.format(pcfg_path_path))


def load_models(model_folder='.'):
    """Returns: generative_model, inference network
    """
    generative_model_path = os.path.join(model_folder, 'gen.pt')
    inference_network_path = os.path.join(model_folder, 'inf.pt')
    pcfg_path_path = os.path.join(model_folder, 'pcfg_path.txt')

    with open(pcfg_path_path) as f:
        pcfg_path = f.read()
    grammar, _ = read_pcfg(pcfg_path)
    generative_model = models.GenerativeModel(grammar)
    inference_network = models.InferenceNetwork(grammar)
    generative_model.load_state_dict(torch.load(generative_model_path))
    print_with_time('Loaded from {}'.format(generative_model_path))
    inference_network.load_state_dict(torch.load(inference_network_path))
    print_with_time('Loaded from {}'.format(inference_network_path))

    return generative_model, inference_network


def word_to_index(word, terminals):
    """Convert word to int.

    Args:
        word: string
        terminals: set of terminal strings

    Returns: int; -1 if word is not in terminals
    """

    try:
        return sorted(terminals).index(word)
    except ValueError:
        return -1


def sentence_to_indices(sentence, terminals):
    """Convert sentence to list of indices.

    Args:
        sentence: list of strings
        terminals: set of terminal strings

    Returns: list of indices of length len(sentence); index is -1 if word is
        not in terminals
    """
    return [word_to_index(word, terminals) for word in sentence]


def _indices_to_string(indices):
    return ''.join([string.printable[index] for index in indices])


def _sentence_to_string(sentence, terminals):
    return _indices_to_string(sentence_to_indices(sentence, terminals))


def get_levenshtein_distance(sentence_1, sentence_2, terminals):
    """Levenshtein distance between two sentences.

    Args:
        sentence_1: list of strings
        sentence_2: list of strings
        terminals: vocabulary; list of valid strings

    Returns: int"""
    return Levenshtein.distance(_sentence_to_string(sentence_1, terminals),
                                _sentence_to_string(sentence_2, terminals))


def init_models(pcfg_path):
    """Returns: generative_model, inference_network, true_generative_model"""

    grammar, true_production_probs = read_pcfg(pcfg_path)
    generative_model = models.GenerativeModel(grammar)
    inference_network = models.InferenceNetwork(grammar)
    true_generative_model = models.GenerativeModel(
        grammar, true_production_probs)

    return generative_model, inference_network, true_generative_model


def range_except(end, i):
    """Outputs an increasing list from 0 to (end - 1) except i.
    Args:
        end: int
        i: int

    Returns: list of length (end - 1)
    """

    result = list(set(range(end)))
    return result[:i] + result[(i + 1):]


def normalize(x, dim=0):
    return x / torch.sum(x, dim=0, keepdim=True)


def get_production_probs(generative_model):
    return {
        non_terminal: normalize(torch.exp(
            generative_model.production_logits[non_terminal])).detach()
        for non_terminal in generative_model.grammar['non_terminals']}


def get_kl(probs_1, probs_2):
    """KL between two probability tensors.

    Args:
        probs_1: probability tensor of shape [num_probs]
        probs_2: probability tensor of shape [num_probs]

    Returns: KL(p1 || p2), scalar tensor
    """
    return torch.sum(probs_1 * (torch.log(probs_1) - torch.log(probs_2)))


def get_q_error(generative_model, inference_network, num_samples=100):
    """Expected KL(posterior || q) + const as a measure of q's quality.

    Returns: detached scalar E_p(x)[KL(p(z | x) || q(z | x))] + H(z | x) where
        the second term is constant wrt the inference network.
    """

    return losses.get_sleep_loss(generative_model, inference_network,
                                 num_samples).detach()


def get_p_error(true_generative_model, generative_model):
    """Average KL between true and learned productions probs."""

    true_generative_model_probs = get_production_probs(true_generative_model)
    generative_model_probs = get_production_probs(generative_model)
    non_terminals = true_generative_model.grammar['non_terminals']
    result = 0
    for non_terminal in non_terminals:
        kl = get_kl(true_generative_model_probs[non_terminal],
                    generative_model_probs[non_terminal])
        result += kl
    return result / len(non_terminals)


# https://stackoverflow.com/questions/4529815/saving-an-object-data-persistence
def save_object(obj, filename):
    dir = os.path.dirname(filename)
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    print_with_time('Saved to {}'.format(filename))


def load_object(filename):
    with open(filename, 'rb') as input_:
        obj = pickle.load(input_)
    return obj


def get_stats_filename(model_folder='.'):
    return os.path.join(model_folder, 'stats.pkl')


def get_variance_analysis_filename():
    return './variance_analysis/data.pkl'


def get_uuid():
    return str(uuid.uuid4())[:8]


def get_yyyymmdd():
    return str(datetime.date.today()).replace('-', '')


def get_hhmmss():
    return datetime.datetime.now().strftime('%H:%M:%S')


def get_model_folder(rootdir='./models/'):
    return os.path.join(rootdir, get_yyyymmdd() + '_' + get_uuid())


def get_args_filename(model_folder='.'):
    return os.path.join(model_folder, 'args.pkl')


def args_match(model_folder, **kwargs):
    """Do training args match kwargs?"""

    args_filename = get_args_filename(model_folder)
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


def print_with_time(str):
    print(get_yyyymmdd() + ' ' + get_hhmmss() + ' ' + str)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def tree_to_string(tree):
    return str(tree).replace('\'', '')\
                    .replace(',', '')\
                    .replace('[', '(')\
                    .replace(']', ')')


def tree_to_nltk_tree(tree):
    return nltk.Tree.fromstring(tree_to_string(tree))


def logaddexp(a, b):
    """Returns log(exp(a) + exp(b))."""

    torch.logsumexp(torch.cat([a.unsqueeze(0), b.unsqueeze(0)]), dim=0)


def get_posterior(generative_model, inference_network, obs, num_particles=100):
    """Returns a sequence of (tree, log_weight) tuples sorted by
    weight in a descending order. tree is a string representation
    of a tree.
    """

    trees = [inference_network.sample_tree(obs=obs)
             for _ in range(num_particles)]
    log_weights = [(generative_model.get_log_prob(tree, obs) -
                    inference_network.get_tree_log_prob(
                        tree, obs=obs)).detach()
                   for tree in trees]
    tree_log_weight_dict = dict()
    for tree, log_weight in zip(trees, log_weights):
        string_tree = tree_to_string(tree)
        if string_tree in tree_log_weight_dict:
            tree_log_weight_dict[string_tree] = torch.logsumexp(
                torch.cat([tree_log_weight_dict[string_tree].unsqueeze(0),
                           log_weight.unsqueeze(0)]), dim=0)
        else:
            tree_log_weight_dict[string_tree] = log_weight
    return sorted(list(tree_log_weight_dict.items()),
                  key=lambda x: x[1], reverse=True)


def get_inference_network_distribution(inference_network, obs,
                                       num_particles=1000):
    """Returns a sequence of (tree, log_weight) tuples sorted by weight in a
    descending order. tree is a string representation.
    """
    trees = [inference_network.sample_tree(obs=obs)
             for _ in range(num_particles)]
    log_weights = [-torch.log(torch.tensor(num_particles, dtype=torch.float))
                   for _ in trees]

    # refactor (this is just copying the code snippet from get_posterior)
    tree_log_weight_dict = dict()
    for tree, log_weight in zip(trees, log_weights):
        string_tree = tree_to_string(tree)
        if string_tree in tree_log_weight_dict:
            tree_log_weight_dict[string_tree] = torch.logsumexp(
                torch.cat([tree_log_weight_dict[string_tree].unsqueeze(0),
                           log_weight.unsqueeze(0)]), dim=0)
        else:
            tree_log_weight_dict[string_tree] = log_weight
    return sorted(list(tree_log_weight_dict.items()),
                  key=lambda x: x[1], reverse=True)


def empty_list_of_size(*sizes):
    if len(sizes) == 1:
        return [None for _ in range(sizes[0])]
    else:
        return [empty_list_of_size(*sizes[1:]) for _ in range(sizes[0])]


def eval_quadratic(tree, x):
    if isinstance(tree, list):
        root = tree[0]
        subtrees = tree[1:]
        if root == 'Q' or root == 'L':
            if len(subtrees) == 3:
                a, op, b = subtrees
                if op == '+':
                    return eval_quadratic(a, x) + eval_quadratic(b, x)
                elif op == '-':
                    return eval_quadratic(a, x) - eval_quadratic(b, x)
                elif op == '*':
                    return eval_quadratic(a, x) * eval_quadratic(b, x)
                else:
                    raise ArithmeticError
            elif len(subtrees) == 1:
                return eval_quadratic(subtrees[0], x)
        elif root == 'N':
            return eval_quadratic(subtrees[0], x)
    else:
        root = tree
        if root == 'x':
            return x
        elif root == 'x**2':
            return x**2
        elif int(root) in range(1, 21):
            return torch.full_like(x, int(root), dtype=torch.float)


def eval_polynomial(tree, x):
    if isinstance(tree, list):
        root = tree[0]
        subtrees = tree[1:]
        if root == 'E':
            a, op, b = subtrees
            if op == '+':
                return eval_polynomial(a, x) + eval_polynomial(b, x)
            elif op == '-':
                return eval_polynomial(a, x) - eval_polynomial(b, x)
            elif op == '*':
                return eval_polynomial(a, x) * eval_polynomial(b, x)
            else:
                raise ArithmeticError
        elif root == 'E1' or root == 'N':
            return eval_polynomial(subtrees[0], x)
    else:
        root = tree
        if root == 'x':
            return x
        elif int(root) in range(1, 4):
            return torch.full_like(x, int(root), dtype=torch.float)


def mse(ys1, ys2):
    return torch.mean((ys1 - ys2)**2)


def fig2rgba(fig):
    """Convert a Matplotlib figure to a 4D numpy array with RGBA channels and
        return it. From
        http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure

    Args:
        fig: a matplotlib figure

    Returns: a numpy 3D array of RGBA values of shape [w, h, 4]
    """

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    rgba.shape = (w, h, 4)
    rgba = np.roll(rgba, 3, axis=2)
    plt.close(fig)

    return rgba


def rgba2gray(rgba):
    # from https://stackoverflow.com/a/12201744/1357509
    return np.dot(rgba[..., :3], [0.299, 0.587, 0.114])


# This function needs to be optimized
def xsys2gray(xs, ys):
    """Args:
        xs: tensor of shape [100]
        ys: tensor of shape [100]

    Returns: grayscale image repr. by tensor of shape
        [100, 100] where 1 is white and 0 is black"""

    fig, ax = plt.subplots(1, 1, figsize=(1, 1), dpi=100)
    ax.plot(xs.numpy(), ys.numpy(), color='black')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(-100, 100)
    ax.set_xlim(-10, 10)
    sns.despine(ax=ax, left=True, bottom=True)
    fig.tight_layout(pad=0)

    return torch.tensor(rgba2gray(fig2rgba(fig)) / 255.0,
                        dtype=torch.float)


def get_most_recent_model_folder_args_match(**kwargs):
    model_folders = list_model_folders_args_match(**kwargs)
    if len(model_folders) > 0:
        return model_folders[np.argmax(
            [os.stat(x).st_mtime for x in model_folders])]
