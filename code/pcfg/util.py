import torch
import itertools
import json
import os
import models
import string
import Levenshtein


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

    Returns: one hot tensor of shape [len(terminals)]
    """
    num_bins = len(terminals)
    i = sorted(terminals).index(word)
    return one_hot(torch.tensor([i]), num_bins)[0]


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
        'start_symbol': data['start_symbol']
    }
    true_production_probs = production_probs_to_tensor(
        data['production_probs'])

    return grammar, true_production_probs


def save_models(generative_model, inference_network, pcfg_path,
                model_folder='.'):
    generative_model_path = os.path.join(model_folder, 'gen.pt')
    inference_network_path = os.path.join(model_folder, 'inf.pt')
    pcfg_path_path = os.path.join(model_folder, 'pcfg_path.txt')

    torch.save(generative_model.state_dict(), generative_model_path)
    print('Saved to {}'.format(generative_model_path))
    torch.save(inference_network.state_dict(), inference_network_path)
    print('Saved to {}'.format(inference_network_path))
    with open(pcfg_path_path, 'w') as f:
        f.write(pcfg_path)
    print('Saved to {}'.format(pcfg_path_path))


def load_models(model_folder='.'):
    generative_model_path = os.path.join(model_folder, 'gen.pt')
    inference_network_path = os.path.join(model_folder, 'inf.pt')
    pcfg_path_path = os.path.join(model_folder, 'pcfg_path.txt')

    with open(pcfg_path_path) as f:
        pcfg_path = f.read()
    grammar, _ = read_pcfg(pcfg_path)
    generative_model = models.GenerativeModel(grammar)
    inference_network = models.InferenceNetwork(grammar)
    generative_model.load_state_dict(torch.load(generative_model_path))
    print('Loaded from {}'.format(generative_model_path))
    inference_network.load_state_dict(torch.load(inference_network_path))
    print('Loaded from {}'.format(inference_network_path))

    return generative_model, inference_network


def word_to_index(word, terminals):
    """Convert word to int.

    Args:
        word: string
        terminals: set of terminal strings

    Returns: int
    """

    return sorted(terminals).index(word)


def sentence_to_indices(sentence, terminals):
    """Convert sentence to list of indices.

    Args:
        sentence: list of strings
        terminals: set of terminal strings

    Returns: list of indices of length len(sentence)
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
        terminal: vocabulary; list of valid strings

    Returns: int"""
    return Levenshtein.distance(_sentence_to_string(sentence_1, terminals),
                                _sentence_to_string(sentence_2, terminals))


def init_models(pcfg_path):
    grammar, true_production_probs = read_pcfg(pcfg_path)
    generative_model = models.GenerativeModel(grammar)
    inference_network = models.InferenceNetwork(grammar)
    true_generative_model = models.GenerativeModel(
        grammar, true_production_probs)

    return generative_model, inference_network, true_generative_model
