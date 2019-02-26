import torch
import torch.nn as nn
import util
from torch.distributions import *


class GenerativeModel(nn.Module):
    def __init__(self, grammar, production_probs_init=None):
        super(GenerativeModel, self).__init__()
        self.grammar = grammar
        if production_probs_init is None:
            self.production_logits = nn.ParameterDict({
                k: nn.Parameter(torch.randn((len(v),)))
                for k, v in grammar['productions'].items()})
        else:
            self.production_logits = nn.ParameterDict({
                k: nn.Parameter(torch.log(v))
                for k, v in production_probs_init.items()})

    def sample_tree(self, symbol=None, depth=0, max_depth=100):
        """Sample tree from prior.

        Args: start symbol
        Returns: list of lists or string
        """

        if symbol is None:
            symbol = self.grammar['start_symbol']

        if symbol in self.grammar['terminals']:
            return symbol
        elif depth > max_depth:
            return symbol
        else:
            dist = Categorical(logits=self.production_logits[symbol])
            production_index = dist.sample().detach()
            production = self.grammar['productions'][symbol][production_index]
            return [symbol] + \
                [self.sample_tree(s, depth=depth + 1) for s in production]

    def get_tree_log_prob(self, tree):
        """Log probability of tree.

        Args: list of lists or string
        Returns: scalar tensor
        """

        if isinstance(tree, list):
            non_terminal = tree[0]
            subtrees = tree[1:]
            production = [util.get_root(subtree) for subtree in subtrees]
            production_index = util.get_production_index(
                non_terminal, production, self.grammar['productions'])
            dist = Categorical(logits=self.production_logits[non_terminal])
            log_prob = dist.log_prob(torch.tensor(production_index))
            subtree_log_probs = [self.get_tree_log_prob(subtree)
                                 for subtree in subtrees]
            return log_prob + sum(subtree_log_probs)
        else:
            return torch.zeros(())

    def get_abc_log_likelihood(self, sentence, tree):
        """ABC distance instead of likelihood p(sentence | tree).

        Args:
            sentence: list of strings
            tree: list of lists or string

        Returns: scalar tensor"""

        sentence_from_tree = util.get_leaves(tree)
        levenshtein_distance = torch.tensor(
            util.get_levenshtein_distance(sentence_from_tree, sentence,
                                          self.grammar['terminals']),
            dtype=torch.float)
        return -levenshtein_distance

    def get_log_prob(self, tree, sentence, abc=True):
        """Joint log probability p(sentence, tree).

        Args:
            tree: list of lists or string
            sentence: list of strings
            abc: use ABC likelihood

        Returns: scalar tensor
        """

        if abc:
            return self.get_tree_log_prob(tree) + self.get_abc_log_likelihood(
                sentence, tree)
        else:
            if util.get_leaves(tree) == sentence:
                return self.get_tree_log_prob(tree)
            else:
                return torch.tensor(float('-inf'))


class InferenceNetwork(nn.Module):
    def __init__(self, grammar, sentence_embedding_dim=100,
                 inference_hidden_dim=100):
        super(InferenceNetwork, self).__init__()
        self.grammar = grammar
        self.sentence_embedding_dim = sentence_embedding_dim
        self.inference_hidden_dim = inference_hidden_dim
        self.sample_address_embedding_dim = len(grammar['non_terminals'])
        self.word_embedding_dim = len(self.grammar['terminals'])

        self.sentence_embedder_gru = nn.GRU(
            input_size=self.word_embedding_dim,
            hidden_size=self.sentence_embedding_dim,
            num_layers=1)
        self.sample_embedding_dim = max(
            [len(v) for _, v in self.grammar['productions'].items()])
        self.inference_gru = nn.GRUCell(
            input_size=self.sentence_embedding_dim + self.sample_embedding_dim
            + self.sample_address_embedding_dim,
            hidden_size=self.inference_hidden_dim)
        self.proposal_layers = nn.ModuleDict({
            k: nn.Sequential(nn.Linear(inference_hidden_dim, 50),
                             nn.ReLU(),
                             nn.Linear(50, 25),
                             nn.ReLU(),
                             nn.Linear(25, len(v)))
            for k, v in grammar['productions'].items()})

    def get_sentence_embedding(self, sentence):
        """Args:
            sentence: list of strings.

        Returns: tensor of shape [sentence_embedding_dim].
        """

        output, _ = self.sentence_embedder_gru(util.sentence_to_one_hots(
            sentence, self.grammar['terminals']).unsqueeze(1))
        return output[-1][0]

    def get_logits_from_inference_gru_output(self, inference_gru_output,
                                             non_terminal):
        """Args:
            inference_gru_output: tensor of shape [inference_hidden_dim]
            non_terminal: string

        Returns: logits for Categorical distribution
        """

        input_ = inference_gru_output.unsqueeze(0)
        return self.proposal_layers[non_terminal](input_).squeeze(0)

    def get_sample_embedding(self, production_index):
        """Args: int
        Returns: one hot vector of shape [sample_embedding_dim]
        """
        return util.one_hot(torch.tensor([production_index]),
                            self.sample_embedding_dim)[0]

    def get_inference_gru_output(self, sentence_embedding,
                                 previous_sample_embedding,
                                 sample_address_embedding, inference_hidden):
        """Args:
            sentence_embedding: tensor [sentence_embedding_dim]
            previous_sample_embedding: tensor [sample_embedding_dim]
            inference_hidden: tensor [inference_hidden_dim]

        Returns: tensor [inference_hidden_dim]
        """

        return self.inference_gru(
            torch.cat([sentence_embedding,
                       previous_sample_embedding,
                       sample_address_embedding]).unsqueeze(0),
            inference_hidden.unsqueeze(0)).squeeze(0)

    def get_tree_log_prob(self, tree, sentence_embedding=None,
                          previous_sample_embedding=None,
                          inference_hidden=None, sentence=None):
        """Log probability of tree given sentence.

        Args:
            tree: list or string
            sentence_embedding: tensor [sentence_embedding_dim]
            previous_sample_embedding: tensor [sample_embedding_dim]
            inference_hidden: tensor [inference_hidden_dim]
            sentence: list of strings

        Returns: log_prob (scalar tensor)"""

        if sentence_embedding is None:
            sentence_embedding = self.get_sentence_embedding(sentence)

        if previous_sample_embedding is None:
            previous_sample_embedding = torch.zeros(
                (self.sample_embedding_dim,))

        if inference_hidden is None:
            inference_hidden = torch.zeros((self.inference_hidden_dim,))

        if isinstance(tree, list):
            non_terminal = tree[0]
            sample_address_embedding = util.get_sample_address_embedding(
                non_terminal, self.grammar['non_terminals'])
            inference_gru_output = self.get_inference_gru_output(
                sentence_embedding, previous_sample_embedding,
                sample_address_embedding, inference_hidden)

            subtrees = tree[1:]
            production = [util.get_root(subtree) for subtree in subtrees]
            production_index = util.get_production_index(
                non_terminal, production, self.grammar['productions'])
            sample_embedding = self.get_sample_embedding(production_index)
            logits = self.get_logits_from_inference_gru_output(
                inference_gru_output, non_terminal)
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(torch.tensor(production_index))
            subtree_log_probs = [
                self.get_tree_log_prob(subtree, sentence_embedding,
                                       sample_embedding, inference_gru_output)
                for subtree in subtrees]
            return log_prob + sum(subtree_log_probs)
        else:
            return torch.zeros(())

    def sample_tree(self, symbol=None, sentence_embedding=None,
                    previous_sample_embedding=None, inference_hidden=None,
                    sentence=None, depth=0, max_depth=100):
        """Samples a tree given a sentence and a start symbol (can be terminal
            or non-terminal).

        Args:
            symbol: string
            sentence_embedding: tensor [sentence_embedding_dim]
            previous_sample_embedding: tensor [sample_embedding_dim]
            inference_hidden: tensor [inference_hidden_dim]
            sentence: list of strings

        Returns: tree represented as vector of vectors where each node is
            augmented with a log_prob
        """

        if symbol is None:
            symbol = self.grammar['start_symbol']

        if sentence_embedding is None:
            sentence_embedding = self.get_sentence_embedding(sentence)

        if previous_sample_embedding is None:
            previous_sample_embedding = torch.zeros(
                (self.sample_embedding_dim,))

        if inference_hidden is None:
            inference_hidden = torch.zeros((self.inference_hidden_dim,))

        if symbol in self.grammar['terminals']:
            return symbol
        elif depth > max_depth:
            return symbol
        else:
            sample_address_embedding = util.get_sample_address_embedding(
                symbol, self.grammar['non_terminals'])
            inference_gru_output = self.get_inference_gru_output(
                sentence_embedding, previous_sample_embedding,
                sample_address_embedding, inference_hidden)
            logits = self.get_logits_from_inference_gru_output(
                inference_gru_output, symbol)
            dist = Categorical(logits=logits)
            production_index = dist.sample().detach()
            sample_embedding = self.get_sample_embedding(production_index)
            production = self.grammar['productions'][symbol][production_index]

            return [symbol] + [
                self.sample_tree(s, sentence_embedding, sample_embedding,
                                 inference_gru_output, depth=depth + 1)
                for s in production]
