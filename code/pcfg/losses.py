import torch
import numpy as np
import util


def get_sleep_loss(generative_model, inference_network, num_samples=1):
    log_q_sum = 0
    for _ in range(num_samples):
        tree = generative_model.sample_tree()
        sentence = util.get_leaves(tree)
        log_q = inference_network.get_tree_log_prob(
            tree, sentence=sentence)
        log_q_sum = log_q_sum + log_q
    return -log_q_sum / num_samples


def get_wake_theta_loss(generative_model, inference_network, sentences,
                        num_particles=1):
    log_weight = torch.zeros(len(sentences), num_particles)
    for sentence_idx, sentence in enumerate(sentences):
        for particle_idx in range(num_particles):
            tree = inference_network.sample_tree(
                sentence=sentence)
            log_q = inference_network.get_tree_log_prob(
                tree, sentence=sentence)
            log_p = generative_model.get_log_prob(
                tree, sentence)
            log_weight[sentence_idx, particle_idx] = log_p - log_q
    log_evidence = torch.logsumexp(log_weight, dim=1) - np.log(
        num_particles)
    return -torch.mean(log_evidence)


def get_wake_phi_loss(generative_model, inference_network, sentences,
                      num_particles=1):
    log_weight = torch.zeros(len(sentences), num_particles)
    log_q = torch.zeros(len(sentences), num_particles)
    for sentence_idx, sentence in enumerate(sentences):
        for particle_idx in range(num_particles):
            tree = inference_network.sample_tree(
                sentence=sentence)
            log_q_ = inference_network.get_tree_log_prob(
                tree, sentence=sentence)
            log_p_ = generative_model.get_log_prob(
                tree, sentence)
            log_weight[sentence_idx, particle_idx] = log_p_ - log_q_
            log_q[sentence_idx, particle_idx] = log_q_
    normalized_weight = util.exponentiate_and_normalize(log_weight, dim=1)
    return torch.mean(
        -torch.sum(normalized_weight.detach() * log_q, dim=1))
