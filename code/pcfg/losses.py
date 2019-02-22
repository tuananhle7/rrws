import torch
import numpy as np
import util


def get_sleep_loss(generative_model, inference_network, num_samples=1):
    """Returns:

        loss: scalar that we call .backward() on and step the optimizer.
    """
    log_q_sum = 0
    for _ in range(num_samples):
        tree = generative_model.sample_tree()
        sentence = util.get_leaves(tree)
        log_q = inference_network.get_tree_log_prob(
            tree, sentence=sentence)
        log_q_sum = log_q_sum + log_q
    return -log_q_sum / num_samples


def get_log_weight_and_log_q(generative_model, inference_network, sentences,
                             num_particles=1):
    """Compute log weight and log prob of inference network.

    Returns:
        log_weight: tensor of shape [num_sentences, num_particles]
        log_q: tensor of shape [num_sentences, num_particles]
    """
    log_weight = torch.zeros(len(sentences), num_particles)
    log_q = torch.zeros(len(sentences), num_particles)
    for sentence_idx, sentence in enumerate(sentences):
        for particle_idx in range(num_particles):
            tree = inference_network.sample_tree(sentence=sentence)
            log_q_ = inference_network.get_tree_log_prob(tree,
                                                         sentence=sentence)
            log_p_ = generative_model.get_log_prob(tree, sentence)
            log_weight[sentence_idx, particle_idx] = log_p_ - log_q_
            log_q[sentence_idx, particle_idx] = log_q_
    return log_weight, log_q


def get_wake_theta_loss_from_log_weight(log_weight):
    """Returns:

        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """
    _, num_particles = log_weight.shape
    log_evidence = torch.logsumexp(log_weight, dim=1) - np.log(num_particles)
    elbo = torch.mean(log_evidence)
    return -elbo, elbo


def get_wake_theta_loss(generative_model, inference_network, sentences,
                        num_particles=1):
    """Returns:

        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """
    log_weight, _ = get_log_weight_and_log_q(
        generative_model, inference_network, sentences, num_particles)
    return get_wake_theta_loss_from_log_weight(log_weight)


def get_wake_phi_loss_from_log_weight_and_log_q(log_weight, log_q):
    """Returns:

        loss: scalar that we call .backward() on and step the optimizer.
    """
    normalized_weight = util.exponentiate_and_normalize(log_weight, dim=1)
    return torch.mean(-torch.sum(normalized_weight.detach() * log_q, dim=1))


def get_wake_phi_loss(generative_model, inference_network, sentences,
                      num_particles=1):
    """Returns:

        loss: scalar that we call .backward() on and step the optimizer.
    """
    log_weight, log_q = get_log_weight_and_log_q(
        generative_model, inference_network, sentences, num_particles)
    return get_wake_phi_loss_from_log_weight_and_log_q(log_weight, log_q)


def get_reinforce_loss(generative_model, inference_network, sentences,
                       num_particles=1):
    """Returns:

        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """
    log_weight, log_q = get_log_weight_and_log_q(
        generative_model, inference_network, sentences, num_particles)
    log_evidence = torch.logsumexp(log_weight, dim=1) - np.log(num_particles)

    # this is term 1 in equation (2) of https://arxiv.org/pdf/1805.10469.pdf
    reinforce_correction = log_evidence.detach() * torch.sum(log_q, dim=1)

    elbo = torch.mean(log_evidence)
    loss = -elbo - torch.mean(reinforce_correction)
    return loss, elbo


def get_vimco_loss(generative_model, inference_network, sentences,
                   num_particles=1):
    """Returns:

        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """
    log_weight, log_q = get_log_weight_and_log_q(
        generative_model, inference_network, sentences, num_particles)
    log_evidence = torch.logsumexp(log_weight, dim=1) - np.log(num_particles)
    reinforce_correction = 0
    for i in range(num_particles):
        log_weight_ = log_weight[:, util.range_except(num_particles, i)]

        # this is the B term in VIMCO gradient in
        # https://arxiv.org/pdf/1805.10469.pdf
        control_variate = torch.logsumexp(
            torch.cat([log_weight_,
                       torch.mean(log_weight_, dim=1, keepdim=True)], dim=1),
            dim=1)
        reinforce_correction = reinforce_correction + \
            (log_evidence.detach() - control_variate.detach()) * log_q[:, i]

    elbo = torch.mean(log_evidence)
    loss = -elbo - torch.mean(reinforce_correction)
    return loss, elbo
