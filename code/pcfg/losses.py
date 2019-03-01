import torch
import numpy as np
import util


def get_sleep_loss(generative_model, inference_network, num_samples=1):
    """Returns:
        loss: scalar that we call .backward() on and step the optimizer.
    """
    log_q_sum = 0
    for _ in range(num_samples):
        tree, obs = generative_model.sample_tree_and_obs()
        log_q = inference_network.get_tree_log_prob(tree, obs=obs)
        log_q_sum = log_q_sum + log_q
    return -log_q_sum / num_samples


def get_log_weight_and_log_q(generative_model, inference_network, obss,
                             num_particles=1):
    """Compute log weight and log prob of inference network.

    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obss: list of obs each of which is either a sentence (list of strings)
            or ys (tensor of shape [50])
        num_particles: int

    Returns:
        log_weight: tensor of shape [num_obss, num_particles]
        log_q: tensor of shape [num_obss, num_particles]
    """
    log_weight = torch.zeros(len(obss), num_particles)
    log_q = torch.zeros(len(obss), num_particles)
    for obs_idx, obs in enumerate(obss):
        for particle_idx in range(num_particles):
            tree = inference_network.sample_tree(obs=obs)
            log_q_ = inference_network.get_tree_log_prob(tree, obs=obs)
            log_p_ = generative_model.get_log_prob(tree, obs)
            log_weight[obs_idx, particle_idx] = log_p_ - log_q_
            log_q[obs_idx, particle_idx] = log_q_
    return log_weight, log_q


def get_wake_theta_loss_from_log_weight(log_weight):
    """Args:
        log_weight: tensor of shape [num_obs, num_particles]

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """
    _, num_particles = log_weight.shape
    log_evidence = torch.logsumexp(log_weight, dim=1) - np.log(num_particles)
    elbo = torch.mean(log_evidence)
    return -elbo, elbo


def get_wake_theta_loss(generative_model, inference_network, obss,
                        num_particles=1):
    """Scalar that we call .backward() on and step the optimizer.

    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obss: list of obs each of which is either a sentence (list of strings)
            or ys (tensor of shape [50])
        num_particles: int

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """
    log_weight, _ = get_log_weight_and_log_q(
        generative_model, inference_network, obss, num_particles)
    return get_wake_theta_loss_from_log_weight(log_weight)


def get_wake_phi_loss_from_log_weight_and_log_q(log_weight, log_q):
    """Returns:
        loss: scalar that we call .backward() on and step the optimizer.
    """
    normalized_weight = util.exponentiate_and_normalize(log_weight, dim=1)
    return torch.mean(-torch.sum(normalized_weight.detach() * log_q, dim=1))


def get_wake_phi_loss(generative_model, inference_network, obss,
                      num_particles=1):
    """
    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obss: list of obs each of which is either a sentence (list of strings)
            or ys (tensor of shape [50])
        num_particles: int

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
    """
    log_weight, log_q = get_log_weight_and_log_q(
        generative_model, inference_network, obss, num_particles)
    return get_wake_phi_loss_from_log_weight_and_log_q(log_weight, log_q)


def get_reinforce_loss(generative_model, inference_network, obss,
                       num_particles=1):
    """
    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obss: list of obs each of which is either a sentence (list of strings)
            or ys (tensor of shape [50])
        num_particles: int

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """
    log_weight, log_q = get_log_weight_and_log_q(
        generative_model, inference_network, obss, num_particles)
    log_evidence = torch.logsumexp(log_weight, dim=1) - np.log(num_particles)

    # this is term 1 in equation (2) of https://arxiv.org/pdf/1805.10469.pdf
    reinforce_correction = log_evidence.detach() * torch.sum(log_q, dim=1)

    elbo = torch.mean(log_evidence)
    loss = -elbo - torch.mean(reinforce_correction)
    return loss, elbo


def get_vimco_loss(generative_model, inference_network, obss,
                   num_particles=1):
    """Returns:

        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """
    log_weight, log_q = get_log_weight_and_log_q(
        generative_model, inference_network, obss, num_particles)
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
