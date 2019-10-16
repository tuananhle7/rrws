import util
import torch
import numpy as np


def get_sleep_loss(generative_model, inference_network, num_samples=1):
    """Returns:
        loss: scalar that we call .backward() on and step the optimizer.
    """

    latent, obs = generative_model.sample_latent_and_obs(num_samples)
    return -torch.mean(inference_network.get_log_prob(latent, obs))


def get_log_weight_and_log_q(generative_model, inference_network, obs,
                             num_particles=1, reparam=False):
    """Compute log weight and log prob of inference network.

    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size]
        num_particles: int
        reparam: reparameterize sampling from q (only applicable if z is
            Concrete)

    Returns:
        log_weight: tensor of shape [batch_size, num_particles]
        log_q: tensor of shape [batch_size, num_particles]
    """

    latent_dist = inference_network.get_latent_dist(obs)
    latent = inference_network.sample_from_latent_dist(
        latent_dist, num_particles, reparam=reparam)
    log_p = generative_model.get_log_prob(latent, obs).transpose(0, 1)
    log_q = inference_network.get_log_prob_from_latent_dist(
        latent_dist, latent).transpose(0, 1)
    log_weight = log_p - log_q
    return log_weight, log_q


def get_wake_theta_loss_from_log_weight(log_weight):
    """Args:
        log_weight: tensor of shape [batch_size, num_particles]

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """

    _, num_particles = log_weight.shape
    elbo = torch.mean(
        torch.logsumexp(log_weight, dim=1) - np.log(num_particles))
    return -elbo, elbo


def get_wake_theta_loss(generative_model, inference_network, obs,
                        num_particles=1):
    """Scalar that we call .backward() on and step the optimizer.

    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size]
        num_particles: int

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """
    log_weight, _ = get_log_weight_and_log_q(
        generative_model, inference_network, obs, num_particles)
    return get_wake_theta_loss_from_log_weight(log_weight)


def get_wake_phi_loss_from_log_weight_and_log_q(log_weight, log_q):
    """Returns:
        loss: scalar that we call .backward() on and step the optimizer.
    """
    normalized_weight = util.exponentiate_and_normalize(log_weight, dim=1)
    return torch.mean(-torch.sum(normalized_weight.detach() * log_q, dim=1))


def get_wake_phi_loss(generative_model, inference_network, obs,
                      num_particles=1):
    """
    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size]
        num_particles: int

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
    """
    log_weight, log_q = get_log_weight_and_log_q(
        generative_model, inference_network, obs, num_particles)
    return get_wake_phi_loss_from_log_weight_and_log_q(log_weight, log_q)


def get_defensive_wake_phi_loss(generative_model, inference_network, obs,
                                delta, num_particles=1):
    num_mixtures = inference_network.num_mixtures
    batch_size = len(obs)
    latent_dist = inference_network.get_latent_dist(obs)
    latent = inference_network.sample_from_latent_dist(latent_dist,
                                                       num_particles)
    latent_uniform = torch.distributions.OneHotCategorical(
        logits=torch.ones((num_mixtures,))).sample((num_particles, batch_size))

    catted = torch.cat([x.unsqueeze(0) for x in
                        [latent, latent_uniform]], dim=0)
    indices = torch.distributions.Bernoulli(
        probs=torch.tensor(delta)).sample(
        (num_particles, batch_size)).unsqueeze(0).unsqueeze(-1).expand(
        1, num_particles, batch_size, num_mixtures).long()
    latent_mixture = torch.gather(catted, 0, indices).squeeze(0)
    log_p = generative_model.get_log_prob(latent_mixture, obs).transpose(0, 1)
    log_latent = inference_network.get_log_prob_from_latent_dist(
        latent_dist, latent).transpose(0, 1)

    log_uniform = torch.ones_like(log_latent) * (-np.log(num_mixtures))
    log_q_mixture = util.logaddexp(log_latent + np.log(1 - delta),
                                   log_uniform + np.log(delta))
    log_q = inference_network.get_log_prob_from_latent_dist(
        latent_dist, latent_mixture).transpose(0, 1)
    log_weight = log_p - log_q_mixture
    normalized_weight = util.exponentiate_and_normalize(log_weight, dim=1)
    return torch.mean(-torch.sum(normalized_weight.detach() * log_q, dim=1))


def get_reinforce_loss(generative_model, inference_network, obs,
                       num_particles=1):
    """
    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size]
        num_particles: int

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """
    log_weight, log_q = get_log_weight_and_log_q(
        generative_model, inference_network, obs, num_particles)
    log_evidence = torch.logsumexp(log_weight, dim=1) - np.log(num_particles)

    # this is term 1 in equation (2) of https://arxiv.org/pdf/1805.10469.pdf
    reinforce_correction = log_evidence.detach() * torch.sum(log_q, dim=1)

    elbo = torch.mean(log_evidence)
    loss = -elbo - torch.mean(reinforce_correction)
    return loss, elbo


def get_vimco_loss(generative_model, inference_network, obs,
                   num_particles=1):
    """
    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size]
        num_particles: int

    Returns:

        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """
    log_weight, log_q = get_log_weight_and_log_q(
        generative_model, inference_network, obs, num_particles)
    log_evidence = torch.logsumexp(log_weight, dim=1) - np.log(num_particles)
    reinforce_correction = 0
    for i in range(num_particles):
        log_weight_ = log_weight[:, util.range_except(num_particles, i)]

        # this is the B term in VIMCO gradient in
        # https://arxiv.org/pdf/1805.10469.pdf
        control_variate = torch.logsumexp(
            torch.cat([log_weight_,
                       torch.mean(log_weight_, dim=1, keepdim=True)], dim=1),
            dim=1) - np.log(num_particles)
        reinforce_correction = reinforce_correction + \
            (log_evidence.detach() - control_variate.detach()) * log_q[:, i]

    elbo = torch.mean(log_evidence)
    loss = -elbo - torch.mean(reinforce_correction)
    return loss, elbo


def get_concrete_loss(generative_model, inference_network, obs,
                      num_particles=1):
    """
    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size]
        num_particles: int

    Returns:

        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """
    log_weight, _ = get_log_weight_and_log_q(
        generative_model, inference_network, obs, num_particles, reparam=True)
    log_evidence = torch.logsumexp(log_weight, dim=1) - np.log(num_particles)
    elbo = torch.mean(log_evidence)
    loss = -elbo
    return loss, elbo


def get_relax_loss(generative_model, inference_network, control_variate,
                   obs, num_particles=1):
    """
    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        control_variate: models.ControlVariate object
        obs: tensor of shape [batch_size]
        num_particles: int

    Returns:

        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """
    # batch_size = len(obs)
    # num_mixtures = generative_model.num_mixtures
    latent, latent_aux, latent_aux_tilde = util.sample_relax(
        inference_network, control_variate, obs, num_particles)
    log_q = inference_network.get_log_prob(latent.transpose(0, 1), obs)\
        .transpose(0, 1)
    log_p = generative_model.get_log_prob(latent.transpose(0, 1), obs)\
        .transpose(0, 1)
    log_weight = log_p - log_q
    log_evidence = torch.logsumexp(log_weight, dim=1) - np.log(num_particles)
    c = control_variate(latent_aux, obs)
    c_tilde = control_variate(latent_aux_tilde, obs)
    c_tilde_detached_latent = control_variate(latent_aux_tilde.detach(), obs)

    return -torch.mean(
        (log_evidence.detach() - c_tilde_detached_latent) *
        torch.sum(log_q, dim=1) + c - c_tilde + log_evidence), \
        torch.mean(log_evidence)
