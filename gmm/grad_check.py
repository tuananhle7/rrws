import torch
import util
import numpy as np
import argparse
import losses


args = argparse.Namespace()
args.device = torch.device('cpu')
args.num_mixtures = 20
args.init_mixture_logits = np.ones(args.num_mixtures)
args.softmax_multiplier = 0.5
args.relaxed_one_hot = False
args.temperature = None
temp = np.arange(args.num_mixtures) + 5
true_p_mixture_probs = temp / np.sum(temp)
args.true_mixture_logits = \
    np.log(true_p_mixture_probs) / args.softmax_multiplier

args.seed = 1
# init models
util.set_seed(args.seed)
generative_model, inference_network, true_generative_model = \
    util.init_models(args)

optimizer_phi = torch.optim.Adam(inference_network.parameters())
optimizer_theta = torch.optim.Adam(generative_model.parameters())
batch_size = 3
num_particles = 4
obs = true_generative_model.sample_obs(batch_size)


def get_grads_correct(seed):
    util.set_seed(seed)

    theta_grads_correct = []
    phi_grads_correct = []

    log_weight, log_q = losses.get_log_weight_and_log_q(
        generative_model, inference_network, obs, num_particles)

    optimizer_phi.zero_grad()
    optimizer_theta.zero_grad()
    wake_theta_loss, elbo = losses.get_wake_theta_loss_from_log_weight(
        log_weight)
    wake_theta_loss.backward(retain_graph=True)
    theta_grads_correct = [parameter.grad.clone() for parameter in
                           generative_model.parameters()]
    # in rws, we step as we compute the grads
    # optimizer_theta.step()

    optimizer_phi.zero_grad()
    optimizer_theta.zero_grad()
    wake_phi_loss = losses.get_wake_phi_loss_from_log_weight_and_log_q(
        log_weight, log_q)
    wake_phi_loss.backward()
    phi_grads_correct = [parameter.grad.clone() for parameter in
                         inference_network.parameters()]
    # in rws, we step as we compute the grads
    # optimizer_phi.step()
    return theta_grads_correct, phi_grads_correct


def get_grads_in_one(seed):
    util.set_seed(seed)

    theta_grads_in_one = []
    phi_grads_in_one = []

    log_weight, log_q = losses.get_log_weight_and_log_q(
        generative_model, inference_network, obs, num_particles)

    optimizer_phi.zero_grad()
    optimizer_theta.zero_grad()
    wake_theta_loss, elbo = losses.get_wake_theta_loss_from_log_weight(
        log_weight)
    wake_theta_loss.backward(retain_graph=True)

    optimizer_phi.zero_grad()
    # optimizer_theta.zero_grad()
    wake_phi_loss = losses.get_wake_phi_loss_from_log_weight_and_log_q(
        log_weight, log_q)
    wake_phi_loss.backward()

    # only get the grads in the end!
    theta_grads_in_one = [parameter.grad.clone() for parameter in
                          generative_model.parameters()]
    phi_grads_in_one = [parameter.grad.clone() for parameter in
                        inference_network.parameters()]

    # in pyro, we want step to be in a different stage
    # optimizer_theta.step()
    # optimizer_phi.step()
    return theta_grads_in_one, phi_grads_in_one


def get_grads_in_one_no_zeroing(seed):
    util.set_seed(seed)

    theta_grads_in_one = []
    phi_grads_in_one = []

    log_weight, log_q = losses.get_log_weight_and_log_q(
        generative_model, inference_network, obs, num_particles)

    optimizer_phi.zero_grad()
    optimizer_theta.zero_grad()
    wake_theta_loss, elbo = losses.get_wake_theta_loss_from_log_weight(
        log_weight)
    wake_theta_loss.backward(retain_graph=True)

    # optimizer_phi.zero_grad() -> don't zero phi grads
    # optimizer_theta.zero_grad()
    wake_phi_loss = losses.get_wake_phi_loss_from_log_weight_and_log_q(
        log_weight, log_q)
    wake_phi_loss.backward()

    # only get the grads in the end!
    theta_grads_in_one = [parameter.grad.clone() for parameter in
                          generative_model.parameters()]
    phi_grads_in_one = [parameter.grad.clone() for parameter in
                        inference_network.parameters()]

    # in pyro, we want step to be in a different stage
    # optimizer_theta.step()
    # optimizer_phi.step()
    return theta_grads_in_one, phi_grads_in_one


def get_log_weight_and_log_q_weird_detach(generative_model, inference_network, obs,
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
    log_weight = log_p - log_q.detach()
    return log_weight, log_q


def get_grads_weird_detach(seed):
    util.set_seed(seed)

    theta_grads_in_one = []
    phi_grads_in_one = []

    log_weight, log_q = get_log_weight_and_log_q_weird_detach(
        generative_model, inference_network, obs, num_particles)

    optimizer_phi.zero_grad()
    optimizer_theta.zero_grad()
    wake_theta_loss, elbo = losses.get_wake_theta_loss_from_log_weight(
        log_weight)
    wake_theta_loss.backward(retain_graph=True)

    # optimizer_phi.zero_grad() -> don't zero phi grads
    # optimizer_theta.zero_grad()
    wake_phi_loss = losses.get_wake_phi_loss_from_log_weight_and_log_q(
        log_weight, log_q)
    wake_phi_loss.backward()

    # only get the grads in the end!
    theta_grads_in_one = [parameter.grad.clone() for parameter in
                          generative_model.parameters()]
    phi_grads_in_one = [parameter.grad.clone() for parameter in
                        inference_network.parameters()]

    # in pyro, we want step to be in a different stage
    # optimizer_theta.step()
    # optimizer_phi.step()
    return theta_grads_in_one, phi_grads_in_one


def get_grads_correct_sleep(seed):
    util.set_seed(seed)

    theta_grads_correct = []
    phi_grads_correct = []

    log_weight, log_q = losses.get_log_weight_and_log_q(
        generative_model, inference_network, obs, num_particles)

    optimizer_phi.zero_grad()
    optimizer_theta.zero_grad()
    wake_theta_loss, elbo = losses.get_wake_theta_loss_from_log_weight(
        log_weight)
    wake_theta_loss.backward(retain_graph=True)
    theta_grads_correct = [parameter.grad.clone() for parameter in
                           generative_model.parameters()]
    # in rws, we step as we compute the grads
    # optimizer_theta.step()

    optimizer_phi.zero_grad()
    optimizer_theta.zero_grad()
    wake_phi_loss = losses.get_wake_phi_loss_from_log_weight_and_log_q(
        log_weight, log_q)
    wake_phi_loss.backward()
    wake_phi_grads_correct = [parameter.grad.clone() for parameter in
                              inference_network.parameters()]
    # in rws, we step as we compute the grads
    # optimizer_phi.step()

    optimizer_phi.zero_grad()
    optimizer_theta.zero_grad()
    sleep_phi_loss = losses.get_sleep_loss(
        generative_model, inference_network, num_samples=num_particles)
    sleep_phi_loss.backward()
    sleep_phi_grads_correct = [parameter.grad.clone() for parameter in
                               inference_network.parameters()]

    wake_factor = 0.7
    phi_grads_correct = [wake_factor * wake_phi_grad_correct + (1 - wake_factor) * sleep_phi_grad_correct
                         for wake_phi_grad_correct, sleep_phi_grad_correct in
                         zip(wake_phi_grads_correct, sleep_phi_grads_correct)]

    return theta_grads_correct, phi_grads_correct


def get_grads_weird_detach_sleep(seed):
    util.set_seed(seed)

    theta_grads_in_one = []
    phi_grads_in_one = []

    log_weight, log_q = get_log_weight_and_log_q_weird_detach(
        generative_model, inference_network, obs, num_particles)

    optimizer_phi.zero_grad()
    optimizer_theta.zero_grad()
    wake_theta_loss, elbo = losses.get_wake_theta_loss_from_log_weight(
        log_weight)

    # optimizer_phi.zero_grad() -> don't zero phi grads
    # optimizer_theta.zero_grad()
    wake_phi_loss = losses.get_wake_phi_loss_from_log_weight_and_log_q(
        log_weight, log_q)

    sleep_phi_loss = losses.get_sleep_loss(
        generative_model, inference_network, num_samples=num_particles)

    wake_factor = 0.7
    phi_loss = wake_factor * wake_phi_loss + (1 - wake_factor) * sleep_phi_loss

    wake_theta_loss.backward(retain_graph=True)
    phi_loss.backward()

    # only get the grads in the end!
    theta_grads_in_one = [parameter.grad.clone() for parameter in
                          generative_model.parameters()]
    phi_grads_in_one = [parameter.grad.clone() for parameter in
                        inference_network.parameters()]

    # in pyro, we want step to be in a different stage
    # optimizer_theta.step()
    # optimizer_phi.step()
    return theta_grads_in_one, phi_grads_in_one


def are_tensors_equal(xs, ys):
    return all([torch.all(torch.eq(x, y)) for x, y in zip(xs, ys)])


seed = 1
grads_correct = sum(get_grads_correct(seed), [])
grads_in_one = sum(get_grads_in_one(seed), [])
grads_in_one_no_zeroing = sum(get_grads_in_one_no_zeroing(seed), [])
grads_weird_detach = sum(get_grads_weird_detach(seed), [])
sleep_grads_correct = sum(get_grads_correct_sleep(seed), [])
sleep_grads_weird_detach = sum(get_grads_weird_detach_sleep(seed), [])

# is computing grads all in once ok?
print('Computing grads all at once is ok: {}'.format(
    are_tensors_equal(grads_correct, grads_in_one)))
print('Computing grads all at once and not zeroing phi grads is ok: {}'.format(
    are_tensors_equal(grads_correct, grads_in_one_no_zeroing)))
print('Computing grads with weird detach is ok: {}'.format(
    are_tensors_equal(grads_correct, grads_weird_detach)))
print('Computing sleep grads with weird detach is ok: {}'.format(
    are_tensors_equal(sleep_grads_correct, sleep_grads_weird_detach)))
