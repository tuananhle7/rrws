import torch
import losses
import util
import itertools


def train_sleep(generative_model, inference_network, num_samples,
                num_iterations, callback=None):
    optimizer = torch.optim.Adam(inference_network.parameters())
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        sleep_loss = losses.get_sleep_loss(generative_model, inference_network,
                                           num_samples=num_samples)
        sleep_loss.backward()
        optimizer.step()
        if callback is not None:
            callback(iteration, sleep_loss.item(), generative_model,
                     inference_network, optimizer)

    return optimizer


class TrainSleepCallback():
    def __init__(self, logging_interval=10, saving_interval=10):
        self.sleep_loss_history = []
        self.logging_interval = logging_interval

    def __call__(self, iteration, sleep_loss, generative_model,
                 inference_network, optimizer):
        if iteration % self.logging_interval == 0:
            print('Iteration {}: loss = {:.3f}'.format(iteration, sleep_loss))
            self.sleep_loss_history.append(sleep_loss)


def train_wake_wake(generative_model, inference_network,
                    true_generative_model, batch_size,
                    num_iterations, num_particles, callback=None):
    optimizer_phi = torch.optim.Adam(inference_network.parameters())
    optimizer_theta = torch.optim.Adam(generative_model.parameters())

    for iteration in range(num_iterations):
        # generate synthetic data
        sentences = [util.get_leaves(true_generative_model.sample_tree())
                     for _ in range(batch_size)]

        # wake theta
        optimizer_phi.zero_grad()
        optimizer_theta.zero_grad()
        wake_theta_loss = losses.get_wake_theta_loss(
            generative_model, inference_network, sentences, num_particles)
        wake_theta_loss.backward()
        optimizer_theta.step()

        # wake phi
        optimizer_phi.zero_grad()
        optimizer_theta.zero_grad()
        wake_phi_loss = losses.get_wake_phi_loss(
            generative_model, inference_network, sentences, num_particles)
        wake_phi_loss.backward()
        optimizer_phi.step()

        if callback is not None:
            callback(iteration, wake_theta_loss.item(), wake_phi_loss.item(),
                     generative_model, inference_network, optimizer_theta,
                     optimizer_phi)

    return optimizer_theta, optimizer_phi


class TrainWakeWakeCallback():
    def __init__(self, logging_interval=10):
        self.wake_theta_loss_history = []
        self.wake_phi_loss_history = []
        self.logging_interval = logging_interval

    def __call__(self, iteration, wake_theta_loss, wake_phi_loss,
                 generative_model, inference_network, optimizer_theta,
                 optimizer_phi):
        if iteration % self.logging_interval == 0:
            print('Iteration {} losses: theta = {:.3f}, phi = {:.3f}'.format(
                iteration, wake_theta_loss, wake_phi_loss))
            self.wake_theta_loss_history.append(wake_theta_loss)
            self.wake_phi_loss_history.append(wake_phi_loss)


def train_iwae(algorithm, generative_model, inference_network,
               true_generative_model, batch_size, num_iterations,
               num_particles, callback=None):
    """Train using IWAE objective.

    Args:
        algorithm: reinforce or vimco
    """

    parameters = itertools.chain.from_iterable(
        [x.parameters() for x in [generative_model, inference_network]])
    optimizer = torch.optim.Adam(parameters)

    for iteration in range(num_iterations):
        # generate synthetic data
        sentences = [util.get_leaves(true_generative_model.sample_tree())
                     for _ in range(batch_size)]

        # wake theta
        optimizer.zero_grad()
        if algorithm == 'vimco':
            loss, elbo = losses.get_vimco_loss(
                generative_model, inference_network, sentences, num_particles)
        elif algorithm == 'reinforce':
            loss, elbo = losses.get_reinforce_loss(
                generative_model, inference_network, sentences, num_particles)
        loss.backward()
        optimizer.step()

        if callback is not None:
            callback(iteration, loss.item(), elbo.item(), generative_model,
                     inference_network, optimizer)

    return optimizer


class TrainIwaeCallback():
    def __init__(self, logging_interval=10):
        self.loss_history = []
        self.elbo_history = []
        self.logging_interval = logging_interval

    def __call__(self, iteration, loss, elbo, generative_model,
                 inference_network, optimizer):
        if iteration % self.logging_interval == 0:
            print('Iteration {} loss = {:.3f}, elbo = {:.3f}'.format(
                iteration, loss, elbo))
            self.loss_history.append(loss)
            self.elbo_history.append(elbo)
