from data import *
from neural_nets import *
from torch.autograd import Variable

import argparse
import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data

SMALL_SIZE = 7
MEDIUM_SIZE = 9
BIGGER_SIZE = 11

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def train_vae(vae, optimizer, train_dataloader, valid_dataloader, num_epochs, num_valid_particles):
    train_elbo_history = np.zeros([num_epochs])
    valid_elbo_history = np.zeros([num_epochs])
    train_start_time = datetime.datetime.now()
    for epoch_idx in range(num_epochs):
        for observation in iter(train_dataloader):
            optimizer.zero_grad()
            loss, train_elbo = vae(Variable(observation))
            loss.backward()
            optimizer.step()

            train_elbo_history[epoch_idx] += torch.sum(train_elbo).data[0] / len(train_dataloader.dataset)

        for observation in iter(valid_dataloader):
            valid_elbo_history[epoch_idx] += torch.sum(vae.generative_network.elbo(
                Variable(observation),
                vae.inference_network,
                num_particles=num_valid_particles
            )).data[0] / len(valid_dataloader.dataset)

        if epoch_idx % 10 == 0:
            estimated_remaining_time = (datetime.datetime.now() - train_start_time) / (epoch_idx + 1) * (num_epochs - (epoch_idx + 1))
            logging.info('Epoch {}: train {} | valid {} | est. {}'.format(
                epoch_idx,
                train_elbo_history[epoch_idx],
                valid_elbo_history[epoch_idx],
                estimated_remaining_time
            ))

    return train_elbo_history, valid_elbo_history, vae, optimizer


def train_vae_relax(vae, vae_optimizer, control_variate_optimizer, train_dataloader, valid_dataloader, num_epochs, num_valid_particles):
    train_elbo_history = np.zeros([num_epochs])
    valid_elbo_history = np.zeros([num_epochs])

    num_parameters = sum([len(p) for p in vae.vae_params])
    train_start_time = datetime.datetime.now()
    for epoch_idx in range(num_epochs):
        for observation in iter(train_dataloader):
            # Train VAE
            vae_optimizer.zero_grad()
            loss, train_elbo = vae(Variable(observation))
            loss.backward(create_graph=True)

            two_loss_grad_detached = [2 * p.grad.detach() / num_parameters for p in vae.vae_params]

            vae_optimizer.step()

            # Train control variate
            control_variate_optimizer.zero_grad()
            loss_grad = [p.grad for p in vae.vae_params]
            torch.autograd.backward(loss_grad, two_loss_grad_detached)
            control_variate_optimizer.step()

            train_elbo_history[epoch_idx] += torch.sum(train_elbo).data[0] / len(train_dataloader.dataset)

        for observation in iter(valid_dataloader):
            valid_elbo_history[epoch_idx] += torch.sum(vae.generative_network.elbo(
                Variable(observation),
                vae.inference_network,
                num_particles=num_valid_particles
            )).data[0] / len(valid_dataloader.dataset)

        if epoch_idx % 10 == 0:
            estimated_remaining_time = (datetime.datetime.now() - train_start_time) / (epoch_idx + 1) * (num_epochs - (epoch_idx + 1))
            logging.info('Epoch {}: train {} | valid {} | est. {}'.format(
                epoch_idx,
                train_elbo_history[epoch_idx],
                valid_elbo_history[epoch_idx],
                estimated_remaining_time
            ))

    return train_elbo_history, valid_elbo_history, vae, vae_optimizer, control_variate_optimizer


def train_cdae(generative_network, inference_network, theta_optimizer, phi_optimizer, num_theta_train_particles, train_dataloader, valid_dataloader, num_epochs, num_valid_particles):
    train_theta_loss_history = np.zeros([num_epochs])
    train_phi_loss_history = np.zeros([num_epochs])
    valid_elbo_history = np.zeros([num_epochs])

    train_start_time = datetime.datetime.now()
    for epoch_idx in range(num_epochs):
        for observation in iter(train_dataloader):
            # Train generative network (theta)
            theta_optimizer.zero_grad()
            loss, train_elbo = generative_network(Variable(observation), inference_network, num_theta_train_particles)
            loss.backward()
            theta_optimizer.step()
            train_theta_loss_history[epoch_idx] += torch.sum(-train_elbo).data[0] / len(train_dataloader.dataset)

            # Train inference network (phi)
            batch_size = observation.size(0)
            generated_latent, generated_observation = generative_network.sample(batch_size)
            phi_optimizer.zero_grad()
            loss = inference_network(generated_latent, generated_observation)
            loss.backward()
            phi_optimizer.step()
            train_phi_loss_history[epoch_idx] += torch.sum(loss).data[0] / len(train_dataloader.dataset)

        for observation in iter(valid_dataloader):
            valid_elbo_history[epoch_idx] += torch.sum(generative_network.elbo(
                Variable(observation),
                inference_network,
                num_particles=num_valid_particles
            )).data[0] / len(valid_dataloader.dataset)

        if epoch_idx % 10 == 0:
            estimated_remaining_time = (datetime.datetime.now() - train_start_time) / (epoch_idx + 1) * (num_epochs - (epoch_idx + 1))
            logging.info('Epoch {}: theta {} | phi {} | valid {} | est. {}'.format(
                epoch_idx,
                train_theta_loss_history[epoch_idx],
                train_phi_loss_history[epoch_idx],
                valid_elbo_history[epoch_idx],
                estimated_remaining_time
            ))

    return train_theta_loss_history, train_phi_loss_history, valid_elbo_history, generative_network, inference_network, theta_optimizer, phi_optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--dataset', help='one of mnist, omniglot')
    parser.add_argument('--estimator', help='one of reinforce, relax, cdae')
    parser.add_argument('--architecture', help='one of L1, L2, NL')
    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        datefmt='%Y-%m-%d %H:%M:%S',
        format='%(asctime)s %(message)s',
        level=logging.INFO
    )

    # Cuda
    if args.cuda:
        if torch.cuda.is_available():
            cuda = True
        else:
            cuda = False
            logging.info('Cuda is not available.')
    else:
        cuda = False

    # Load data
    if args.dataset == 'mnist':
        train_observation_numpy, valid_observation_numpy, _ = load_binarized_mnist()
    elif args.dataset == 'omniglot':
        train_observation_numpy, valid_observation_numpy, _ = load_binarized_omniglot()

    # train_observation_numpy = train_observation_numpy[:10]
    # valid_observation_numpy = valid_observation_numpy[:10]

    train_observation = torch.from_numpy(train_observation_numpy).float().cuda() if cuda else torch.from_numpy(train_observation_numpy).float()
    valid_observation = torch.from_numpy(valid_observation_numpy).float().cuda() if cuda else torch.from_numpy(valid_observation_numpy).float()
    train_observation_mean = Variable(
        torch.from_numpy(np.mean(train_observation_numpy, axis=0, keepdims=True)).float().cuda() if cuda else
        torch.from_numpy(np.mean(train_observation_numpy, axis=0, keepdims=True)).float()
    )
    train_observation_bias = torch.log(1 / torch.clamp(train_observation_mean, 0.001, 0.999) - 1.)

    # Train
    num_epochs = 2000
    batch_size = 24
    num_valid_particles = 100
    train_observation_dataloader = torch.utils.data.DataLoader(train_observation, batch_size=batch_size, shuffle=True)
    valid_observation_dataloader = torch.utils.data.DataLoader(valid_observation, batch_size=batch_size, shuffle=False)
    if args.estimator == 'reinforce':
        if args.architecture == 'L1':
            vae = VAEL1(args.estimator, train_observation_mean, train_observation_bias)
            if cuda:
                vae.cuda()
            vae_optimizer = torch.optim.Adam(vae.parameters())
            train_elbo_history, valid_elbo_history, vae, vae_optimizer = train_vae(
                vae,
                vae_optimizer,
                train_observation_dataloader,
                valid_observation_dataloader,
                num_epochs,
                num_valid_particles
            )
        elif args.architecture == 'L2':
            raise NotImplementedError
        elif args.architecture == 'NL':
            raise NotImplementedError

        filename = '{}_{}_{}_train_elbo_history.npy'.format(args.dataset, args.estimator, args.architecture)
        np.save(filename, train_elbo_history)
        logging.info('Saved to {}'.format(filename))

        filename = '{}_{}_{}_valid_elbo_history.npy'.format(args.dataset, args.estimator, args.architecture)
        np.save(filename, valid_elbo_history)
        logging.info('Saved to {}'.format(filename))

        filename = '{}_{}_{}_vae.pt'.format(args.dataset, args.estimator, args.architecture)
        torch.save(vae.state_dict(), filename)
        logging.info('Saved to {}'.format(filename))

        filename = '{}_{}_{}_vae_optimizer.pt'.format(args.dataset, args.estimator, args.architecture)
        torch.save(vae_optimizer.state_dict(), filename)
        logging.info('Saved to {}'.format(filename))

        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(3.25, 2)
        ax.set_axisbelow(True)
        ax.grid(True, linestyle='dashed', axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.plot(train_elbo_history, color='black', linestyle='dashed', label='train')
        ax.plot(valid_elbo_history, color='black', label='valid')
        ax.set_ylabel('ELBO')
        ax.set_xlabel('Epoch')
        ax.legend()

        fig.tight_layout()
        filename = '{}_{}_{}_loss.pdf'.format(args.dataset, args.estimator, args.architecture)
        fig.savefig(filename, bbox_inches='tight')
        logging.info('Saved to {}'.format(filename))
    elif args.estimator == 'relax':
        if args.architecture == 'L1':
            vae = VAEL1(args.estimator, train_observation_mean, train_observation_bias)
            if cuda:
                vae.cuda()
            vae_optimizer = torch.optim.Adam(vae.vae_params)
            control_variate_optimizer = torch.optim.Adam(vae.control_variate.parameters())

            train_elbo_history, valid_elbo_history, vae, vae_optimizer, control_variate_optimizer = train_vae_relax(
                vae,
                vae_optimizer,
                control_variate_optimizer,
                train_observation_dataloader,
                valid_observation_dataloader,
                num_epochs,
                num_valid_particles
            )
        elif args.architecture == 'L2':
            raise NotImplementedError
        elif args.architecture == 'NL':
            raise NotImplementedError

        filename = '{}_{}_{}_train_elbo_history.npy'.format(args.dataset, args.estimator, args.architecture)
        np.save(filename, train_elbo_history)
        logging.info('Saved to {}'.format(filename))

        filename = '{}_{}_{}_valid_elbo_history.npy'.format(args.dataset, args.estimator, args.architecture)
        np.save(filename, valid_elbo_history)
        logging.info('Saved to {}'.format(filename))

        filename = '{}_{}_{}_vae.pt'.format(args.dataset, args.estimator, args.architecture)
        torch.save(vae.state_dict(), filename)
        logging.info('Saved to {}'.format(filename))

        filename = '{}_{}_{}_vae_optimizer.pt'.format(args.dataset, args.estimator, args.architecture)
        torch.save(vae_optimizer.state_dict(), filename)
        logging.info('Saved to {}'.format(filename))

        filename = '{}_{}_{}_control_variate_optimizer.pt'.format(args.dataset, args.estimator, args.architecture)
        torch.save(control_variate_optimizer.state_dict(), filename)
        logging.info('Saved to {}'.format(filename))

        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(3.25, 2)
        ax.set_axisbelow(True)
        ax.grid(True, linestyle='dashed', axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.plot(train_elbo_history, color='black', linestyle='dashed', label='train')
        ax.plot(valid_elbo_history, color='black', label='valid')
        ax.set_ylabel('ELBO')
        ax.set_xlabel('Epoch')
        ax.legend()

        fig.tight_layout()
        filename = '{}_{}_{}_loss.pdf'.format(args.dataset, args.estimator, args.architecture)
        fig.savefig(filename, bbox_inches='tight')
        logging.info('Saved to {}'.format(filename))
    elif args.estimator == 'cdae':
        if args.architecture == 'L1':
            generative_network = GenerativeNetworkL1(train_observation_bias)
            inference_network = InferenceNetworkL1(train_observation_mean)
            if cuda:
                generative_network.cuda()
                inference_network.cuda()

            theta_optimizer = torch.optim.Adam(generative_network.parameters())
            phi_optimizer = torch.optim.Adam(inference_network.parameters())
            num_theta_train_particles = 1

            train_theta_loss_history, train_phi_loss_history, valid_elbo_history, generative_network, inference_network, theta_optimizer, phi_optimizer = train_cdae(
                generative_network,
                inference_network,
                theta_optimizer,
                phi_optimizer,
                num_theta_train_particles,
                train_observation_dataloader,
                valid_observation_dataloader,
                num_epochs,
                num_valid_particles
            )
        elif args.architecture == 'L2':
            raise NotImplementedError
        elif args.architecture == 'NL':
            raise NotImplementedError

        filename = '{}_{}_{}_train_theta_loss_history.npy'.format(args.dataset, args.estimator, args.architecture)
        np.save(filename, train_theta_loss_history)
        logging.info('Saved to {}'.format(filename))

        filename = '{}_{}_{}_train_phi_loss_history.npy'.format(args.dataset, args.estimator, args.architecture)
        np.save(filename, train_phi_loss_history)
        logging.info('Saved to {}'.format(filename))

        filename = '{}_{}_{}_valid_elbo_history.npy'.format(args.dataset, args.estimator, args.architecture)
        np.save(filename, valid_elbo_history)
        logging.info('Saved to {}'.format(filename))

        filename = '{}_{}_{}_generative_network.pt'.format(args.dataset, args.estimator, args.architecture)
        torch.save(generative_network.state_dict(), filename)
        logging.info('Saved to {}'.format(filename))

        filename = '{}_{}_{}_inference_network.pt'.format(args.dataset, args.estimator, args.architecture)
        torch.save(inference_network.state_dict(), filename)
        logging.info('Saved to {}'.format(filename))

        filename = '{}_{}_{}_theta_optimizer.pt'.format(args.dataset, args.estimator, args.architecture)
        torch.save(theta_optimizer.state_dict(), filename)
        logging.info('Saved to {}'.format(filename))

        filename = '{}_{}_{}_phi_optimizer.pt'.format(args.dataset, args.estimator, args.architecture)
        torch.save(phi_optimizer.state_dict(), filename)
        logging.info('Saved to {}'.format(filename))

        fig, axs = plt.subplots(3, 1, sharex=True)
        fig.set_size_inches(3.25, 3.25)

        for ax in axs:
            ax.set_axisbelow(True)
            ax.grid(True, linestyle='dashed', axis='y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        axs[0].plot(valid_elbo_history, color='black')
        axs[0].set_ylabel('Valid ELBO')

        axs[1].plot(train_theta_loss_history, color='black')
        axs[1].set_ylabel('$\\theta$ loss')

        axs[2].plot(train_phi_loss_history, color='black')
        axs[2].set_ylabel('$\phi$ loss')

        axs[2].set_xlabel('Epoch')
        fig.tight_layout()
        filename = '{}_{}_{}_loss.pdf'.format(args.dataset, args.estimator, args.architecture)
        fig.savefig(filename, bbox_inches='tight')
        logging.info('Saved to {}'.format(filename))


if __name__ == '__main__':
    main()
