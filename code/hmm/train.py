from neural_nets import *
from util import *

import data
import aesmc.state as st
import aesmc.util
import aesmc.wakesleep
import argparse
import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

SMALL_SIZE = 7
MEDIUM_SIZE = 9
BIGGER_SIZE = 11

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.switch_backend('agg')


def train_wakesleep(
    wakesleep, theta_optimizer, phi_optimizer, resample, num_particles,
    train_dataloader, num_epochs, num_theta_updates=1, num_phi_updates=1
):
    num_states = wakesleep.initial_network.num_states
    train_theta_loss_history = np.zeros([num_epochs])
    train_phi_loss_history = np.zeros([num_epochs])
    transition_matrix_history = np.zeros([num_epochs, num_states, num_states])

    train_start_time = datetime.datetime.now()
    for epoch_idx in range(num_epochs):
        for observations in iter(train_dataloader):
            # Train generative network (theta)
            batch_size = observations[0].size(0)
            observation_states = [st.State(
                batch_size=batch_size,
                num_particles=num_particles,
                values={
                    'y': Variable(observation.expand(batch_size, num_particles))
                }
            ) for observation in observations]
            for _ in range(num_theta_updates):
                theta_optimizer.zero_grad()
                loss = wakesleep.forward(
                    optimization_mode='generative',
                    resample=resample,
                    observation_states=observation_states
                )
                torch.mean(loss).backward()
                theta_optimizer.step()
            train_theta_loss_history[epoch_idx] += torch.sum(loss).data[0] / len(train_dataloader.dataset)
            transition_matrix_history[epoch_idx] = wakesleep.transition_network.get_transition_matrix().data.cpu().numpy() if aesmc.util.cuda else wakesleep.transition_network.get_transition_matrix().data.numpy()

            # Train inference network (phi)
            for _ in range(num_phi_updates):
                phi_optimizer.zero_grad()
                loss = wakesleep.forward(
                    optimization_mode='inference',
                    inference_optimization_num_samples=batch_size,
                    inference_optimization_num_timesteps=len(observations)
                )
                torch.mean(loss).backward()
                phi_optimizer.step()
            train_phi_loss_history[epoch_idx] += torch.sum(loss).data[0] / len(train_dataloader.dataset)

        if epoch_idx % 10 == 0:
            estimated_remaining_time = (datetime.datetime.now() - train_start_time) / (epoch_idx + 1) * (num_epochs - (epoch_idx + 1))
            logging.info('Epoch {}: theta {} | phi {} | est. {}'.format(
                epoch_idx,
                train_theta_loss_history[epoch_idx],
                train_phi_loss_history[epoch_idx],
                estimated_remaining_time
            ))

    return train_theta_loss_history, train_phi_loss_history, transition_matrix_history, wakesleep, theta_optimizer, phi_optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--resample', action='store_true')
    parser.add_argument('--num-timesteps', type=int, default=20)
    parser.add_argument('--num-particles', type=int, default=20)
    parser.add_argument('--num-epochs', type=int, default=10)
    args = parser.parse_args()

    # Seed and Cuda
    np.random.seed(1)
    if args.cuda:
        if torch.cuda.is_available():
            cuda = True
        else:
            cuda = False
            logging.info('Cuda is not available.')
    else:
        cuda = False
    aesmc.util.init(cuda, 1, None)

    # Logging
    logging.basicConfig(
        datefmt='%Y-%m-%d %H:%M:%S',
        format='%(asctime)s %(message)s',
        level=logging.INFO
    )

    hmm_train_dataloader = torch.utils.data.DataLoader(
        data.HMMDataset(num_timesteps=args.num_timesteps, filename='model.csv'),
        batch_size=32
    )
    initial_probs, transition_probs, obs_means, obs_vars = \
        read_model('model.csv')
    num_states = len(initial_probs)
    hmm_initial = HMMInitialDistribution(initial_probs)
    hmm_transition = HMMTransitionNetwork(num_states)
    hmm_emission = HMMEmissionDistribution(obs_means, obs_vars)
    hmm_proposal = HMMProposalNetwork(num_states)

    hmm_wakesleep = aesmc.wakesleep.WakeSleep(
        hmm_initial, hmm_transition, hmm_emission, hmm_proposal
    )
    theta_optimizer = torch.optim.Adam(hmm_wakesleep.generative_parameters())
    phi_optimizer = torch.optim.Adam(hmm_wakesleep.inference_parameters())

    train_theta_loss_history, train_phi_loss_history, \
        transition_matrix_history, hmm_wakesleep, theta_optimizer, \
        phi_optimizer = train_wakesleep(
            hmm_wakesleep, theta_optimizer, phi_optimizer, args.resample,
            args.num_particles, hmm_train_dataloader, args.num_epochs
        )

    filename = '{}_{}_train_theta_loss_history.npy'.format('smc' if args.resample else 'is', args.num_particles)
    np.save(filename, train_theta_loss_history)
    logging.info('Saved to {}'.format(filename))

    filename = '{}_{}_train_phi_loss_history.npy'.format('smc' if args.resample else 'is', args.num_particles)
    np.save(filename, train_phi_loss_history)
    logging.info('Saved to {}'.format(filename))

    filename = '{}_{}_transition_matrix_history.npy'.format('smc' if args.resample else 'is', args.num_particles)
    np.save(filename, transition_matrix_history)
    logging.info('Saved to {}'.format(filename))

    filename = '{}_{}_hmm_wakesleep.pt'.format('smc' if args.resample else 'is', args.num_particles)
    torch.save(hmm_wakesleep.state_dict(), filename)
    logging.info('Saved to {}'.format(filename))

    filename = '{}_{}_theta_optimizer.pt'.format('smc' if args.resample else 'is', args.num_particles)
    torch.save(theta_optimizer.state_dict(), filename)
    logging.info('Saved to {}'.format(filename))

    filename = '{}_{}_phi_optimizer.pt'.format('smc' if args.resample else 'is', args.num_particles)
    torch.save(phi_optimizer.state_dict(), filename)
    logging.info('Saved to {}'.format(filename))

    fig, axs = plt.subplots(2, 1, sharex=True)
    fig.set_size_inches(3.25, 2)

    for ax in axs:
        ax.set_axisbelow(True)
        ax.grid(True, linestyle='dashed', axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axs[0].plot(train_theta_loss_history, color='black')
    axs[0].set_ylabel('$\\theta$ loss')

    axs[1].plot(train_phi_loss_history, color='black')
    axs[1].set_ylabel('$\phi$ loss')

    axs[-1].set_xlabel('Epoch')
    fig.tight_layout()
    filename = '{}_{}_loss.pdf'.format('smc' if args.resample else 'is', args.num_particles)
    fig.savefig(filename, bbox_inches='tight')
    logging.info('Saved to {}'.format(filename))


if __name__ == '__main__':
    main()
