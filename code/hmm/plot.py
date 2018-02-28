from neural_nets import *
from util import *

import aesmc.util
import aesmc.wakesleep
import logging
import matplotlib.pyplot as plt
import numpy as np

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


def main():
    # Logging
    logging.basicConfig(
        datefmt='%Y-%m-%d %H:%M:%S',
        format='%(asctime)s %(message)s',
        level=logging.INFO
    )
    algorithm_list = ['is', 'smc']
    num_particles_list = [20, 100]

    initial_probs, true_transition_probs, obs_means, obs_vars = \
        read_model('model.csv')
    num_states = len(initial_probs)
    test_data = np.genfromtxt('test_data.csv', delimiter=',')

    transition_matrix_fig, transition_matrix_ax = plt.subplots(1, 1)
    transition_matrix_fig.set_size_inches(3.25, 2)
    for algorithm in algorithm_list:
        for num_particles in num_particles_list:
            # Transition matrix
            filename = '{}_{}_transition_matrix_history.npy'.format(algorithm, num_particles)
            transition_matrix_history = np.load(filename)
            num_epochs = transition_matrix_history.shape[0]
            transition_matrix_norm = np.zeros([num_epochs])
            for epoch_idx in range(num_epochs):
                transition_matrix_norm[epoch_idx] = np.linalg.norm(
                    transition_matrix_history[epoch_idx] -
                    np.array(true_transition_probs)
                )

            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(3.25, 1)
            ax.plot(transition_matrix_norm, color='black')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('$||A - A^*||$')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # fig.tight_layout()
            filename = '{}_{}_transition_matrix_norm.pdf'.format(algorithm, num_particles)
            fig.savefig(filename, bbox_inches='tight')
            logging.info('Saved to {}'.format(filename))

            transition_matrix_ax.plot(transition_matrix_norm[:], label='{} {}'.format(algorithm, num_particles))

            # Losses
            filename = '{}_{}_train_theta_loss_history.npy'.format(algorithm, num_particles)
            train_theta_loss_history = np.load(filename)

            filename = '{}_{}_train_phi_loss_history.npy'.format(algorithm, num_particles)
            train_phi_loss_history = np.load(filename)

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
            filename = '{}_{}_loss.pdf'.format(algorithm, num_particles)
            fig.savefig(filename, bbox_inches='tight')
            logging.info('Saved to {}'.format(filename))

            # Inference
            hmm_initial = HMMInitialDistribution(initial_probs)
            hmm_transition = HMMTransitionNetwork(num_states)
            hmm_emission = HMMEmissionDistribution(obs_means, obs_vars)
            hmm_proposal = HMMProposalNetwork(num_states)

            hmm_wakesleep = aesmc.wakesleep.WakeSleep(
                hmm_initial, hmm_transition, hmm_emission, hmm_proposal
            )
            hmm_wakesleep.load_state_dict(torch.load(
                    '{}_{}_hmm_wakesleep.pt'.format(algorithm, num_particles),
                    map_location=lambda storage, loc: storage
            ))
    transition_matrix_ax.set_xlabel('Epoch')
    transition_matrix_ax.set_ylabel('$||A - A^*||$')
    transition_matrix_ax.spines['top'].set_visible(False)
    transition_matrix_ax.spines['right'].set_visible(False)
    transition_matrix_ax.legend()
    transition_matrix_ax.set_yscale('log')
    filename = 'transition_matrix_norm.pdf'
    transition_matrix_fig.savefig(filename, bbox_inches='tight')
    logging.info('Saved to {}'.format(filename))


if __name__ == '__main__':
    main()
