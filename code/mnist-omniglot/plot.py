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

    # datasets = ['mnist', 'omniglot']
    # estimators = ['relax', 'cdae']
    # architectures = ['L1', 'L2', 'NL']
    datasets = ['omniglot', 'mnist']
    estimators = ['cdae']
    architectures = ['L1', 'L2', 'NL']

    valid_elbo_fig, valid_elbo_ax = plt.subplots(1, 1)
    valid_elbo_fig.set_size_inches(3.25, 2)
    for dataset in datasets:
        for estimator in estimators:
            for architecture in architectures:
                logging.info('{} {} {}:'.format(dataset, estimator, architecture))
                if estimator == 'cdae':
                    filename = '{}_{}_{}_train_theta_loss_history.npy'.format(dataset, estimator, architecture)
                    train_theta_loss_history = np.load(filename)

                    filename = '{}_{}_{}_train_phi_loss_history.npy'.format(dataset, estimator, architecture)
                    train_phi_loss_history = np.load(filename)

                    filename = '{}_{}_{}_valid_elbo_history.npy'.format(dataset, estimator, architecture)
                    valid_elbo_history = np.load(filename)

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
                    filename = '{}_{}_{}_loss.pdf'.format(dataset, estimator, architecture)
                    fig.savefig(filename, bbox_inches='tight')
                    logging.info('Saved to {}'.format(filename))

                    # Valid ELBO
                    valid_elbo_ax.plot(valid_elbo_history, label='{} {} {}'.format(dataset, estimator, architecture))

                logging.info('Best train theta loss ({} {} {}) = {}'.format(dataset, estimator, architecture, np.min(train_theta_loss_history)))
                logging.info('Best train phi loss ({} {} {}) = {}'.format(dataset, estimator, architecture, np.min(train_phi_loss_history)))
                logging.info('Best valid ELBO ({} {} {}) = {}'.format(dataset, estimator, architecture, np.max(valid_elbo_history)))

    valid_elbo_ax.set_axisbelow(True)
    valid_elbo_ax.grid(True, linestyle='dashed', axis='y')
    valid_elbo_ax.spines['top'].set_visible(False)
    valid_elbo_ax.spines['right'].set_visible(False)
    valid_elbo_ax.set_xlabel('Epoch')
    valid_elbo_ax.set_ylabel('Valid ELBO')
    valid_elbo_ax.legend(ncol=3, bbox_to_anchor=(0.5, -0.25), loc='upper center')
    valid_elbo_fig.tight_layout()
    filename = 'valid_elbo.pdf'
    valid_elbo_fig.savefig(filename, bbox_inches='tight')
    logging.info('Saved to {}'.format(filename))



if __name__ == '__main__':
    main()
