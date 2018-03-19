import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

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
plt.rc('lines', linewidth=0.8)           # line thickness


def main(args):
    true_log_evidence = np_load('true_log_evidence.npy')
    num_mixtures = np_load('num_mixtures.npy')
    num_iterations = np_load('num_iterations.npy')
    logging_interval = np_load('logging_interval.npy')
    logging_iterations = np.arange(0, num_iterations, logging_interval)

    # Plotting
    fig, axs = plt.subplots(6, 1, sharex=True)
    fig.set_size_inches(3.25, 8)

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    ## IWAE
    iwae_filenames = ['log_evidence_history', 'elbo_history', 'posterior_norm_history', 'true_posterior_norm_history', 'p_mixture_probs_norm_history', 'mean_multiplier_history', 'p_grad_std_history', 'q_grad_std_history']

    if args.all or args.reinforce:
        iwae_reinforce = dict(zip(
            iwae_filenames,
            map(
                lambda iwae_filename: np_load('iwae_reinforce_{}'.format(iwae_filename)),
                iwae_filenames
            )
        ))

        axs[0].plot(logging_iterations, iwae_reinforce['log_evidence_history'], linestyle=':', color='0.5', label='reinforce')
        axs[1].plot(logging_iterations, iwae_reinforce['p_grad_std_history'], linestyle=':', color='0.5', label='reinforce')
        axs[2].plot(logging_iterations, iwae_reinforce['q_grad_std_history'], linestyle=':', color='0.5', label='reinforce')
        axs[3].plot(logging_iterations, iwae_reinforce['p_mixture_probs_norm_history'], linestyle=':', color='0.5', label='reinforce')
        axs[4].plot(logging_iterations, iwae_reinforce['posterior_norm_history'], linestyle=':', color='0.5', label='reinforce')
        axs[5].plot(logging_iterations, iwae_reinforce['true_posterior_norm_history'], linestyle=':', color='0.5', label='reinforce')

    if args.all or args.vimco:
        iwae_vimco = dict(zip(
            iwae_filenames,
            map(
                lambda iwae_filename: np_load('iwae_vimco_{}'.format(iwae_filename)),
                iwae_filenames
            )
        ))

        axs[0].plot(logging_iterations, iwae_vimco['log_evidence_history'], linestyle=':', color='black', label='vimco')
        axs[1].plot(logging_iterations, iwae_vimco['p_grad_std_history'], linestyle=':', color='black', label='vimco')
        axs[2].plot(logging_iterations, iwae_vimco['q_grad_std_history'], linestyle=':', color='black', label='vimco')
        axs[3].plot(logging_iterations, iwae_vimco['p_mixture_probs_norm_history'], linestyle=':', color='black', label='vimco')
        axs[4].plot(logging_iterations, iwae_vimco['posterior_norm_history'], linestyle=':', color='black', label='reinforce')
        axs[5].plot(logging_iterations, iwae_vimco['true_posterior_norm_history'], linestyle=':', color='black', label='reinforce')

    ## RWS
    rws_filenames = ['log_evidence_history', 'posterior_norm_history', 'true_posterior_norm_history', 'p_mixture_probs_norm_history', 'mean_multiplier_history', 'p_grad_std_history', 'q_grad_std_history']

    if args.all or args.ws:
        ws = dict(zip(
            rws_filenames,
            map(
                lambda rws_filename: np_load('ws_{}'.format(rws_filename)),
                rws_filenames
            )
        ))

        axs[0].plot(logging_iterations, ws['log_evidence_history'], linestyle='-.',color='0.5', label='ws')
        axs[1].plot(logging_iterations, ws['p_grad_std_history'], linestyle='-.',color='0.5', label='ws')
        axs[2].plot(logging_iterations, ws['q_grad_std_history'], linestyle='-.',color='0.5', label='ws')
        axs[3].plot(logging_iterations, ws['p_mixture_probs_norm_history'], linestyle='-.',color='0.5', label='ws')
        axs[4].plot(logging_iterations, ws['posterior_norm_history'], linestyle='-.',color='0.5', label='ws')
        axs[5].plot(logging_iterations, ws['true_posterior_norm_history'], linestyle='-.',color='0.5', label='ws')

    if args.all or args.ww:
        ww = dict(zip(
            rws_filenames,
            map(
                lambda rws_filename: np_load('ww_{}'.format(rws_filename)),
                rws_filenames
            )
        ))

        axs[0].plot(logging_iterations, ww['log_evidence_history'], linestyle='--',color='0.8', label='ww')
        axs[1].plot(logging_iterations, ww['p_grad_std_history'], linestyle='--',color='0.8', label='ww')
        axs[2].plot(logging_iterations, ww['q_grad_std_history'], linestyle='--',color='0.8', label='ww')
        axs[3].plot(logging_iterations, ww['p_mixture_probs_norm_history'], linestyle='--',color='0.8', label='ww')
        axs[4].plot(logging_iterations, ww['posterior_norm_history'], linestyle='--',color='0.8', label='ww')
        axs[5].plot(logging_iterations, ww['true_posterior_norm_history'], linestyle='--',color='0.8', label='ww')

    if args.all or args.wsw:
        wsw = dict(zip(
            rws_filenames,
            map(
                lambda rws_filename: np_load('wsw_{}'.format(rws_filename)),
                rws_filenames
            )
        ))

        axs[0].plot(logging_iterations, wsw['log_evidence_history'], linestyle='--',color='0.5', label='wsw')
        axs[1].plot(logging_iterations, wsw['p_grad_std_history'], linestyle='--',color='0.5', label='wsw')
        axs[2].plot(logging_iterations, wsw['q_grad_std_history'], linestyle='--',color='0.5', label='wsw')
        axs[3].plot(logging_iterations, wsw['p_mixture_probs_norm_history'], linestyle='--',color='0.5', label='wsw')
        axs[4].plot(logging_iterations, wsw['posterior_norm_history'], linestyle='--',color='0.5', label='wsw')
        axs[5].plot(logging_iterations, wsw['true_posterior_norm_history'], linestyle='--',color='0.5', label='wsw')

    if args.all or args.wswa:
        wswa = dict(zip(
            rws_filenames,
            map(
                lambda rws_filename: np_load('wswa_{}'.format(rws_filename)),
                rws_filenames
            )
        ))

        axs[0].plot(logging_iterations, wswa['log_evidence_history'], linestyle='-',color='0.5', label='wswa')
        axs[1].plot(logging_iterations, wswa['p_grad_std_history'], linestyle='-',color='0.5', label='wswa')
        axs[2].plot(logging_iterations, wswa['q_grad_std_history'], linestyle='-',color='0.5', label='wswa')
        axs[3].plot(logging_iterations, wswa['p_mixture_probs_norm_history'], linestyle='-',color='0.5', label='wswa')
        axs[4].plot(logging_iterations, wswa['posterior_norm_history'], linestyle='-',color='0.5', label='wswa')
        axs[5].plot(logging_iterations, wswa['true_posterior_norm_history'], linestyle='-',color='0.5', label='wswa')

    axs[0].axhline(true_log_evidence, linestyle='-', color='black', label='true')
    axs[0].set_ylabel('Avg. test\nlog evidence')
    # axs[0].set_ylim(-7, -5)

    axs[1].set_ylim(0)
    axs[1].set_ylabel('Avg. std. of $\\theta$ \n gradient est.')

    axs[2].set_yscale('log')
    # axs[2].set_ylim(0)
    axs[2].set_ylabel('Avg. std. of $\phi$ \n gradient est.')

    axs[3].set_ylim(0)
    axs[3].set_ylabel('L2 of mixture probs.\n to true mixture probs.')

    axs[4].set_ylabel('L2 of q to current p')

    axs[5].set_ylabel('L2 of q to true p')

    axs[-1].legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.4))
    axs[-1].set_xlabel('Iteration')

    fig.tight_layout()
    filename = 'gmm.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))


def np_load(filename):
    return np.load('{}_{:d}_{}.npy'.format(
        os.path.splitext(filename)[0], SEED, UID
    ))


# os.path.splitext(filename)[0]
# globals
SEED = 1
UID = ''


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='GMM open universe')
    parser.add_argument('--uid', type=str, default='', metavar='U',
                        help='run UID')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='run seed')
    parser.add_argument('--all', action='store_true', default=False)
    parser.add_argument('--reinforce', action='store_true', default=False)
    parser.add_argument('--vimco', action='store_true', default=False)
    parser.add_argument('--ws', action='store_true', default=False)
    parser.add_argument('--ww', action='store_true', default=False)
    parser.add_argument('--wsw', action='store_true', default=False)
    parser.add_argument('--wswa', action='store_true', default=False)
    args = parser.parse_args()
    SEED = args.seed
    UID = args.uid

    main(args)
