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
    true_ess = np_load('true_ess.npy')
    logging_iterations = np.arange(0, num_iterations, logging_interval)

    # Plotting
    fig, axs = plt.subplots(4, 1, sharex=True)
    fig.set_size_inches(3.25, 5)

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if args.all or args.reinforce:
        iwae_reinforce_log_evidence_history, iwae_reinforce_elbo_history, iwae_reinforce_p_mixture_probs_ess_history, iwae_reinforce_p_mixture_probs_norm_history, iwae_reinforce_mean_multiplier_history, iwae_reinforce_p_grad_std_history, iwae_reinforce_q_grad_std_history = np_load('iwae_reinforce_log_evidence_history'), np_load('iwae_reinforce_elbo_history'), np_load('iwae_reinforce_p_mixture_probs_ess_history'), np_load('iwae_reinforce_p_mixture_probs_norm_history'), np_load('iwae_reinforce_mean_multiplier_history'), np_load('iwae_reinforce_p_grad_std_history'), np_load('iwae_reinforce_q_grad_std_history')

        axs[0].plot(logging_iterations, iwae_reinforce_log_evidence_history, linestyle=':', color='0.5', label='reinforce')
        axs[1].plot(logging_iterations, iwae_reinforce_p_grad_std_history, linestyle=':', color='0.5', label='reinforce')
        axs[2].plot(logging_iterations, iwae_reinforce_q_grad_std_history, linestyle=':', color='0.5', label='reinforce')
        axs[3].plot(logging_iterations, iwae_reinforce_p_mixture_probs_norm_history, linestyle=':', color='0.5', label='reinforce')

    if args.all or args.vimco:
        iwae_vimco_log_evidence_history, iwae_vimco_elbo_history, iwae_vimco_p_mixture_probs_ess_history, iwae_vimco_p_mixture_probs_norm_history, iwae_vimco_mean_multiplier_history, iwae_vimco_p_grad_std_history, iwae_vimco_q_grad_std_history = np_load('iwae_vimco_log_evidence_history'), np_load('iwae_vimco_elbo_history'), np_load('iwae_vimco_p_mixture_probs_ess_history'), np_load('iwae_vimco_p_mixture_probs_norm_history'), np_load('iwae_vimco_mean_multiplier_history'), np_load('iwae_vimco_p_grad_std_history'), np_load('iwae_vimco_q_grad_std_history')

        axs[0].plot(logging_iterations, iwae_vimco_log_evidence_history, linestyle=':', color='black', label='vimco')
        axs[1].plot(logging_iterations, iwae_vimco_p_grad_std_history, linestyle=':', color='black', label='vimco')
        axs[2].plot(logging_iterations, iwae_vimco_q_grad_std_history, linestyle=':', color='black', label='vimco')
        axs[3].plot(logging_iterations, iwae_vimco_p_mixture_probs_norm_history, linestyle=':', color='black', label='vimco')

    if args.all or args.ws:
        ws_log_evidence_history, ws_p_mixture_probs_ess_history, ws_p_mixture_probs_norm_history, ws_mean_multiplier_history, ws_p_grad_std_history, ws_q_grad_std_history = np_load('ws_log_evidence_history'), np_load('ws_p_mixture_probs_ess_history'), np_load('ws_p_mixture_probs_norm_history'), np_load('ws_mean_multiplier_history'), np_load('ws_p_grad_std_history'), np_load('ws_q_grad_std_history')

        axs[0].plot(logging_iterations, ws_log_evidence_history, linestyle='-.',color='0.5', label='ws')
        axs[1].plot(logging_iterations, ws_p_grad_std_history, linestyle='-.',color='0.5', label='ws')
        axs[2].plot(logging_iterations, ws_q_grad_std_history, linestyle='-.',color='0.5', label='ws')
        axs[3].plot(logging_iterations, ws_p_mixture_probs_norm_history, linestyle='-.',color='0.5', label='ws')

    if args.all or args.ww:
        ww_log_evidence_history, ww_p_mixture_probs_ess_history, ww_p_mixture_probs_norm_history, ww_mean_multiplier_history, ww_p_grad_std_history, ww_q_grad_std_history = np_load('ww_log_evidence_history'), np_load('ww_p_mixture_probs_ess_history'), np_load('ww_p_mixture_probs_norm_history'), np_load('ww_mean_multiplier_history'), np_load('ww_p_grad_std_history'), np_load('ww_q_grad_std_history')

        axs[0].plot(logging_iterations, ww_log_evidence_history, linestyle='--',color='0.8', label='ww')
        axs[1].plot(logging_iterations, ww_p_grad_std_history, linestyle='--',color='0.8', label='ww')
        axs[2].plot(logging_iterations, ww_q_grad_std_history, linestyle='--',color='0.8', label='ww')
        axs[3].plot(logging_iterations, ww_p_mixture_probs_norm_history, linestyle='--',color='0.8', label='ww')

    if args.all or args.wsw:
        wsw_log_evidence_history, wsw_p_mixture_probs_ess_history, wsw_p_mixture_probs_norm_history, wsw_mean_multiplier_history, wsw_p_grad_std_history, wsw_q_grad_std_history = np_load('wsw_log_evidence_history'), np_load('wsw_p_mixture_probs_ess_history'), np_load('wsw_p_mixture_probs_norm_history'), np_load('wsw_mean_multiplier_history'), np_load('wsw_p_grad_std_history'), np_load('wsw_q_grad_std_history')

        axs[0].plot(logging_iterations, wsw_log_evidence_history, linestyle='--',color='0.5', label='wsw')
        axs[1].plot(logging_iterations, wsw_p_grad_std_history, linestyle='--',color='0.5', label='wsw')
        axs[2].plot(logging_iterations, wsw_q_grad_std_history, linestyle='--',color='0.5', label='wsw')
        axs[3].plot(logging_iterations, wsw_p_mixture_probs_norm_history, linestyle='--',color='0.5', label='wsw')

    if args.all or args.wswa:
        wswa_log_evidence_history, wswa_p_mixture_probs_ess_history, wswa_p_mixture_probs_norm_history, wswa_mean_multiplier_history, wswa_p_grad_std_history, wswa_q_grad_std_history = np_load('wswa_log_evidence_history'), np_load('wswa_p_mixture_probs_ess_history'), np_load('wswa_p_mixture_probs_norm_history'), np_load('wswa_mean_multiplier_history'), np_load('wswa_p_grad_std_history'), np_load('wswa_q_grad_std_history')

        axs[0].plot(logging_iterations, wswa_log_evidence_history, linestyle='-',color='0.5', label='wswa')
        axs[1].plot(logging_iterations, wswa_p_grad_std_history, linestyle='-',color='0.5', label='wswa')
        axs[2].plot(logging_iterations, wswa_q_grad_std_history, linestyle='-',color='0.5', label='wswa')
        axs[3].plot(logging_iterations, wswa_p_mixture_probs_norm_history, linestyle='-',color='0.5', label='wswa')

    axs[0].axhline(true_log_evidence, linestyle='-', color='black', label='true')
    axs[0].set_ylabel('Avg. test\nlog evidence')
    # axs[0].set_ylim(-7, -5)

    axs[1].set_ylim(0)
    axs[1].set_ylabel('Avg. std. of $\\theta$ \n gradient est.')

    axs[2].set_yscale('log')
    # axs[2].set_ylim(0)
    axs[2].set_ylabel('Avg. std. of $\phi$ \n gradient est.')

    axs[3].set_ylim(0)
    axs[3].set_ylabel('Norm of mixture probs.\n to true mixture probs.')

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
