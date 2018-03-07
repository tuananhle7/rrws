import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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


def main():
    true_log_evidence = np.load('true_log_evidence.npy')
    num_mixtures = np.load('num_mixtures.npy')
    num_iterations = np.load('num_iterations.npy')
    logging_interval = np.load('logging_interval.npy')
    true_ess = np.load('true_ess.npy')
    logging_iterations = np.arange(0, num_iterations, logging_interval)

    iwae_reinforce_log_evidence_history, iwae_reinforce_elbo_history, iwae_reinforce_p_mixture_probs_ess_history, iwae_reinforce_p_mixture_probs_norm_history, iwae_reinforce_mean_multiplier_history, iwae_reinforce_p_grad_std_history, iwae_reinforce_q_grad_std_history = np.load('iwae_reinforce_log_evidence_history.npy'), np.load('iwae_reinforce_elbo_history.npy'), np.load('iwae_reinforce_p_mixture_probs_ess_history.npy'), np.load('iwae_reinforce_p_mixture_probs_norm_history.npy'), np.load('iwae_reinforce_mean_multiplier_history.npy'), np.load('iwae_reinforce_p_grad_std_history.npy'), np.load('iwae_reinforce_q_grad_std_history.npy')

    iwae_vimco_log_evidence_history, iwae_vimco_elbo_history, iwae_vimco_p_mixture_probs_ess_history, iwae_vimco_p_mixture_probs_norm_history, iwae_vimco_mean_multiplier_history, iwae_vimco_p_grad_std_history, iwae_vimco_q_grad_std_history = np.load('iwae_vimco_log_evidence_history.npy'), np.load('iwae_vimco_elbo_history.npy'), np.load('iwae_vimco_p_mixture_probs_ess_history.npy'), np.load('iwae_vimco_p_mixture_probs_norm_history.npy'), np.load('iwae_vimco_mean_multiplier_history.npy'), np.load('iwae_vimco_p_grad_std_history.npy'), np.load('iwae_vimco_q_grad_std_history.npy')

    ws_log_evidence_history, ws_p_mixture_probs_ess_history, ws_p_mixture_probs_norm_history, ws_mean_multiplier_history, ws_p_grad_std_history, ws_q_grad_std_history = np.load('ws_log_evidence_history.npy'), np.load('ws_p_mixture_probs_ess_history.npy'), np.load('ws_p_mixture_probs_norm_history.npy'), np.load('ws_mean_multiplier_history.npy'), np.load('ws_p_grad_std_history.npy'), np.load('ws_q_grad_std_history.npy')

    ww_log_evidence_history, ww_p_mixture_probs_ess_history, ww_p_mixture_probs_norm_history, ww_mean_multiplier_history, ww_p_grad_std_history, ww_q_grad_std_history = np.load('ww_log_evidence_history.npy'), np.load('ww_p_mixture_probs_ess_history.npy'), np.load('ww_p_mixture_probs_norm_history.npy'), np.load('ww_mean_multiplier_history.npy'), np.load('ww_p_grad_std_history.npy'), np.load('ww_q_grad_std_history.npy')

    wsw_log_evidence_history, wsw_p_mixture_probs_ess_history, wsw_p_mixture_probs_norm_history, wsw_mean_multiplier_history, wsw_p_grad_std_history, wsw_q_grad_std_history = np.load('wsw_log_evidence_history.npy'), np.load('wsw_p_mixture_probs_ess_history.npy'), np.load('wsw_p_mixture_probs_norm_history.npy'), np.load('wsw_mean_multiplier_history.npy'), np.load('wsw_p_grad_std_history.npy'), np.load('wsw_q_grad_std_history.npy')

    # Plotting
    fig, axs = plt.subplots(5, 1, sharex=True)
    fig.set_size_inches(3.25, 6)

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axs[-1].set_xlabel('Iteration')

    axs[0].plot(logging_iterations, iwae_reinforce_log_evidence_history, linestyle=':', color='0.5', label='reinforce')
    axs[0].plot(logging_iterations, iwae_vimco_log_evidence_history, linestyle=':', color='black', label='vimco')
    axs[0].plot(logging_iterations, ws_log_evidence_history, linestyle='-.',color='0.5', label='ws')
    axs[0].plot(logging_iterations, ww_log_evidence_history, linestyle='--',color='0.8', label='ww')
    axs[0].plot(logging_iterations, wsw_log_evidence_history, linestyle='--',color='0.5', label='wsw')
    axs[0].axhline(true_log_evidence, linestyle='-', color='black', label='true')
    axs[0].set_ylabel('Avg. test\nlog evidence')
    axs[0].set_ylim(-10, 0)

    axs[1].plot(logging_iterations, iwae_reinforce_p_grad_std_history, linestyle=':', color='0.5', label='reinforce')
    axs[1].plot(logging_iterations, iwae_vimco_p_grad_std_history, linestyle=':', color='black', label='vimco')
    axs[1].plot(logging_iterations, ws_p_grad_std_history, linestyle='-.',color='0.5', label='ws')
    axs[1].plot(logging_iterations, ww_p_grad_std_history, linestyle='--',color='0.8', label='ww')
    axs[1].plot(logging_iterations, wsw_p_grad_std_history, linestyle='--',color='0.5', label='wsw')
    axs[1].set_ylim(0)
    axs[1].set_ylabel('Avg. std. of $\\theta$ \n gradient est.')

    axs[2].plot(logging_iterations, iwae_reinforce_q_grad_std_history, linestyle=':', color='0.5', label='reinforce')
    axs[2].plot(logging_iterations, iwae_vimco_q_grad_std_history, linestyle=':', color='black', label='vimco')
    axs[2].plot(logging_iterations, ws_q_grad_std_history, linestyle='-.',color='0.5', label='ws')
    axs[2].plot(logging_iterations, ww_q_grad_std_history, linestyle='--',color='0.8', label='ww')
    axs[2].plot(logging_iterations, wsw_q_grad_std_history, linestyle='--',color='0.5', label='wsw')
    axs[2].set_yscale('log')
    # axs[2].set_ylim(0)
    axs[2].set_ylabel('Avg. std. of $\phi$ \n gradient est.')

    axs[3].plot(logging_iterations, iwae_reinforce_p_mixture_probs_norm_history, linestyle=':', color='0.5', label='reinforce')
    axs[3].plot(logging_iterations, iwae_vimco_p_mixture_probs_norm_history, linestyle=':', color='black', label='vimco')
    axs[3].plot(logging_iterations, ws_p_mixture_probs_norm_history, linestyle='-.',color='0.5', label='ws')
    axs[3].plot(logging_iterations, ww_p_mixture_probs_norm_history, linestyle='--',color='0.8', label='ww')
    axs[3].plot(logging_iterations, wsw_p_mixture_probs_norm_history, linestyle='--',color='0.5', label='wsw')
    axs[3].set_ylim(0)
    axs[3].set_ylabel('Norm of mixture probs.\n to true mixture probs.')

    axs[4].plot(logging_iterations, iwae_reinforce_mean_multiplier_history, linestyle=':', color='0.5', label='reinforce')
    axs[4].plot(logging_iterations, iwae_vimco_mean_multiplier_history, linestyle=':', color='black', label='vimco')
    axs[4].plot(logging_iterations, ws_mean_multiplier_history, linestyle='-.',color='0.5', label='ws')
    axs[4].plot(logging_iterations, ww_mean_multiplier_history, linestyle='--',color='0.8', label='ww')
    axs[4].plot(logging_iterations, wsw_mean_multiplier_history, linestyle='--',color='0.5', label='wsw')
    axs[4].set_ylim(0)
    axs[4].set_ylabel('Mean multiplier')

    axs[-1].legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.4))

    fig.tight_layout()
    filename = 'gmm.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))


if __name__ == '__main__':
    main()
