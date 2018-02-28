import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

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


def get_proposal_params(prior_mean, prior_std, obs_std):
    posterior_var = 1 / (1 / prior_std**2 + 1 / obs_std**2)
    posterior_std = np.sqrt(posterior_var)
    multiplier = posterior_var / obs_std**2
    offset = posterior_var * prior_mean / prior_std**2

    return multiplier, offset, posterior_std


def main():
    true_prior_mean = 0
    true_prior_std = 1
    true_obs_std = 1

    init_prior_mean = 10
    prior_std = true_prior_std
    init_obs_std = 0.5
    init_multiplier = 2
    init_offset = 2
    init_std = 2

    num_iterations = 10000
    learning_rate = 0.01

    true_multiplier, true_offset, true_std = get_proposal_params(true_prior_mean, true_prior_std, true_obs_std)

    # IWAE
    iwae_num_particles = 10
    iwae_prior_mean_history, iwae_obs_std_history, iwae_multiplier_history, iwae_offset_history, iwae_std_history, iwae_loss_history, iwae_true_multiplier_history, iwae_true_offset_history, iwae_true_std_history = np.load('iwae_prior_mean_history.npy'), np.load('iwae_obs_std_history.npy'), np.load('iwae_multiplier_history.npy'), np.load('iwae_offset_history.npy'), np.load('iwae_std_history.npy'), np.load('iwae_loss_history.npy'), np.load('iwae_true_multiplier_history.npy'), np.load('iwae_true_offset_history.npy'), np.load('iwae_true_std_history.npy')

    # RWS
    rws_num_particles = 10
    ws_prior_mean_history, ws_obs_std_history, ws_multiplier_history, ws_offset_history, ws_std_history, ws_wake_theta_loss_history, ws_sleep_phi_loss_history, ws_true_multiplier_history, ws_true_offset_history, ws_true_std_history = np.load('ws_prior_mean_history.npy'), np.load('ws_obs_std_history.npy'), np.load('ws_multiplier_history.npy'), np.load('ws_offset_history.npy'), np.load('ws_std_history.npy'), np.load('ws_wake_theta_loss_history.npy'), np.load('ws_sleep_phi_loss_history.npy'), np.load('ws_true_multiplier_history.npy'), np.load('ws_true_offset_history.npy'), np.load('ws_true_std_history.npy')

    ww_prior_mean_history, ww_obs_std_history, ww_multiplier_history, ww_offset_history, ww_std_history, ww_wake_theta_loss_history, ww_wake_phi_loss_history, ww_true_multiplier_history, ww_true_offset_history, ww_true_std_history = np.load('ww_prior_mean_history.npy'), np.load('ww_obs_std_history.npy'), np.load('ww_multiplier_history.npy'), np.load('ww_offset_history.npy'), np.load('ww_std_history.npy'), np.load('ww_wake_theta_loss_history.npy'), np.load('ww_wake_phi_loss_history.npy'), np.load('ww_true_multiplier_history.npy'), np.load('ww_true_offset_history.npy'), np.load('ww_true_std_history.npy')

    wsw_prior_mean_history, wsw_obs_std_history, wsw_multiplier_history, wsw_offset_history, wsw_std_history, wsw_wake_theta_loss_history, wsw_sleep_phi_loss_history, wsw_wake_phi_loss_history, wsw_true_multiplier_history, wsw_true_offset_history, wsw_true_std_history = np.load('wsw_prior_mean_history.npy'), np.load('wsw_obs_std_history.npy'), np.load('wsw_multiplier_history.npy'), np.load('wsw_offset_history.npy'), np.load('wsw_std_history.npy'), np.load('wsw_wake_theta_loss_history.npy'), np.load('wsw_sleep_phi_loss_history.npy'), np.load('wsw_wake_phi_loss_history.npy'), np.load('wsw_true_multiplier_history.npy'), np.load('wsw_true_offset_history.npy'), np.load('wsw_true_std_history.npy')

    waw_prior_mean_history, waw_obs_std_history, waw_multiplier_history, waw_offset_history, waw_std_history, waw_wake_theta_loss_history, waw_wake_phi_loss_history, waw_anneal_factor_history, waw_true_multiplier_history, waw_true_offset_history, waw_true_std_history = np.load('waw_prior_mean_history.npy'), np.load('waw_obs_std_history.npy'), np.load('waw_multiplier_history.npy'), np.load('waw_offset_history.npy'), np.load('waw_std_history.npy'), np.load('waw_wake_theta_loss_history.npy'), np.load('waw_wake_phi_loss_history.npy'), np.load('waw_anneal_factor_history.npy'), np.load('waw_true_multiplier_history.npy'), np.load('waw_true_offset_history.npy'), np.load('waw_true_std_history.npy')

    wsaw_prior_mean_history, wsaw_obs_std_history, wsaw_multiplier_history, wsaw_offset_history, wsaw_std_history, wsaw_wake_theta_loss_history, wsaw_wake_phi_loss_history, wsaw_anneal_factor_history, wsaw_true_multiplier_history, wsaw_true_offset_history, wsaw_true_std_history = np.load('wsaw_prior_mean_history.npy'), np.load('wsaw_obs_std_history.npy'), np.load('wsaw_multiplier_history.npy'), np.load('wsaw_offset_history.npy'), np.load('wsaw_std_history.npy'), np.load('wsaw_wake_theta_loss_history.npy'), np.load('wsaw_wake_phi_loss_history.npy'), np.load('wsaw_anneal_factor_history.npy'), np.load('wsaw_true_multiplier_history.npy'), np.load('wsaw_true_offset_history.npy'), np.load('wsaw_true_std_history.npy')

    # Plotting
    fig, axs = plt.subplots(5, 1, sharex=True)

    fig.set_size_inches(3.25, 4)

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # axs[0].set_title('Generative network parameters')
    axs[0].plot(iwae_prior_mean_history, color='black', linestyle=':', label='iwae-{}'.format(iwae_num_particles))
    axs[0].plot(ws_prior_mean_history, color='black', linestyle='-.', label='ws-{}'.format(rws_num_particles))
    axs[0].plot(ww_prior_mean_history, color='0.8', linestyle='-', label='ww-{}'.format(rws_num_particles))
    axs[0].plot(wsw_prior_mean_history, color='0.4', linestyle='-', label='wsw-{}'.format(rws_num_particles))
    axs[0].plot(waw_prior_mean_history, color='0.8', linestyle='--', label='waw-{}'.format(rws_num_particles))
    axs[0].plot(wsaw_prior_mean_history, color='0.4', linestyle='--', label='wsaw-{}'.format(rws_num_particles))
    axs[0].axhline(true_prior_mean, color='black', label='true')
    axs[0].set_ylabel('$\mu_0$')

    axs[1].plot(iwae_obs_std_history, color='black', linestyle=':', label='iwae-{}'.format(iwae_num_particles))
    axs[1].plot(ws_obs_std_history, color='black', linestyle='-.', label='ws-{}'.format(rws_num_particles))
    axs[1].plot(ww_obs_std_history, color='0.8', linestyle='-', label='ww-{}'.format(rws_num_particles))
    axs[1].plot(wsw_obs_std_history, color='0.4', linestyle='-', label='wsw-{}'.format(rws_num_particles))
    axs[1].plot(waw_obs_std_history, color='0.8', linestyle='--', label='waw-{}'.format(rws_num_particles))
    axs[1].plot(wsaw_obs_std_history, color='0.4', linestyle='--', label='wsaw-{}'.format(rws_num_particles))
    axs[1].axhline(true_obs_std, color='black', label='true')
    axs[1].set_ylabel('$\sigma$')
    # axs[1].set_xlabel('Iteration')

    # axs[2].set_title('Inference network parameters')
    axs[2].plot(iwae_multiplier_history, color='black', linestyle=':', label='iwae-{}'.format(iwae_num_particles))
    axs[2].plot(ws_multiplier_history, color='black', linestyle='-.', label='ws-{}'.format(rws_num_particles))
    axs[2].plot(ww_multiplier_history, color='0.8', linestyle='-', label='ww-{}'.format(rws_num_particles))
    axs[2].plot(wsw_multiplier_history, color='0.4', linestyle='-', label='wsw-{}'.format(rws_num_particles))
    axs[2].plot(waw_multiplier_history, color='0.8', linestyle='--', label='waw-{}'.format(rws_num_particles))
    axs[2].plot(wsaw_multiplier_history, color='0.4', linestyle='--', label='wsaw-{}'.format(rws_num_particles))
    axs[2].axhline(true_multiplier, color='black', label='true')
    axs[2].set_ylabel('$a$')

    axs[3].plot(iwae_offset_history, color='black', linestyle=':', label='iwae-{}'.format(iwae_num_particles))
    axs[3].plot(ws_offset_history, color='black', linestyle='-.', label='ws-{}'.format(rws_num_particles))
    axs[3].plot(ww_offset_history, color='0.8', linestyle='-', label='ww-{}'.format(rws_num_particles))
    axs[3].plot(wsw_offset_history, color='0.4', linestyle='-', label='wsw-{}'.format(rws_num_particles))
    axs[3].plot(waw_offset_history, color='0.8', linestyle='--', label='waw-{}'.format(rws_num_particles))
    axs[3].plot(wsaw_offset_history, color='0.4', linestyle='--', label='wsaw-{}'.format(rws_num_particles))
    axs[3].axhline(true_offset, color='black', label='true')
    axs[3].set_ylabel('$b$')

    axs[4].plot(iwae_std_history, color='black', linestyle=':', label='iwae-{}'.format(iwae_num_particles))
    axs[4].plot(ws_std_history, color='black', linestyle='-.', label='ws-{}'.format(rws_num_particles))
    axs[4].plot(ww_std_history, color='0.8', linestyle='-', label='ww-{}'.format(rws_num_particles))
    axs[4].plot(wsw_std_history, color='0.4', linestyle='-', label='wsw-{}'.format(rws_num_particles))
    axs[4].plot(waw_std_history, color='0.8', linestyle='--', label='waw-{}'.format(rws_num_particles))
    axs[4].plot(wsaw_std_history, color='0.4', linestyle='--', label='wsaw-{}'.format(rws_num_particles))
    axs[4].axhline(true_std, color='black', label='true')
    axs[4].set_ylabel('$c$')
    axs[4].set_xlabel('Iteration')

    axs[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.6), ncol=4)
    fig.tight_layout()

    filename = 'gaussian.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))


if __name__ == '__main__':
    main()
