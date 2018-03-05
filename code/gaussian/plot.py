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


def plot_with_error_bars(ax, data, color, linestyle, label):
    # the error bars turn out to be negligibly small
    num_iterations = data.shape[1]
    mean = np.mean(data, axis=0)
    # std = np.std(data, axis=0)
    ax.plot(mean, color=color, linestyle=linestyle, label=label)
    # ax.fill_between(np.arange(num_iterations), mean - 3 * std, mean + 3 * std, color=color, alpha=0.3)
    return ax


def main():
    num_repeats = 10
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

    iwae_prior_mean_history = np.zeros([num_repeats, num_iterations])
    iwae_obs_std_history = np.zeros([num_repeats, num_iterations])
    iwae_multiplier_history = np.zeros([num_repeats, num_iterations])
    iwae_offset_history = np.zeros([num_repeats, num_iterations])
    iwae_std_history = np.zeros([num_repeats, num_iterations])
    iwae_loss_history = np.zeros([num_repeats, num_iterations])
    iwae_true_multiplier_history = np.zeros([num_repeats, num_iterations])
    iwae_true_offset_history = np.zeros([num_repeats, num_iterations])
    iwae_true_std_history = np.zeros([num_repeats, num_iterations])

    ws_prior_mean_history = np.zeros([num_repeats, num_iterations])
    ws_obs_std_history = np.zeros([num_repeats, num_iterations])
    ws_multiplier_history = np.zeros([num_repeats, num_iterations])
    ws_offset_history = np.zeros([num_repeats, num_iterations])
    ws_std_history = np.zeros([num_repeats, num_iterations])
    ws_wake_theta_loss_history = np.zeros([num_repeats, num_iterations])
    ws_sleep_phi_loss_history = np.zeros([num_repeats, num_iterations])
    ws_true_multiplier_history = np.zeros([num_repeats, num_iterations])
    ws_true_offset_history = np.zeros([num_repeats, num_iterations])
    ws_true_std_history = np.zeros([num_repeats, num_iterations])

    ww_prior_mean_history = np.zeros([num_repeats, num_iterations])
    ww_obs_std_history = np.zeros([num_repeats, num_iterations])
    ww_multiplier_history = np.zeros([num_repeats, num_iterations])
    ww_offset_history = np.zeros([num_repeats, num_iterations])
    ww_std_history = np.zeros([num_repeats, num_iterations])
    ww_wake_theta_loss_history = np.zeros([num_repeats, num_iterations])
    ww_wake_phi_loss_history = np.zeros([num_repeats, num_iterations])
    ww_true_multiplier_history = np.zeros([num_repeats, num_iterations])
    ww_true_offset_history = np.zeros([num_repeats, num_iterations])
    ww_true_std_history = np.zeros([num_repeats, num_iterations])

    wsw_prior_mean_history = np.zeros([num_repeats, num_iterations])
    wsw_obs_std_history = np.zeros([num_repeats, num_iterations])
    wsw_multiplier_history = np.zeros([num_repeats, num_iterations])
    wsw_offset_history = np.zeros([num_repeats, num_iterations])
    wsw_std_history = np.zeros([num_repeats, num_iterations])
    wsw_wake_theta_loss_history = np.zeros([num_repeats, num_iterations])
    wsw_sleep_phi_loss_history = np.zeros([num_repeats, num_iterations])
    wsw_wake_phi_loss_history = np.zeros([num_repeats, num_iterations])
    wsw_true_multiplier_history = np.zeros([num_repeats, num_iterations])
    wsw_true_offset_history = np.zeros([num_repeats, num_iterations])
    wsw_true_std_history = np.zeros([num_repeats, num_iterations])

    waw_prior_mean_history = np.zeros([num_repeats, num_iterations])
    waw_obs_std_history = np.zeros([num_repeats, num_iterations])
    waw_multiplier_history = np.zeros([num_repeats, num_iterations])
    waw_offset_history = np.zeros([num_repeats, num_iterations])
    waw_std_history = np.zeros([num_repeats, num_iterations])
    waw_wake_theta_loss_history = np.zeros([num_repeats, num_iterations])
    waw_wake_phi_loss_history = np.zeros([num_repeats, num_iterations])
    waw_anneal_factor_history = np.zeros([num_repeats, num_iterations])
    waw_true_multiplier_history = np.zeros([num_repeats, num_iterations])
    waw_true_offset_history = np.zeros([num_repeats, num_iterations])
    waw_true_std_history = np.zeros([num_repeats, num_iterations])

    wsaw_prior_mean_history = np.zeros([num_repeats, num_iterations])
    wsaw_obs_std_history = np.zeros([num_repeats, num_iterations])
    wsaw_multiplier_history = np.zeros([num_repeats, num_iterations])
    wsaw_offset_history = np.zeros([num_repeats, num_iterations])
    wsaw_std_history = np.zeros([num_repeats, num_iterations])
    wsaw_wake_theta_loss_history = np.zeros([num_repeats, num_iterations])
    wsaw_wake_phi_loss_history = np.zeros([num_repeats, num_iterations])
    wsaw_anneal_factor_history = np.zeros([num_repeats, num_iterations])
    wsaw_true_multiplier_history = np.zeros([num_repeats, num_iterations])
    wsaw_true_offset_history = np.zeros([num_repeats, num_iterations])
    wsaw_true_std_history = np.zeros([num_repeats, num_iterations])

    for repeat_idx in range(num_repeats):
        # IWAE
        iwae_num_particles = 10
        iwae_prior_mean_history[repeat_idx], iwae_obs_std_history[repeat_idx], iwae_multiplier_history[repeat_idx], iwae_offset_history[repeat_idx], iwae_std_history[repeat_idx], iwae_loss_history[repeat_idx], iwae_true_multiplier_history[repeat_idx], iwae_true_offset_history[repeat_idx], iwae_true_std_history[repeat_idx] = np.load('iwae_prior_mean_history.npy'), np.load('iwae_obs_std_history.npy'), np.load('iwae_multiplier_history.npy'), np.load('iwae_offset_history.npy'), np.load('iwae_std_history.npy'), np.load('iwae_loss_history.npy'), np.load('iwae_true_multiplier_history.npy'), np.load('iwae_true_offset_history.npy'), np.load('iwae_true_std_history.npy')

        # RWS
        rws_num_particles = 10
        ws_prior_mean_history[repeat_idx], ws_obs_std_history[repeat_idx], ws_multiplier_history[repeat_idx], ws_offset_history[repeat_idx], ws_std_history[repeat_idx], ws_wake_theta_loss_history[repeat_idx], ws_sleep_phi_loss_history[repeat_idx], ws_true_multiplier_history[repeat_idx], ws_true_offset_history[repeat_idx], ws_true_std_history[repeat_idx] = np.load('ws_prior_mean_history.npy'), np.load('ws_obs_std_history.npy'), np.load('ws_multiplier_history.npy'), np.load('ws_offset_history.npy'), np.load('ws_std_history.npy'), np.load('ws_wake_theta_loss_history.npy'), np.load('ws_sleep_phi_loss_history.npy'), np.load('ws_true_multiplier_history.npy'), np.load('ws_true_offset_history.npy'), np.load('ws_true_std_history.npy')

        ww_prior_mean_history[repeat_idx], ww_obs_std_history[repeat_idx], ww_multiplier_history[repeat_idx], ww_offset_history[repeat_idx], ww_std_history[repeat_idx], ww_wake_theta_loss_history[repeat_idx], ww_wake_phi_loss_history[repeat_idx], ww_true_multiplier_history[repeat_idx], ww_true_offset_history[repeat_idx], ww_true_std_history[repeat_idx] = np.load('ww_prior_mean_history.npy'), np.load('ww_obs_std_history.npy'), np.load('ww_multiplier_history.npy'), np.load('ww_offset_history.npy'), np.load('ww_std_history.npy'), np.load('ww_wake_theta_loss_history.npy'), np.load('ww_wake_phi_loss_history.npy'), np.load('ww_true_multiplier_history.npy'), np.load('ww_true_offset_history.npy'), np.load('ww_true_std_history.npy')

        wsw_prior_mean_history[repeat_idx], wsw_obs_std_history[repeat_idx], wsw_multiplier_history[repeat_idx], wsw_offset_history[repeat_idx], wsw_std_history[repeat_idx], wsw_wake_theta_loss_history[repeat_idx], wsw_sleep_phi_loss_history[repeat_idx], wsw_wake_phi_loss_history[repeat_idx], wsw_true_multiplier_history[repeat_idx], wsw_true_offset_history[repeat_idx], wsw_true_std_history[repeat_idx] = np.load('wsw_prior_mean_history.npy'), np.load('wsw_obs_std_history.npy'), np.load('wsw_multiplier_history.npy'), np.load('wsw_offset_history.npy'), np.load('wsw_std_history.npy'), np.load('wsw_wake_theta_loss_history.npy'), np.load('wsw_sleep_phi_loss_history.npy'), np.load('wsw_wake_phi_loss_history.npy'), np.load('wsw_true_multiplier_history.npy'), np.load('wsw_true_offset_history.npy'), np.load('wsw_true_std_history.npy')

        waw_prior_mean_history[repeat_idx], waw_obs_std_history[repeat_idx], waw_multiplier_history[repeat_idx], waw_offset_history[repeat_idx], waw_std_history[repeat_idx], waw_wake_theta_loss_history[repeat_idx], waw_wake_phi_loss_history[repeat_idx], waw_anneal_factor_history[repeat_idx], waw_true_multiplier_history[repeat_idx], waw_true_offset_history[repeat_idx], waw_true_std_history[repeat_idx] = np.load('waw_prior_mean_history.npy'), np.load('waw_obs_std_history.npy'), np.load('waw_multiplier_history.npy'), np.load('waw_offset_history.npy'), np.load('waw_std_history.npy'), np.load('waw_wake_theta_loss_history.npy'), np.load('waw_wake_phi_loss_history.npy'), np.load('waw_anneal_factor_history.npy'), np.load('waw_true_multiplier_history.npy'), np.load('waw_true_offset_history.npy'), np.load('waw_true_std_history.npy')

        wsaw_prior_mean_history[repeat_idx], wsaw_obs_std_history[repeat_idx], wsaw_multiplier_history[repeat_idx], wsaw_offset_history[repeat_idx], wsaw_std_history[repeat_idx], wsaw_wake_theta_loss_history[repeat_idx], wsaw_wake_phi_loss_history[repeat_idx], wsaw_anneal_factor_history[repeat_idx], wsaw_true_multiplier_history[repeat_idx], wsaw_true_offset_history[repeat_idx], wsaw_true_std_history[repeat_idx] = np.load('wsaw_prior_mean_history.npy'), np.load('wsaw_obs_std_history.npy'), np.load('wsaw_multiplier_history.npy'), np.load('wsaw_offset_history.npy'), np.load('wsaw_std_history.npy'), np.load('wsaw_wake_theta_loss_history.npy'), np.load('wsaw_wake_phi_loss_history.npy'), np.load('wsaw_anneal_factor_history.npy'), np.load('wsaw_true_multiplier_history.npy'), np.load('wsaw_true_offset_history.npy'), np.load('wsaw_true_std_history.npy')

    # Plotting
    fig, axs = plt.subplots(5, 1, sharex=True)

    fig.set_size_inches(3.25, 4)

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # axs[0].set_title('Generative network parameters')
    axs[0] = plot_with_error_bars(axs[0], iwae_prior_mean_history, color='black', linestyle=':', label='iwae-{}'.format(iwae_num_particles))
    axs[0] = plot_with_error_bars(axs[0], ws_prior_mean_history, color='black', linestyle='-.', label='ws-{}'.format(rws_num_particles))
    axs[0] = plot_with_error_bars(axs[0], ww_prior_mean_history, color='0.8', linestyle='-', label='ww-{}'.format(rws_num_particles))
    axs[0] = plot_with_error_bars(axs[0], wsw_prior_mean_history, color='0.4', linestyle='-', label='wsw-{}'.format(rws_num_particles))
    axs[0] = plot_with_error_bars(axs[0], waw_prior_mean_history, color='0.8', linestyle='--', label='waw-{}'.format(rws_num_particles))
    axs[0] = plot_with_error_bars(axs[0], wsaw_prior_mean_history, color='0.4', linestyle='--', label='wsaw-{}'.format(rws_num_particles))
    axs[0].axhline(true_prior_mean, color='black', label='true')
    axs[0].set_ylabel('$\mu_0$')

    axs[1] = plot_with_error_bars(axs[1], iwae_obs_std_history, color='black', linestyle=':', label='iwae-{}'.format(iwae_num_particles))
    axs[1] = plot_with_error_bars(axs[1], ws_obs_std_history, color='black', linestyle='-.', label='ws-{}'.format(rws_num_particles))
    axs[1] = plot_with_error_bars(axs[1], ww_obs_std_history, color='0.8', linestyle='-', label='ww-{}'.format(rws_num_particles))
    axs[1] = plot_with_error_bars(axs[1], wsw_obs_std_history, color='0.4', linestyle='-', label='wsw-{}'.format(rws_num_particles))
    axs[1] = plot_with_error_bars(axs[1], waw_obs_std_history, color='0.8', linestyle='--', label='waw-{}'.format(rws_num_particles))
    axs[1] = plot_with_error_bars(axs[1], wsaw_obs_std_history, color='0.4', linestyle='--', label='wsaw-{}'.format(rws_num_particles))
    axs[1].axhline(true_obs_std, color='black', label='true')
    axs[1].set_ylabel('$\sigma$')
    # axs[1].set_xlabel('Iteration')

    # axs[2].set_title('Inference network parameters')
    axs[2] = plot_with_error_bars(axs[2], iwae_multiplier_history, color='black', linestyle=':', label='iwae-{}'.format(iwae_num_particles))
    axs[2] = plot_with_error_bars(axs[2], ws_multiplier_history, color='black', linestyle='-.', label='ws-{}'.format(rws_num_particles))
    axs[2] = plot_with_error_bars(axs[2], ww_multiplier_history, color='0.8', linestyle='-', label='ww-{}'.format(rws_num_particles))
    axs[2] = plot_with_error_bars(axs[2], wsw_multiplier_history, color='0.4', linestyle='-', label='wsw-{}'.format(rws_num_particles))
    axs[2] = plot_with_error_bars(axs[2], waw_multiplier_history, color='0.8', linestyle='--', label='waw-{}'.format(rws_num_particles))
    axs[2] = plot_with_error_bars(axs[2], wsaw_multiplier_history, color='0.4', linestyle='--', label='wsaw-{}'.format(rws_num_particles))
    axs[2].axhline(true_multiplier, color='black', label='true')
    axs[2].set_ylabel('$a$')

    axs[3] = plot_with_error_bars(axs[3], iwae_offset_history, color='black', linestyle=':', label='iwae-{}'.format(iwae_num_particles))
    axs[3] = plot_with_error_bars(axs[3], ws_offset_history, color='black', linestyle='-.', label='ws-{}'.format(rws_num_particles))
    axs[3] = plot_with_error_bars(axs[3], ww_offset_history, color='0.8', linestyle='-', label='ww-{}'.format(rws_num_particles))
    axs[3] = plot_with_error_bars(axs[3], wsw_offset_history, color='0.4', linestyle='-', label='wsw-{}'.format(rws_num_particles))
    axs[3] = plot_with_error_bars(axs[3], waw_offset_history, color='0.8', linestyle='--', label='waw-{}'.format(rws_num_particles))
    axs[3] = plot_with_error_bars(axs[3], wsaw_offset_history, color='0.4', linestyle='--', label='wsaw-{}'.format(rws_num_particles))
    axs[3].axhline(true_offset, color='black', label='true')
    axs[3].set_ylabel('$b$')

    axs[4] = plot_with_error_bars(axs[4], iwae_std_history, color='black', linestyle=':', label='iwae-{}'.format(iwae_num_particles))
    axs[4] = plot_with_error_bars(axs[4], ws_std_history, color='black', linestyle='-.', label='ws-{}'.format(rws_num_particles))
    axs[4] = plot_with_error_bars(axs[4], ww_std_history, color='0.8', linestyle='-', label='ww-{}'.format(rws_num_particles))
    axs[4] = plot_with_error_bars(axs[4], wsw_std_history, color='0.4', linestyle='-', label='wsw-{}'.format(rws_num_particles))
    axs[4] = plot_with_error_bars(axs[4], waw_std_history, color='0.8', linestyle='--', label='waw-{}'.format(rws_num_particles))
    axs[4] = plot_with_error_bars(axs[4], wsaw_std_history, color='0.4', linestyle='--', label='wsaw-{}'.format(rws_num_particles))
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
