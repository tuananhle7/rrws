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
    num_clusters_probs = [0.5, 0.5]
    true_mean_1 = 0
    std_1 = 1
    mixture_probs = [0.5, 0.5]
    means_2 = [-5, 5]
    stds_2 = [1, 1]
    obs_std = 1

    init_mean_1 = 2

    num_iterations = 10
    learning_rate = 0.001

    # VAE
    vae_num_samples = 100
    vae_reinforce_elbo_history, vae_reinforce_mean_1_history = np.load('vae_reinforce_elbo_history.npy'), np.load('vae_reinforce_mean_1_history.npy')

    # IWAE
    iwae_num_samples = 100
    iwae_num_particles = 10

    ## Reinforce
    iwae_reinforce_elbo_history, iwae_reinforce_mean_1_history = np.load('iwae_reinforce_elbo_history.npy'), np.load('iwae_reinforce_mean_1_history.npy')

    ## VIMCO
    iwae_vimco_elbo_history, iwae_vimco_mean_1_history = np.load('iwae_vimco_elbo_history.npy'), np.load('iwae_vimco_mean_1_history.npy')

    # WS
    theta_learning_rate = learning_rate
    phi_learning_rate = learning_rate
    ws_num_samples = 100
    ws_sleep_num_samples = 100

    ws1_mean_1_history, ws1_wake_theta_loss_history, ws1_sleep_phi_loss_history = np.load('ws1_mean_1_history.npy'), np.load('ws1_wake_theta_loss_history.npy'), np.load('ws1_sleep_phi_loss_history.npy')

    # RWS
    rws_num_particles = 10
    rws_num_samples = 100
    rws_sleep_num_samples = 100

    ## WS
    ws_mean_1_history, ws_wake_theta_loss_history, ws_sleep_phi_loss_history = np.load('ws_mean_1_history.npy'), np.load('ws_wake_theta_loss_history.npy'), np.load('ws_sleep_phi_loss_history.npy')

    ## WW
    ww_mean_1_history, ww_wake_theta_loss_history, ww_wake_phi_loss_history = np.load('ww_mean_1_history.npy'), np.load('ww_wake_theta_loss_history.npy'), np.load('ww_wake_phi_loss_history.npy')

    ## WSW
    wsw_mean_1_history, wsw_wake_theta_loss_history, wsw_sleep_phi_loss_history, wsw_wake_phi_loss_history = np.load('wsw_mean_1_history.npy'), np.load('wsw_wake_theta_loss_history.npy'), np.load('wsw_sleep_phi_loss_history.npy'), np.load('wsw_wake_phi_loss_history.npy')

    ## WaW
    waw_mean_1_history, waw_wake_theta_loss_history, waw_wake_phi_loss_history, waw_anneal_factor_history = np.load('waw_mean_1_history.npy'), np.load('waw_wake_theta_loss_history.npy'), np.load('waw_wake_phi_loss_history.npy'), np.load('waw_anneal_factor_history.npy')

    ## WSaW
    wsaw_mean_1_history, wsaw_wake_theta_loss_history, wsaw_sleep_phi_loss_history, wsaw_wake_phi_loss_history, wsaw_anneal_factor_history = np.load('wsaw_mean_1_history.npy'), np.load('wsaw_wake_theta_loss_history.npy'), np.load('wsaw_sleep_phi_loss_history.npy'), np.load('wsaw_wake_phi_loss_history.npy'), np.load('wsaw_anneal_factor_history.npy')

    # Plot params
    fig, ax = plt.subplots(1, 1)

    fig.set_size_inches(3.25, 2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.plot(iwae_vimco_mean_1_history, color='0.8', linestyle=':', label='iwae-vimco-{}'.format(iwae_num_particles))
    ax.plot(iwae_reinforce_mean_1_history, color='0.2', linestyle=':', label='iwae-reinforce-{}'.format(iwae_num_particles))
    ax.plot(vae_reinforce_mean_1_history, color='0.2', linestyle=':', label='vae-reinforce')
    ax.plot(ws1_mean_1_history, color='black', linestyle='-.', label='ws-1')
    ax.plot(ws_mean_1_history, color='black', linestyle='-.', label='ws-{}'.format(rws_num_particles))
    ax.plot(ww_mean_1_history, color='0.8', linestyle='-', label='ww-{}'.format(rws_num_particles))
    ax.plot(wsw_mean_1_history, color='0.2', linestyle='-', label='wsw-{}'.format(rws_num_particles))
    ax.plot(waw_mean_1_history, color='0.8', linestyle='--', label='waw-{}'.format(rws_num_particles))
    ax.plot(wsaw_mean_1_history, color='0.2', linestyle='--', label='wsaw-{}'.format(rws_num_particles))

    ax.axhline(true_mean_1, color='black', label='true')
    ax.set_ylabel('$\mu_{1, 1}$')

    ax.set_xlabel('Iteration')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=4)
    fig.tight_layout()

    filename = '3gs_params.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))

    # Plot losses
    fig, ax = plt.subplots(1, 1)

    fig.set_size_inches(3.25, 2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.plot(-iwae_vimco_elbo_history, color='0.8', linestyle=':', label='iwae-vimco-{}'.format(iwae_num_particles))
    ax.plot(-iwae_reinforce_elbo_history, color='0.2', linestyle=':', label='iwae-reinforce-{}'.format(iwae_num_particles))
    ax.plot(-vae_reinforce_elbo_history, color='black', linestyle=':', label='vae-reinforce')
    ax.plot(ws1_wake_theta_loss_history, color='0.8', linestyle='-.', label='ws-1')
    ax.plot(ws_wake_theta_loss_history, color='black', linestyle='-.', label='ws-{}'.format(rws_num_particles))
    ax.plot(ww_wake_theta_loss_history, color='0.8', linestyle='-', label='ww-{}'.format(rws_num_particles))
    ax.plot(wsw_wake_theta_loss_history, color='0.2', linestyle='-', label='wsw-{}'.format(rws_num_particles))
    ax.plot(waw_wake_theta_loss_history, color='0.8', linestyle='--', label='waw-{}'.format(rws_num_particles))
    ax.plot(wsaw_wake_theta_loss_history, color='0.2', linestyle='--', label='wsaw-{}'.format(rws_num_particles))

    ax.set_ylabel('Loss')

    ax.set_xlabel('Iteration')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=4)
    fig.tight_layout()

    filename = '3gs_losses.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))

if __name__ == '__main__':
    main()
