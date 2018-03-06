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
    num_particles_list = [10 * int(i) for i in np.arange(1, 11)]
    iwae_reinforce_theta_grad, iwae_reinforce_phi_grad, iwae_vimco_theta_grad, iwae_vimco_phi_grad, wake_theta_grad, wake_phi_grad, sleep_phi_grad = np.load('iwae_reinforce_theta_grad.npy'), np.load('iwae_reinforce_phi_grad.npy'), np.load('iwae_vimco_theta_grad.npy'), np.load('iwae_vimco_phi_grad.npy'), np.load('wake_theta_grad.npy'), np.load('wake_phi_grad.npy'), np.load('sleep_phi_grad.npy')

    # Plotting
    iwae_reinforce_theta_grad_std = np.mean(np.std(iwae_reinforce_theta_grad, axis=1), axis=1)
    iwae_reinforce_phi_grad_std = np.mean(np.std(iwae_reinforce_phi_grad, axis=1), axis=1)

    iwae_vimco_theta_grad_std = np.mean(np.std(iwae_vimco_theta_grad, axis=1), axis=1)
    iwae_vimco_phi_grad_std = np.mean(np.std(iwae_vimco_phi_grad, axis=1), axis=1)

    wake_theta_grad_std = np.mean(np.std(wake_theta_grad, axis=1), axis=1)
    wake_phi_grad_std = np.mean(np.std(wake_phi_grad, axis=1), axis=1)
    sleep_phi_grad_std = np.mean(np.std(sleep_phi_grad, axis=1), axis=1)

    fig, axs = plt.subplots(2, 1, sharex=True)
    fig.set_size_inches(3.25, 3)

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axs[-1].set_xlabel('number of particles')

    axs[0].plot(num_particles_list, iwae_reinforce_theta_grad_std, linestyle='-', color='black', label='reinforce')
    axs[0].plot(num_particles_list, iwae_vimco_theta_grad_std, linestyle='--', color='black', label='vimco')
    axs[0].plot(num_particles_list, wake_theta_grad_std, linestyle='-.',color='black', label='wake-theta')
    axs[0].set_ylabel('average std. of $\\theta$\n gradient estimator')
    axs[0].legend()

    axs[1].plot(num_particles_list, iwae_reinforce_phi_grad_std, linestyle='-', color='black', label='reinforce')
    axs[1].plot(num_particles_list, iwae_vimco_phi_grad_std, linestyle='--', color='black', label='vimco')
    axs[1].plot(num_particles_list, wake_phi_grad_std, linestyle='-.', color='black', label='wake-phi')
    axs[1].plot(num_particles_list, sleep_phi_grad_std, linestyle=':', color='black', label='sleep-phi')

    axs[1].set_yscale('log')
    axs[1].set_ylim(1)
    axs[1].legend(ncol=2, loc='center left', bbox_to_anchor=(0, 0.35))
    axs[1].set_ylabel('average std. of $\phi$\n gradient estimator')

    fig.tight_layout()
    filename = 'gmm_gradient.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))


if __name__ == '__main__':
    main()
