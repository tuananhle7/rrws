from gmm_open_universe_vae import VAE
from gmm_open_universe_vae_relax import VAERelax
from gmm_open_universe_cdae import InferenceNetwork
from util import *

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

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


def get_bar_center(bar_center, bar_width, bar_gap, bar_index, num_bars):
    return bar_center + 0.5 * (bar_width + bar_gap) * (2 * bar_index + 1 - num_bars)


def get_running_mean_and_std(x, window):
    running_mean = np.zeros([len(x) - window + 1])
    running_std = np.zeros([len(x) - window + 1])

    for i in range(len(x) - window + 1):
        running_mean[i] = np.mean(x[i:(i + window)])
        running_std[i] = np.std(x[i:(i + window)])

    return running_mean, running_std


def main():
    # True parameters
    num_clusters_probs = [0.5, 0.5]
    mean_1 = 0
    std_1 = 1
    mixture_probs = [0.5, 0.5]
    means_2 = [-5, 5]
    stds_2 = [1, 1]
    obs_std = 1

    # Load VAE
    filename = 'vae_loss_history.npy'
    vae_loss_history = np.load(filename)
    print('Loaded from {}'.format(filename))

    filename = 'vae_mean_1_history.npy'
    vae_mean_1_history = np.load(filename)
    print('Loaded from {}'.format(filename))

    filename = 'vae.pt'
    vae_state_dict = torch.load(filename)
    print('Loaded from {}'.format(filename))

    init_mean_1 = np.random.normal(2, 0.1)
    vae = VAE(num_clusters_probs, init_mean_1, std_1, mixture_probs, means_2, stds_2, obs_std)
    vae.load_state_dict(vae_state_dict)

    # Load VAE_Relax
    filename = 'vae_relax_loss_history.npy'
    vae_relax_loss_history = np.load(filename)
    print('Loaded from {}'.format(filename))

    filename = 'vae_relax_mean_1_history.npy'
    vae_relax_mean_1_history = np.load(filename)
    print('Loaded from {}'.format(filename))

    filename = 'vae_relax.pt'
    vae_relax_state_dict = torch.load(filename)
    print('Loaded from {}'.format(filename))

    init_mean_1 = np.random.normal(2, 0.1)
    vae_relax = VAERelax(num_clusters_probs, init_mean_1, std_1, mixture_probs, means_2, stds_2, obs_std)
    vae_relax.load_state_dict(vae_relax_state_dict)

    # Load CDAE
    filename = 'cdae_theta_loss_history.npy'
    cdae_theta_loss_history = np.load(filename).flatten()
    print('Loaded from {}'.format(filename))

    filename = 'cdae_phi_loss_history.npy'
    cdae_phi_loss_history = np.load(filename).flatten()
    print('Loaded from {}'.format(filename))

    filename = 'cdae_mean_1_history.npy'
    cdae_mean_1_history = np.load(filename)
    print('Loaded from {}'.format(filename))

    filename = 'cdae_generative_network.pt'
    cdae_generative_network_state_dict = torch.load(filename)
    print('Loaded from {}'.format(filename))

    filename = 'cdae_inference_network.pt'
    cdae_inference_network_state_dict = torch.load(filename)
    print('Loaded from {}'.format(filename))

    cdae_inference_network = InferenceNetwork(len(num_clusters_probs), len(mixture_probs))
    cdae_inference_network.load_state_dict(cdae_inference_network_state_dict)

    # Plot Losses
    fig, axs = plt.subplots(4, 1, sharex=True)
    fig.set_size_inches(3.25, 3)

    axs[0].plot(cdae_phi_loss_history, color='black', label='wake-sleep')
    axs[0].set_ylabel('wake-sleep\n$\phi$ loss')
    axs[1].plot(cdae_theta_loss_history, color='black', label='wake-sleep')
    axs[1].set_ylabel('wake-sleep\n$\\theta$ loss')
    axs[2].plot(vae_loss_history, color='gray', label='vae')
    axs[2].set_ylabel('vae loss')
    axs[3].plot(vae_relax_loss_history, color='lightgray', label='relax')
    axs[3].set_ylabel('relax loss')

    for ax in axs[1:]:
        ax.set_ylim(-3, 15)

    # axs[0].set_title('Loss')
    axs[-1].set_xlabel('Iteration')
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    filename = 'gmm_open_universe_loss.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))

    # Plot Smoothed losses
    window = 100
    num_std = 2
    alpha = 0.3
    vae_loss_history_running_mean, vae_loss_history_running_std = get_running_mean_and_std(vae_loss_history, window)
    vae_relax_loss_history_running_mean, vae_relax_loss_history_running_std = get_running_mean_and_std(vae_relax_loss_history, window)
    cdae_theta_loss_history_running_mean, cdae_theta_loss_history_running_std = get_running_mean_and_std(cdae_theta_loss_history, window)
    cdae_phi_loss_history_running_mean, cdae_phi_loss_history_running_std = get_running_mean_and_std(cdae_phi_loss_history, window)

    fig, axs = plt.subplots(2, 1)
    fig.set_size_inches(3.25, 3.25)

    iterations = np.arange(len(cdae_phi_loss_history_running_mean))
    axs[0].plot(iterations, cdae_phi_loss_history_running_mean, color='black', label='wake-sleep')
    axs[0].fill_between(
        iterations,
        cdae_phi_loss_history_running_mean - num_std * cdae_phi_loss_history_running_std,
        cdae_phi_loss_history_running_mean + num_std * cdae_phi_loss_history_running_std,
        alpha=alpha,
        color='black'
    )
    axs[0].set_ylabel('$\phi$ loss')
    axs[1].plot(iterations, cdae_theta_loss_history_running_mean, color='black', label='wake-sleep')
    axs[1].fill_between(
        iterations,
        cdae_theta_loss_history_running_mean - num_std * cdae_theta_loss_history_running_std,
        cdae_theta_loss_history_running_mean + num_std * cdae_theta_loss_history_running_std,
        alpha=alpha,
        color='black'
    )
    axs[1].plot(iterations, vae_loss_history_running_mean, color='gray', label='vae')
    axs[1].fill_between(
        iterations,
        vae_loss_history_running_mean - num_std * vae_loss_history_running_std,
        vae_loss_history_running_mean + num_std * vae_loss_history_running_std,
        alpha=alpha,
        color='gray'
    )
    axs[1].plot(iterations, vae_relax_loss_history_running_mean, color='lightgray', label='relax')
    axs[1].fill_between(
        iterations,
        vae_relax_loss_history_running_mean - num_std * vae_relax_loss_history_running_std,
        vae_relax_loss_history_running_mean + num_std * vae_relax_loss_history_running_std,
        alpha=alpha,
        color='lightgray'
    )
    axs[1].set_ylabel('$\\theta$ loss')

    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=3)

    # axs[0].set_title('Loss')
    axs[1].set_xlabel('Iteration')
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    filename = 'gmm_open_universe_smoothed_loss.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))

    # Plot Model Params
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(3.25, 1.25)
    ax.plot(cdae_mean_1_history, color='black', label='wake-sleep')
    ax.plot(vae_mean_1_history, color='gray', label='vae')
    ax.plot(vae_relax_mean_1_history, color='lightgray', label='relax')
    ax.axhline(mean_1, color='black', linestyle='dashed', label='true')
    ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('$\mu_{1, 1}$')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    filename = 'gmm_open_universe_model_param.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))


    # Plot losses and model params
    fig, axs = plt.subplots(5, 1, sharex=True)
    fig.set_size_inches(3.25, 3.5)

    axs[0].plot(cdae_mean_1_history, color='black', label='wake-sleep')
    axs[0].plot(vae_mean_1_history, color='gray', label='vae')
    axs[0].plot(vae_relax_mean_1_history, color='lightgray', label='relax')
    axs[0].axhline(mean_1, color='black', linestyle='dashed', label='true')
    axs[0].legend(ncol=2, frameon=False)
    axs[0].set_ylabel('$\mu_{1, 1}$')

    axs[1].plot(cdae_phi_loss_history, color='black', label='wake-sleep')
    axs[1].set_ylabel('wake-sleep\n$\phi$ loss')
    axs[2].plot(cdae_theta_loss_history, color='black', label='wake-sleep')
    axs[2].set_ylabel('wake-sleep\n$\\theta$ loss')
    axs[3].plot(vae_loss_history, color='gray', label='vae')
    axs[3].set_ylabel('vae loss')
    axs[4].plot(vae_relax_loss_history, color='lightgray', label='relax')
    axs[4].set_ylabel('relax loss')

    for ax in axs[2:]:
        ax.set_ylim(-3, 15)

    # axs[0].set_title('Loss')
    axs[-1].set_xlabel('Iteration')
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='dashed', axis='y', alpha=0.5)

    filename = 'gmm_open_universe_model_param_and_loss.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))

    # Plot losses and model params 2
    fig, axs = plt.subplots(3, 1, sharex=True)
    fig.set_size_inches(3.25, 2.5)

    axs[0].plot(cdae_mean_1_history, color='black', label='wake-sleep')
    axs[0].plot(vae_mean_1_history, color='gray', label='vae')
    axs[0].plot(vae_relax_mean_1_history, color='lightgray', label='relax')
    axs[0].axhline(mean_1, color='black', linestyle='dashed', label='true')
    axs[0].legend(ncol=2, frameon=False)
    axs[0].set_ylabel('$\mu_{1, 1}$')

    axs[1].plot(cdae_phi_loss_history, color='black', label='wake-sleep')
    axs[1].set_ylabel('w-s $\phi$ loss')
    axs[2].plot(cdae_theta_loss_history, color='black', label='wake-sleep')
    axs[2].plot(vae_loss_history, color='gray', label='vae')
    axs[2].plot(vae_relax_loss_history, color='lightgray', label='relax')
    axs[2].set_ylabel('-ELBO\n(w-s $\\theta$ loss)')
    # axs[2].legend(ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.4), loc='upper center')
    # axs[2].legend(ncol=3, frameon=False)

    axs[-1].set_xlabel('Iteration')
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    filename = 'gmm_open_universe_model_param_and_loss_2.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))


    # Plot Inference
    num_test_obs = 5
    test_obs_min = -7
    test_obs_max = 7
    test_obss = np.linspace(test_obs_min, test_obs_max, num=num_test_obs)

    num_prior_samples = 1000
    num_posterior_samples = 10000
    num_inference_network_samples = 10000
    num_points = 100
    min_point = min([-10, test_obs_min])
    max_point = max([10, test_obs_max])

    bar_width = 0.1
    bar_gap = 0.02
    num_barplots = 5

    x_points = np.linspace(min_point, max_point, num_points)
    z_points = np.arange(len(mixture_probs))
    k_points = np.arange(len(num_clusters_probs)) + 1

    fig, axs = plt.subplots(3, num_test_obs, sharey='row')
    fig.set_size_inches(6.75, 3)

    for axs_ in axs:
        for ax in axs_:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    for test_obs_idx, test_obs in enumerate(test_obss):
        k_prior_pdf, z_prior_pdf, x_prior_pdf = get_prior_pdf(x_points, num_prior_samples, num_clusters_probs, mean_1, std_1, mixture_probs, means_2, stds_2, obs_std)
        k_posterior_pdf, z_posterior_pdf, x_posterior_pdf = get_posterior_pdf(x_points, num_posterior_samples, test_obs, num_clusters_probs, mean_1, std_1, mixture_probs, means_2, stds_2, obs_std)
        k_cdae_pdf, z_cdae_pdf, x_cdae_pdf = cdae_inference_network.get_pdf(x_points, test_obs, num_inference_network_samples)
        k_vae_pdf, z_vae_pdf, x_vae_pdf = vae.get_pdf(x_points, test_obs, num_inference_network_samples)
        k_vae_relax_pdf, z_vae_relax_pdf, x_vae_relax_pdf = vae_relax.get_pdf(x_points, test_obs, num_inference_network_samples)

        i = 0
        axs[0][test_obs_idx].bar(get_bar_center(k_points, bar_width, bar_gap, i, num_barplots), k_prior_pdf, width=bar_width, color='lightgray', edgecolor='lightgray', fill=True, label='prior')

        i = 1
        axs[0][test_obs_idx].bar(get_bar_center(k_points, bar_width, bar_gap, i, num_barplots), k_posterior_pdf, width=bar_width, color='black', edgecolor='black', fill=True, label='posterior')

        i = 2
        axs[0][test_obs_idx].bar(get_bar_center(k_points, bar_width, bar_gap, i, num_barplots), k_cdae_pdf, width=bar_width, color='black', fill=False, linestyle='dashed', label='wake-sleep')

        i = 3
        axs[0][test_obs_idx].bar(get_bar_center(k_points, bar_width, bar_gap, i, num_barplots), k_vae_pdf, width=bar_width, color='black', fill=False, linestyle='dotted', label='vae')

        i = 4
        axs[0][test_obs_idx].bar(get_bar_center(k_points, bar_width, bar_gap, i, num_barplots), k_vae_relax_pdf, width=bar_width, color='black', fill=False, linestyle='dashdot', label='relax')


        axs[0][test_obs_idx].set_xticks(k_points)
        axs[0][test_obs_idx].set_ylim([0, 1])
        axs[0][test_obs_idx].set_yticks([0, 1])
        axs[0][0].set_ylabel('$k$')

        i = 0
        axs[1][test_obs_idx].bar(get_bar_center(z_points, bar_width, bar_gap, i, num_barplots), z_prior_pdf, width=bar_width, color='lightgray', edgecolor='lightgray', fill=True, label='prior')

        i = 1
        axs[1][test_obs_idx].bar(get_bar_center(z_points, bar_width, bar_gap, i, num_barplots), z_posterior_pdf, width=bar_width, color='black', edgecolor='black', fill=True, label='posterior')

        i = 2
        axs[1][test_obs_idx].bar(get_bar_center(z_points, bar_width, bar_gap, i, num_barplots), z_cdae_pdf, width=bar_width, color='black', fill=False, linestyle='dashed', label='wake-sleep')

        i = 3
        axs[1][test_obs_idx].bar(get_bar_center(z_points, bar_width, bar_gap, i, num_barplots), z_vae_pdf, width=bar_width, color='black', fill=False, linestyle='dotted', label='vae')

        i = 4
        axs[1][test_obs_idx].bar(get_bar_center(z_points, bar_width, bar_gap, i, num_barplots), z_vae_relax_pdf, width=bar_width, color='black', fill=False, linestyle='dashdot', label='relax')

        axs[1][test_obs_idx].set_xticks(z_points)
        axs[1][test_obs_idx].set_ylim([0, 1])
        axs[1][test_obs_idx].set_yticks([0, 1])
        axs[1][0].set_ylabel('$z$')

        axs[2][test_obs_idx].plot(x_points, x_prior_pdf, color='lightgray', label='prior')
        axs[2][test_obs_idx].plot(x_points, x_posterior_pdf, color='black', label='posterior')
        axs[2][test_obs_idx].plot(x_points, x_cdae_pdf, color='black', linestyle='dashed', label='wake-sleep')
        axs[2][test_obs_idx].plot(x_points, x_vae_pdf, color='black', linestyle='dotted', label='vae')
        axs[2][test_obs_idx].plot(x_points, x_vae_relax_pdf, color='black', linestyle='dashdot', label='relax')
        axs[2][test_obs_idx].scatter(x=test_obs, y=0, color='black', label='test obs', marker='x')

        axs[2][test_obs_idx].set_yticks([])
        axs[2][0].set_ylabel('$x$')


    axs[-1][test_obs_idx // 2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=7)
    axs[1][2].remove()
    fig.tight_layout()

    filename = 'gmm_open_universe_inference.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))


if __name__ == '__main__':
    main()
