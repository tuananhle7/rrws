import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import dgm

from gmm import *
from gmm_concrete import *
from util import *

SMALL_SIZE = 5.5
MEDIUM_SIZE = 9
BIGGER_SIZE = 11

plt.switch_backend('agg')
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('axes', linewidth=0.5)            # set the value globally
plt.rc('lines', linewidth=1)           # line thickness


def main():
    seeds = np.arange(1, 11, dtype=int)
    # uids = ['3319b6a9', '03ee5995', '179b8125', '871c4fce']
    # concrete_uids = ['eaf03c8b', '7da5406d', '2c000794', '5b1962a9']
    # reinforce_uids = ['9b28ea68', 'f6ebee25', 'e520c68c', '4c1be354']
    uids = ['3319b6a9', '871c4fce']
    concrete_uids = ['eaf03c8b', '5b1962a9']
    reinforce_uids = ['9b28ea68', '4c1be354']
    iwae_filenames = ['p_mixture_probs_norm_history', 'true_posterior_norm_history', 'q_grad_std_history']
    rws_filenames = ['p_mixture_probs_norm_history', 'true_posterior_norm_history', 'q_grad_std_history']
    concrete_names = ['prior_l2_history', 'true_posterior_l2_history', 'inference_network_grad_phi_std_history']

    # num_particles_list = [2, 5, 10, 20]
    num_particles_list = [2, 20]
    ww_probs = [1.0, 0.8]
    num_test_x = 3

    num_iterations = np.load('{}/num_iterations_{}.npy'.format(WORKING_DIR, uids[0]))
    saving_interval = np.load('{}/saving_interval_{}.npy'.format(WORKING_DIR, uids[0]))
    saving_iterations = np.arange(0, num_iterations, saving_interval)
    num_iterations_to_plot = 4
    iterations_to_plot = saving_iterations[np.floor(
        np.linspace(0, len(saving_iterations) - 1, num=num_iterations_to_plot)
    ).astype(int)]

    num_mixtures = int(np.load('{}/num_mixtures_{}.npy'.format(WORKING_DIR, uids[0])))

    true_p_mixture_probs = np.load('{}/true_p_mixture_probs_{}.npy'.format(WORKING_DIR, uids[0]))
    true_mean_multiplier = float(np.load('{}/true_mean_multiplier_{}.npy'.format(WORKING_DIR, uids[0])))
    true_log_stds = np.load('{}/true_log_stds_{}.npy'.format(WORKING_DIR, uids[0]))

    p_init_mixture_probs_pre_softmax = np.random.rand(num_mixtures)
    init_log_stds = np.random.rand(num_mixtures)
    init_mean_multiplier = float(true_mean_multiplier)
    softmax_multiplier = 0.5

    true_generative_network = GenerativeNetwork(np.log(true_p_mixture_probs) / softmax_multiplier, true_mean_multiplier, true_log_stds)
    test_xs = np.linspace(0, num_mixtures - 1, num=num_test_x) * true_mean_multiplier

    # Plotting
    nrows = num_iterations_to_plot
    ncols = len(num_particles_list) * (num_test_x + 1)
    fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    width = 5.5
    ax_width = width / ncols
    height = nrows * ax_width
    fig.set_size_inches(width, height)

    for iteration_idx, iteration in enumerate(iterations_to_plot):
        axs[iteration_idx, 0].set_ylabel('i = {}'.format(iteration))
        for num_particles_idx, num_particles in enumerate(num_particles_list):
            # Plot the generative network
            ax = axs[iteration_idx, num_particles_idx * (num_test_x + 1)]
            # ax.spines['right'].set_visible(False)
            # ax.spines['top'].set_visible(False)
            ax.set_xticks([0, 19])
            ax.set_xticklabels([0, 19])
            ax.set_yticks([0, 1])
            ax.set_yticklabels([0, 1])
            ax.tick_params(length=0)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(-0.5, 19.5)
            if iteration_idx == 0:
                ax.set_title('$p_\\theta(z)$')

            # ax.spines['bottom'].set_visible(False)
            # ax.spines['left'].set_visible(False)
            # ax.tick_params(bottom="off", left="off")

            ## True generative network
            ax.plot(np.arange(num_mixtures), true_generative_network.get_z_params().data.numpy(), label='true', linewidth=matplotlib.rcParams['lines.linewidth'] * 1.5, color='black')

            ## Learned generative network
            ### WS
            filename='{}/ws_{}_{}_{}.pt'.format(WORKING_DIR, iteration, seeds[0], uids[num_particles_idx])
            ws_state_dict = torch.load(filename)
            ws = RWS(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds)
            ws.load_state_dict(ws_state_dict)
            ax.plot(np.arange(num_mixtures), ws.generative_network.get_z_params().data.numpy(), label='ws', alpha=0.8)

            ### Concrete
            filename = '{}/concrete_{}_{}_{}.pt'.format(WORKING_DIR, iteration, seeds[0], concrete_uids[num_particles_idx])
            concrete_state_dict = torch.load(filename)
            concrete = dgm.autoencoder.AutoEncoder(
                Prior(
                    init_mixture_probs_pre_softmax=p_init_mixture_probs_pre_softmax,
                    softmax_multiplier=softmax_multiplier
                ),
                None,
                Likelihood(0, np.exp(init_log_stds)),
                    InferenceNetwork(
                    num_mixtures, temperature=0.5
                )
            )
            concrete.load_state_dict(concrete_state_dict)
            ax.plot(np.arange(num_mixtures), concrete.initial.probs().data.numpy(), label='concrete', alpha=0.8)

            ### VIMCO, Reinforce
            filename = '{}/reinforce_{}_{}_{}.pt'.format(WORKING_DIR, iteration, seeds[0], reinforce_uids[num_particles_idx])
            iwae_reinforce_state_dict = torch.load(filename)
            iwae_reinforce = IWAE(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds)
            iwae_reinforce.load_state_dict(iwae_reinforce_state_dict)
            ax.plot(np.arange(num_mixtures), iwae_reinforce.generative_network.get_z_params().data.numpy(), label='reinforce', alpha=0.8)

            filename = '{}/vimco_{}_{}_{}.pt'.format(WORKING_DIR, iteration, seeds[0], uids[num_particles_idx])
            iwae_vimco_state_dict = torch.load(filename)
            iwae_vimco = IWAE(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds)
            iwae_vimco.load_state_dict(iwae_vimco_state_dict)
            ax.plot(np.arange(num_mixtures), iwae_vimco.generative_network.get_z_params().data.numpy(), label='vimco', alpha=0.8)

            ### WW
            wws = []
            for ww_prob_idx, ww_prob in enumerate(ww_probs):
                filename = '{}/ww{}_{}_{}_{}.pt'.format(WORKING_DIR, str(ww_prob).replace('.', '-'), iteration, seeds[0], uids[num_particles_idx])
                ww_state_dict = torch.load(filename)
                wws.append(RWS(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds))
                wws[-1].load_state_dict(ww_state_dict)
                ax.plot(np.arange(num_mixtures), wws[-1].generative_network.get_z_params().data.numpy(), label='ww {}'.format(ww_prob), alpha=0.8)

            # Plot the inference network
            for test_x_idx, test_x in enumerate(test_xs):
                ax = axs[iteration_idx, num_particles_idx * (num_test_x + 1) + test_x_idx + 1]
                # ax.spines['right'].set_visible(False)
                # ax.spines['top'].set_visible(False)
                ax.set_xticks([0, 19])
                ax.set_xticklabels([0, 19])
                ax.set_yticks([0, 1])
                ax.set_yticklabels([0, 1])
                ax.tick_params(length=0)
                ax.set_ylim(-0.05, 1.05)
                ax.set_xlim(-0.5, 19.5)
                # ax.spines['bottom'].set_visible(False)
                # ax.spines['left'].set_visible(False)
                # ax.tick_params(bottom="off", left="off")
                test_x_var = Variable(torch.Tensor([test_x]))
                if iteration_idx == 0:
                    ax.set_title('$q_\phi(z | x = {0:.0f})$'.format(test_x_var.data[0]))

                ## True posterior
                ax.plot(np.arange(num_mixtures), true_generative_network.posterior(test_x_var).data.numpy()[0], label='true', linewidth=matplotlib.rcParams['lines.linewidth'] * 1.5, color='black')

                ## Learned approximate posteriors
                ### WS
                ax.plot(np.arange(num_mixtures), ws.inference_network.get_z_params(test_x_var).data.numpy()[0], alpha=0.8)

                ### Concrete
                ax.plot(np.arange(num_mixtures), concrete.proposal.probs(test_x_var).data.numpy()[0], label='concrete', alpha=0.8)

                ### VIMCO, Reinforce
                ax.plot(np.arange(num_mixtures), iwae_reinforce.inference_network.get_z_params(test_x_var).data.numpy()[0], label='reinforce', alpha=0.8)
                ax.plot(np.arange(num_mixtures), iwae_vimco.inference_network.get_z_params(test_x_var).data.numpy()[0], label='vimco', alpha=0.8)

                ### WW
                for ww_prob_idx, (ww, ww_prob) in enumerate(zip(wws, ww_probs)):
                    ax.plot(np.arange(num_mixtures), ww.inference_network.get_z_params(test_x_var).data.numpy()[0], label='ww {}'.format(ww_prob), alpha=0.8)

    for num_particles_idx, num_particles in enumerate(num_particles_list):
        ax = axs[0, num_particles_idx * (num_test_x + 1) + (num_test_x + 1) // 2]
        ax.text(0, 1.3, '$K = {}$'.format(num_particles), verticalalignment='bottom', horizontalalignment='center', transform=ax.transAxes)

    axs[-1, ncols // 2].legend(bbox_to_anchor=(0, -0.25), loc='upper center', ncol=10)
    fig.tight_layout(pad=0)
    filename = 'results/plot_paper_2.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))


if __name__ == '__main__':
    main()
