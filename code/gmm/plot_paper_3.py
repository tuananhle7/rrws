import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import dgm
import gmm_relax

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
plt.rc('lines', linewidth=0.7)           # line thickness
plt.rc('xtick.major', width=0.5)            # set the value globally
plt.rc('ytick.major', width=0.5)            # set the value globally
plt.rc('ytick.major', size=2)            # set the value globally
plt.rc('xtick.major', size=0)            # set the value globally
plt.rc('text', usetex=True)
plt.rc('text.latex',
       preamble=[r'\usepackage{amsmath}',
                 r'\usepackage[cm]{sfmath}'])
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['cm']})
plt.rc('axes', titlepad=3)


def plot_hinton(ax, data, top, bottom, left, right, pad=0, **kwargs):
    cell_width = (right - left) / len(data)
    cell_height = top - bottom
    center_y = (top + bottom) / 2
    for i, datum in enumerate(data):
        if datum > 1 / 1000:
            center_x = cell_width / 2 + i * cell_width
            width = min(cell_width, cell_height) * datum**0.5 - pad
            height = width
            x = center_x - width / 2
            y = center_y - height / 2
            ax.add_artist(matplotlib.patches.Rectangle((x, y), width, height, **kwargs))
    return ax

def main():
    bar_colors = ['black', 'C0', 'C3', 'C4', 'C5', 'C1', 'C6', 'C7']
    seeds = np.arange(1, 11, dtype=int)
    uids = ['3319b6a9', '871c4fce']
    concrete_uids = ['eaf03c8b', '5b1962a9']
    reinforce_uids = ['9b28ea68', '4c1be354']
    relax_uids = ['e24618b3', '3b3475dc']
    iwae_filenames = ['p_mixture_probs_norm_history', 'true_posterior_norm_history', 'q_grad_std_history']
    rws_filenames = ['p_mixture_probs_norm_history', 'true_posterior_norm_history', 'q_grad_std_history']
    concrete_names = ['prior_l2_history', 'true_posterior_l2_history', 'inference_network_grad_phi_std_history']
    relax_names = concrete_names

    num_particles_list = [2, 20]
    ww_probs = [1.0, 0.8]
    num_test_x = 3

    num_iterations = np.load('{}/num_iterations_{}.npy'.format(WORKING_DIR, uids[0]))
    saving_interval = np.load('{}/saving_interval_{}.npy'.format(WORKING_DIR, uids[0]))
    saving_iterations = np.arange(0, num_iterations, saving_interval)
    num_iterations_to_plot = 3
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
        axs[iteration_idx, 0].set_ylabel('Iter. {}'.format(iteration))
        for num_particles_idx, num_particles in enumerate(num_particles_list):
            # Plot the generative network
            ax = axs[iteration_idx, num_particles_idx * (num_test_x + 1)]
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_ylim(0, 8)
            ax.set_xlim(0, 20)
            if iteration_idx == 0:
                ax.set_title(r'$p_\theta(z)$')

            i = 0
            ## True generative network
            plot_hinton(ax, true_generative_network.get_z_params().data.numpy(), 8 - i, 8 - i - 1, 0, 20, label='true', color='black')
            i += 1

            ## Learned generative network
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
            plot_hinton(ax, concrete.initial.probs().data.numpy(), 8 - i, 8 - i - 1, 0, 20, label='Concrete', color='C0')
            i += 1

            ### Relax
            filename = '{}/relax_{}_{}_{}.pt'.format(WORKING_DIR, iteration, seeds[0], relax_uids[num_particles_idx])
            relax_state_dict = torch.load(filename)
            relax_prior = gmm_relax.Prior(p_init_mixture_probs_pre_softmax, softmax_multiplier)
            relax_inference_network = gmm_relax.InferenceNetwork(num_mixtures)
            relax_prior.load_state_dict(relax_state_dict['prior'])
            relax_inference_network.load_state_dict(relax_state_dict['inference_network'])
            plot_hinton(ax, relax_prior.probs().data.numpy(), 8 - i, 8 - i - 1, 0, 20, label='RELAX', color='C3')
            i += 1

            ### VIMCO, Reinforce
            filename = '{}/reinforce_{}_{}_{}.pt'.format(WORKING_DIR, iteration, seeds[0], reinforce_uids[num_particles_idx])
            iwae_reinforce_state_dict = torch.load(filename)
            iwae_reinforce = IWAE(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds)
            iwae_reinforce.load_state_dict(iwae_reinforce_state_dict)
            plot_hinton(ax, iwae_reinforce.generative_network.get_z_params().data.numpy(), 8 - i, 8 - i - 1, 0, 20, label='REINFORCE', color='C4')
            i += 1

            filename = '{}/vimco_{}_{}_{}.pt'.format(WORKING_DIR, iteration, seeds[0], uids[num_particles_idx])
            iwae_vimco_state_dict = torch.load(filename)
            iwae_vimco = IWAE(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds)
            iwae_vimco.load_state_dict(iwae_vimco_state_dict)
            plot_hinton(ax, iwae_vimco.generative_network.get_z_params().data.numpy(), 8 - i, 8 - i - 1, 0, 20, label='VIMCO', color='C5')
            i += 1

            ### WS
            filename='{}/ws_{}_{}_{}.pt'.format(WORKING_DIR, iteration, seeds[0], uids[num_particles_idx])
            ws_state_dict = torch.load(filename)
            ws = RWS(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds)
            ws.load_state_dict(ws_state_dict)
            plot_hinton(ax, ws.generative_network.get_z_params().data.numpy(), 8 - i, 8 - i - 1, 0, 20, label='WS', color='C1')
            i += 1

            ### WW
            wws = []
            for ww_prob_idx, ww_prob in enumerate(ww_probs):
                filename = '{}/ww{}_{}_{}_{}.pt'.format(WORKING_DIR, str(ww_prob).replace('.', '-'), iteration, seeds[0], uids[num_particles_idx])
                ww_state_dict = torch.load(filename)
                wws.append(RWS(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds))
                wws[-1].load_state_dict(ww_state_dict)
                plot_hinton(ax, wws[-1].generative_network.get_z_params().data.numpy(), 8 - i, 8 - i - 1, 0, 20, label='{}'.format('WW' if ww_prob == 1 else r'$\delta$-WW'), color='C{}'.format(ww_prob_idx + 6))
                i += 1

            # Plot the inference network
            for test_x_idx, test_x in enumerate(test_xs):
                ax = axs[iteration_idx, num_particles_idx * (num_test_x + 1) + test_x_idx + 1]
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.set_yticks([])
                ax.set_yticklabels([])
                ax.set_ylim(0, 8)
                ax.set_xlim(0, 20)

                test_x_var = Variable(torch.Tensor([test_x]))
                if iteration_idx == 0:
                    ax.set_title(r'$q_\phi(z | x = {0:.0f})$'.format(test_x_var.data[0]))

                i = 0

                ## True posterior
                plot_hinton(ax, true_generative_network.posterior(test_x_var).data.numpy()[0], 8 - i, 8 - i - 1, 0, 20, label='true', color='black')
                i += 1

                ## Learned approximate posteriors
                ### Concrete
                plot_hinton(ax, concrete.proposal.probs(test_x_var).data.numpy()[0], 8 - i, 8 - i - 1, 0, 20, label='Concrete', color='C0')
                i += 1

                ### Relax
                plot_hinton(ax, relax_inference_network.probs(test_x_var).data.numpy()[0], 8 - i, 8 - i - 1, 0, 20, label='RELAX', color='C3')
                i += 1

                ### VIMCO, Reinforce
                plot_hinton(ax, iwae_reinforce.inference_network.get_z_params(test_x_var).data.numpy()[0], 8 - i, 8 - i - 1, 0, 20, label='REINFORCE', color='C4')
                i += 1

                plot_hinton(ax, iwae_vimco.inference_network.get_z_params(test_x_var).data.numpy()[0], 8 - i, 8 - i - 1, 0, 20, label='VIMCO', color='C5')
                i += 1

                ### WS
                plot_hinton(ax, ws.inference_network.get_z_params(test_x_var).data.numpy()[0], 8 - i, 8 - i - 1, 0, 20, label='WS', color='C1')
                i += 1

                ### WW
                for ww_prob_idx, (ww, ww_prob) in enumerate(zip(wws, ww_probs)):
                    plot_hinton(ax, ww.inference_network.get_z_params(test_x_var).data.numpy()[0], 8 - i, 8 - i - 1, 0, 20, label='{}'.format('WW' if ww_prob == 1 else r'$\delta$-WW'), color='C{}'.format(ww_prob_idx + 6))
                    i += 1

    for num_particles_idx, num_particles in enumerate(num_particles_list):
        ax = axs[0, num_particles_idx * (num_test_x + 1) + (num_test_x + 1) // 2]
        ax.text(0, 1.25, '$K = {}$'.format(num_particles),  fontsize=SMALL_SIZE, verticalalignment='bottom', horizontalalignment='center', transform=ax.transAxes)

    axs[-1, ncols // 2].legend(bbox_to_anchor=(0, -0.1), loc='upper center', ncol=10, handles=[
        matplotlib.patches.Rectangle((0, 0), 1, 1, color='black', label='true'),
        matplotlib.patches.Rectangle((0, 0), 1, 1, color='C0', label='Concrete'),
        matplotlib.patches.Rectangle((0, 0), 1, 1, color='C3', label='RELAX'),
        matplotlib.patches.Rectangle((0, 0), 1, 1, color='C4', label='REINFORCE'),
        matplotlib.patches.Rectangle((0, 0), 1, 1, color='C5', label='VIMCO'),
        matplotlib.patches.Rectangle((0, 0), 1, 1, color='C1', label='WS'),
        matplotlib.patches.Rectangle((0, 0), 1, 1, color='C6', label='WW'),
        matplotlib.patches.Rectangle((0, 0), 1, 1, color='C7', label=r'$\delta$-WW')
    ])

    for ax in axs[-1]:
        ax.set_xlabel(r'$z$', labelpad=0.5)

    # axs[-1, ncols // 2].legend(bbox_to_anchor=(0, -0.25), loc='upper center', ncol=10)
    fig.tight_layout(pad=0)
    filename = 'results/plot_paper_3.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))


if __name__ == '__main__':
    main()
