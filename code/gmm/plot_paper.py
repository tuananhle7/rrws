import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from util import *

SMALL_SIZE = 5.5
MEDIUM_SIZE = 9
BIGGER_SIZE = 11

plt.switch_backend('agg')
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('axes', linewidth=0.5)            # set the value globally
plt.rc('xtick.major', width=0.5)            # set the value globally
plt.rc('ytick.major', width=0.5)            # set the value globally
plt.rc('lines', linewidth=1)           # line thickness
plt.rc('ytick.major', size=2)            # set the value globally
plt.rc('xtick.major', size=2)            # set the value globally


def main():
    seeds = np.arange(1, 11, dtype=int)
    uids = ['3319b6a9', '03ee5995', '179b8125', '871c4fce']
    concrete_uids = ['eaf03c8b', '7da5406d', '2c000794', '5b1962a9']
    reinforce_uids = ['9b28ea68', 'f6ebee25', 'e520c68c', '4c1be354']
    num_particles = [2, 5, 10, 20]
    ww_probs = [1.0, 0.8]

    num_iterations = np.load('{}/num_iterations_{}.npy'.format(WORKING_DIR, uids[0]))
    logging_interval = np.load('{}/logging_interval_{}.npy'.format(WORKING_DIR, uids[0]))
    logging_iterations = np.arange(0, num_iterations, logging_interval)

    # Plotting
    fig, axs = plt.subplots(3, len(uids), sharex=True, sharey='row')
    fig.set_size_inches(5.5, 3.5)

    for ax_idx in range(len(uids)):
        uid = uids[ax_idx]
        concrete_uid = concrete_uids[ax_idx]
        reinforce_uid = reinforce_uids[ax_idx]

        iwae_filenames = ['p_mixture_probs_norm_history', 'true_posterior_norm_history', 'q_grad_std_history']
        rws_filenames = ['p_mixture_probs_norm_history', 'true_posterior_norm_history', 'q_grad_std_history']
        concrete_names = ['prior_l2_history', 'true_posterior_l2_history', 'inference_network_grad_phi_std_history']

        # WS
        ws = read_files('ws', rws_filenames , seeds, uid)
        kwargs = {'linestyle': '-', 'label': 'ws'}

        for idx, filename in enumerate(iwae_filenames):
            plot_with_error_bars(logging_iterations, ws[filename], axs[idx, ax_idx], **kwargs)

        # Concrete
        concrete = read_files('concrete', concrete_names, seeds, concrete_uid)
        kwargs = {'linestyle': '-', 'label': 'concrete'}

        for idx, name in enumerate(concrete_names):
            plot_with_error_bars(logging_iterations, concrete[name], axs[idx, ax_idx], **kwargs)

        ## IWAE
        # Reinforce
        iwae_reinforce = read_files('iwae_reinforce', iwae_filenames, seeds, reinforce_uid)
        kwargs = {'linestyle': '-', 'label': 'reinforce'}

        for idx, filename in enumerate(iwae_filenames):
            plot_with_error_bars(logging_iterations, iwae_reinforce[filename], axs[idx, ax_idx], **kwargs)

        # VIMCO
        iwae_vimco = read_files('iwae_vimco', iwae_filenames, seeds, uid)
        kwargs = {'linestyle': '-', 'label': 'vimco'}

        for idx, filename in enumerate(iwae_filenames):
            plot_with_error_bars(logging_iterations, iwae_vimco[filename], axs[idx, ax_idx], **kwargs)


        # WW
        for q_mixture_prob in ww_probs:
            ww = read_files('ww_{}'.format(str(q_mixture_prob).replace('.', '-')), rws_filenames, seeds, uid)
            kwargs = {'linestyle': '-', 'label': 'ww {}'.format(q_mixture_prob)}

            for idx, filename in enumerate(iwae_filenames):
                plot_with_error_bars(logging_iterations, ww[filename], axs[idx, ax_idx], **kwargs)



    axs[0, 0].set_yscale('log')
    axs[0, 0].set_ylabel('$|| p_{\\theta}(z) - p_{\\theta_{true}}(z) ||$')

    axs[1, 0].set_yscale('log')
    axs[1, 0].set_ylabel('Avg. test\n$|| q_\phi(z | x) - p_{\\theta_{true}}(z | x) ||$')

    axs[2, 0].set_yscale('log')
    axs[2, 0].set_ylabel('Std. of $\phi$ \n gradient est.')

    axs[-1, 1].legend(ncol=6, loc='upper center', bbox_to_anchor=(1, -0.2))
    for i, ax in enumerate(axs[0]):
        ax.set_title('$K = {}$'.format(num_particles[i]))
    for i, ax in enumerate(axs[-1]):
        ax.set_xlabel('Iteration')
        ax.xaxis.set_label_coords(0.5, -0.1)

    for axss in axs:
        for ax in axss:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.minorticks_off()
            ax.set_xticks([0, 100000])

    fig.tight_layout(pad=0)
    filename = 'results/plot_paper.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))


def read_files(algorithm, filenames, seeds, uid):
    num_seeds = len(seeds)
    temp = np.load('{}/{}_{}_{}_{}.npy'.format(WORKING_DIR, algorithm, filenames[0], seeds[0], uid))
    num_data = len(temp)
    result = {}
    for filename in filenames:
        result[filename] = np.zeros([num_seeds, num_data])
        for seed_idx, seed in enumerate(seeds):
            result[filename][seed_idx] = np.load('{}/{}_{}_{}_{}.npy'.format(
                WORKING_DIR, algorithm, filename, seed, uid
            ))
    return result


def plot_with_error_bars(x, ys, ax, *args, **kwargs):
    median = np.median(ys, axis=0)
    first_quartile = np.percentile(ys, 25, axis=0)
    third_quartile = np.percentile(ys, 75, axis=0)
    line = ax.plot(x, median, *args, **kwargs)
    ax.fill_between(x, y1=first_quartile, y2=third_quartile, alpha=0.2, color=line[0].get_color())
    return ax


if __name__ == '__main__':
    main()
