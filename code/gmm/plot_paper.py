import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from util import *

SMALL_SIZE = 6
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
plt.rc('lines', linewidth=0.5)           # line thickness


def main():
    seeds = np.arange(1, 11, dtype=int)
    uids = ['3319b6a9', '03ee5995', '179b8125', '871c4fce']
    num_particles = [2, 5, 10, 20]
    ww_probs = [1.0, 0.8]

    num_iterations = np.load('{}/num_iterations_{}.npy'.format(WORKING_DIR, uids[0]))
    logging_interval = np.load('{}/logging_interval_{}.npy'.format(WORKING_DIR, uids[0]))
    logging_iterations = np.arange(0, num_iterations, logging_interval)

    # Plotting
    fig, axs = plt.subplots(3, len(uids), sharex=True, sharey='row')
    fig.set_size_inches(5.5, 3.5)

    for axss in axs:
        for ax in axss:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    for ax_idx in range(len(uids)):
        uid = uids[ax_idx]

        ## IWAE
        iwae_filenames = ['p_mixture_probs_norm_history', 'true_posterior_norm_history', 'q_grad_std_history']

        # VIMCO
        iwae_vimco = read_files('iwae_vimco', iwae_filenames, seeds, uid)
        kwargs = {'linestyle': '-', 'label': 'vimco'}

        for idx, filename in enumerate(iwae_filenames):
            plot_with_error_bars(logging_iterations, iwae_vimco[filename], axs[idx, ax_idx], **kwargs)

        rws_filenames = ['p_mixture_probs_norm_history', 'true_posterior_norm_history', 'q_grad_std_history']

        # WS
        ws = read_files('ws', rws_filenames , seeds, uid)
        kwargs = {'linestyle': '-', 'label': 'ws'}

        for idx, filename in enumerate(iwae_filenames):
            plot_with_error_bars(logging_iterations, ws[filename], axs[idx, ax_idx], **kwargs)

        # WW
        for q_mixture_prob in ww_probs:
            ww = read_files('ww_{}'.format(str(q_mixture_prob).replace('.', '-')), rws_filenames, seeds, uid)
            kwargs = {'linestyle': '-', 'label': 'ww {}'.format(q_mixture_prob)}

            for idx, filename in enumerate(iwae_filenames):
                plot_with_error_bars(logging_iterations, ww[filename], axs[idx, ax_idx], **kwargs)


    axs[0, 0].set_yscale('log')
    axs[0, 0].set_ylabel('$|| p_{\\theta}(z) - p_{\\theta^*}(z) ||$')

    axs[1, 0].set_yscale('log')
    axs[1, 0].set_ylabel('Avg. test\n$|| q_\phi(z | x) - p_{\\theta^*}(z | x) ||$')

    axs[2, 0].set_yscale('log')
    axs[2, 0].set_ylabel('Std. of $\phi$ \n gradient est.')

    axs[-1, 1].legend(ncol=4, loc='upper center', bbox_to_anchor=(1, -0.33))
    for i, ax in enumerate(axs[-1]):
        ax.set_xlabel('$K = {}$'.format(num_particles[i]))

    fig.tight_layout()
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
