import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

from util import *

SMALL_SIZE = 7
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


def main(args):
    true_log_evidence = np.load('{}/true_log_evidence_{}.npy'.format(WORKING_DIR, args.uid))
    num_mixtures = np.load('{}/num_mixtures_{}.npy'.format(WORKING_DIR, args.uid))
    num_iterations = np.load('{}/num_iterations_{}.npy'.format(WORKING_DIR, args.uid))
    logging_interval = np.load('{}/logging_interval_{}.npy'.format(WORKING_DIR, args.uid))
    logging_iterations = np.arange(0, num_iterations, logging_interval)

    # Plotting
    fig, axs = plt.subplots(7, 1, sharex=True)
    fig.set_size_inches(3.25, 8)

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    ## IWAE
    iwae_filenames = ['log_evidence_history', 'p_mixture_probs_norm_history', 'posterior_norm_history', 'true_posterior_norm_history', 'p_grad_std_history', 'q_grad_std_history', 'q_grad_mean_history']
    if args.all or args.reinforce:
        iwae_reinforce = read_files('iwae_reinforce', iwae_filenames, args.seeds, args.uid)
        kwargs = {'linestyle': '-', 'label': 'reinforce'}

        for idx, filename in enumerate(iwae_filenames):
            if filename == 'log_evidence_history':
                data = np.abs(true_log_evidence - iwae_reinforce[filename])
            else:
                data = iwae_reinforce[filename]
            plot_with_error_bars(logging_iterations, data, axs[idx], **kwargs)

    if args.all or args.vimco:
        iwae_vimco = read_files('iwae_vimco', iwae_filenames, args.seeds, args.uid)
        kwargs = {'linestyle': '-', 'label': 'vimco'}

        for idx, filename in enumerate(iwae_filenames):
            if filename == 'log_evidence_history':
                data = np.abs(true_log_evidence - iwae_vimco[filename])
            else:
                data = iwae_vimco[filename]
            plot_with_error_bars(logging_iterations, data, axs[idx], **kwargs)


    ## RWS
    rws_filenames = ['log_evidence_history', 'p_mixture_probs_norm_history', 'posterior_norm_history', 'true_posterior_norm_history', 'p_grad_std_history', 'q_grad_std_history', 'q_grad_mean_history']

    if args.all or args.ws:
        ws = read_files('ws', rws_filenames , args.seeds, args.uid)
        kwargs = {'linestyle': '-', 'label': 'ws'}

        for idx, filename in enumerate(iwae_filenames):
            if filename == 'log_evidence_history':
                data = np.abs(true_log_evidence - ws[filename])
            else:
                data = ws[filename]
            plot_with_error_bars(logging_iterations, data, axs[idx], **kwargs)

    if args.all or args.ww:
        for q_mixture_prob in args.ww_probs:
            ww = read_files('ww_{}'.format(str(q_mixture_prob).replace('.', '-')), rws_filenames, args.seeds, args.uid)
            kwargs = {'linestyle': '-', 'label': 'ww {}'.format(q_mixture_prob)}

            for idx, filename in enumerate(iwae_filenames):
                if filename == 'log_evidence_history':
                    data = np.abs(true_log_evidence - ww[filename])
                else:
                    data = ww[filename]
                plot_with_error_bars(logging_iterations, data, axs[idx], **kwargs)

    if args.all or args.wsw:
        wsw = read_files('wsw', rws_filenames , args.seeds, args.uid)
        kwargs = {'linestyle': '-', 'label': 'wsw'}

        for idx, filename in enumerate(iwae_filenames):
            if filename == 'log_evidence_history':
                data = np.abs(true_log_evidence - wsw[filename])
            else:
                data = wsw[filename]
            plot_with_error_bars(logging_iterations, data, axs[idx], **kwargs)

    if args.all or args.wswa:
        wswa = read_files('wswa', rws_filenames , args.seeds, args.uid)
        kwargs = {'linestyle': '-', 'label': 'wswa'}

        for idx, filename in enumerate(iwae_filenames):
            if filename == 'log_evidence_history':
                data = np.abs(true_log_evidence - wswa[filename])
            else:
                data = wswa[filename]
            plot_with_error_bars(logging_iterations, data, axs[idx], **kwargs)

    # axs[0].axhline(true_log_evidence, linestyle='-', color='black', label='true')
    # axs[0].set_ylabel('Avg. test\nlog evidence')
    axs[0].set_yscale('log')
    axs[0].set_ylabel('Abs. diff. of current\nand true avg. test\nlog evidence')
    # axs[0].set_ylim(-7, -5)

    axs[1].set_yscale('log')
    # axs[1].set_ylim(0)
    axs[1].set_ylabel('Avg. L2 of current\nto true\nmixture probs.')

    axs[2].set_yscale('log')
    # axs[2].set_ylim(0)
    axs[2].set_ylabel('Avg. test L2 of\nq to current p')

    axs[3].set_yscale('log')
    # axs[3].set_ylim(0)
    axs[3].set_ylabel('Avg. test L2 of\nq to true p')

    axs[4].set_ylim(0)
    axs[4].set_ylabel('Avg. std. of $\\theta$ \n gradient est.')

    axs[5].set_yscale('log')
    # axs[5].set_ylim(0)
    axs[5].set_ylabel('Avg. std. of $\phi$ \n gradient est.')

    axs[6].set_ylabel('Avg. mean of $\phi$ \n gradient est.')

    axs[-1].legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.4))
    axs[-1].set_xlabel('Iteration')

    fig.tight_layout()
    filename = '{}/plot_{}.pdf'.format(OUTPUT_DIR, args.uid)
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
    import argparse

    parser = argparse.ArgumentParser(description='GMM open universe')
    parser.add_argument('--uid', type=str, default='', metavar='U',
                        help='run UID')
    parser.add_argument('--seeds', nargs='*', type=int, default=[1])
    parser.add_argument('--all', action='store_true', default=False)
    parser.add_argument('--reinforce', action='store_true', default=False)
    parser.add_argument('--vimco', action='store_true', default=False)
    parser.add_argument('--ws', action='store_true', default=False)
    parser.add_argument('--ww', action='store_true', default=False)
    parser.add_argument('--ww-probs', nargs='*', type=float, default=[1.0])
    parser.add_argument('--wsw', action='store_true', default=False)
    parser.add_argument('--wswa', action='store_true', default=False)
    args = parser.parse_args()

    main(args)
