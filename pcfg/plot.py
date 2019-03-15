import util
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib.lines as mlines
import torch


num_iterations = 2000
logging_interval = 10
eval_interval = 10
checkpoint_interval = 100
batch_size = 2
seed_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
train_mode_list = ['reinforce', 'vimco', 'ws', 'ww']
num_particles_list = [2, 5, 10, 20]
exp_levenshtein = True
pcfg_path = './pcfgs/astronomers_pcfg.json'


def delete_rows_with_nan(data):
    result = []
    for row in data:
        if not np.isnan(row).any():
            result.append(row)
    return np.array(result)


def plot_with_error_bars(ax, data, **plot_kwargs):
    data = delete_rows_with_nan(data)
    print(len(data))

    mid = np.nanmedian(data, axis=0)
    low = np.nanpercentile(data, 25, axis=0)
    high = np.nanpercentile(data, 75, axis=0)

    # mid = np.nanmean(data, axis=0)
    # std = np.nanstd(data, axis=0)
    # low = mid - std
    # high = mid + std
    num_not_nan = np.count_nonzero(~np.isnan(mid))

    if num_not_nan > 0:
        ax.plot(np.arange(num_not_nan), mid[:num_not_nan], **plot_kwargs)
        ax.fill_between(np.arange(num_not_nan),
                        low[:num_not_nan], high[:num_not_nan],
                        alpha=0.2, **plot_kwargs)


def load_errors():
    len_history = int(num_iterations / eval_interval)
    p_error = np.full((len(seed_list), len(train_mode_list),
                       len(num_particles_list), len_history),
                      np.nan, dtype=np.float)
    q_error_model = np.full((len(seed_list), len(train_mode_list),
                             len(num_particles_list), len_history),
                            np.nan, dtype=np.float)
    q_error_true = np.full((len(seed_list), len(train_mode_list),
                            len(num_particles_list), len_history),
                           np.nan, dtype=np.float)
    for seed_idx, seed in enumerate(seed_list):
        for train_mode_idx, train_mode in enumerate(train_mode_list):
            for num_particles_idx, num_particles in enumerate(
                num_particles_list
            ):
                model_folders = util.list_model_folders_args_match(
                    num_iterations=num_iterations,
                    logging_interval=logging_interval,
                    eval_interval=eval_interval,
                    checkpoint_interval=checkpoint_interval,
                    batch_size=batch_size,
                    seed=seed,
                    train_mode=train_mode,
                    num_particles=num_particles,
                    exp_levenshtein=exp_levenshtein)
                if len(model_folders) > 0:
                    model_folder = model_folders[np.argmax(
                        [os.stat(x).st_mtime for x in model_folders])]
                    stats = util.load_object(
                        util.get_stats_filename(model_folder))
                    p_error[
                        seed_idx, train_mode_idx,
                        num_particles_idx, :len(stats.p_error_history)
                    ] = stats.p_error_history
                    q_error_model[
                        seed_idx, train_mode_idx,
                        num_particles_idx, :len(stats.q_error_to_model_history)
                    ] = stats.q_error_to_model_history
                    q_error_true[
                        seed_idx, train_mode_idx,
                        num_particles_idx, :len(stats.q_error_to_true_history)
                    ] = stats.q_error_to_true_history
    return p_error, q_error_model, q_error_true


def plot_errors():
    p_error, q_error_model, q_error_true = load_errors()
    fig, axss = plt.subplots(nrows=2, ncols=len(num_particles_list),
                             figsize=(12, 4), sharex=True, sharey='row')

    # colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c']
    colors = ['C4', 'C5', 'C1', 'C6']
    linestyles = ['dashed', 'dashed', 'solid', 'solid']
    for train_mode_idx, train_mode in enumerate(train_mode_list):
        for num_particles_idx, num_particles in enumerate(num_particles_list):
            print('{} {}'.format(train_mode, num_particles))
            color = colors[train_mode_idx]
            linestyle = linestyles[train_mode_idx]
            plot_with_error_bars(
                axss[0, num_particles_idx],
                p_error[:, train_mode_idx, num_particles_idx, :], color=color,
                linestyle=linestyle)
            # plot_with_error_bars(
            #     axss[1, num_particles_idx],
            #     q_error_model[:, train_mode_idx, num_particles_idx, :],
            #     color=color)
            plot_with_error_bars(
                axss[-1, num_particles_idx],
                q_error_true[:, train_mode_idx, num_particles_idx, :],
                color=color, linestyle=linestyle)

    handles = []
    for train_mode_idx, train_mode in enumerate(train_mode_list):
        handles.append(mlines.Line2D([0], [1], color=colors[train_mode_idx],
                                     linestyle=linestyles[train_mode_idx]))
    axss[-1, 1].legend(
        handles, list(map(lambda x: x.upper(), train_mode_list)),
        bbox_to_anchor=(1.05, -0.2), loc='upper center',
        ncol=len(train_mode_list))

    axss[0, 0].set_ylabel(r'$p_{\theta}$ error', labelpad=-17)
    # axss[1, 0].set_ylabel(r'$q_\phi$ error to $p_{\theta}$', labelpad=-5)
    axss[-1, 0].set_ylabel(r'$q_\phi$ error to $p_{true}$', labelpad=-15)

    for ax in axss[0]:
        ax.set_ylim(0, 0.3)
    # for ax in axss[1]:
    #     ax.set_ylim(0, 20)
    for ax in axss[-1]:
        ax.set_ylim(0, 40)

    for axs in axss:
        for ax in axs:
            ax.set_xticks([0, num_iterations / eval_interval - 1])
            ax.set_xticklabels([1, num_iterations])
            ax.set_yticks([0, ax.get_yticks()[-1]])
            sns.despine(ax=ax, trim=True)

    for ax in axss[-1]:
        ax.set_xlabel('Iteration', labelpad=-10)

    for ax, num_particles in zip(axss[0], num_particles_list):
        ax.set_title(r'$K = {}$'.format(num_particles))

    fig.tight_layout()
    if not os.path.exists('./plots/'):
        os.makedirs('./plots/')
    filename = './plots/errors.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('saved to {}'.format(filename))


def plot_error_bar(ax, data, x, **kwargs):
    data = delete_rows_with_nan(data)

    mid = np.nanmedian(data, axis=0)
    low = np.nanpercentile(data, 25, axis=0)
    high = np.nanpercentile(data, 75, axis=0)

    num_not_nan = np.count_nonzero(~np.isnan(mid))

    if num_not_nan > 0:
        ax.errorbar([x], [mid[-1]],
                    yerr=[[mid[-1] - low[-1]], [high[-1] - mid[-1]]],
                    marker='.', markersize=5, **kwargs)


def plot_errors_end_points():
    p_error, q_error_model, q_error_true = load_errors()
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6, 2.5))
    # colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c']
    colors = ['C4', 'C5', 'C1', 'C6']
    for train_mode_idx, train_mode in enumerate(train_mode_list):
        for num_particles_idx, num_particles in enumerate(num_particles_list):
            color = colors[train_mode_idx]
            plot_error_bar(
                axs[0], p_error[:, train_mode_idx, num_particles_idx, :],
                num_particles_idx - 0.15 + 0.1 * train_mode_idx, color=color)
            # plot_error_bar(
            #     axs[1],
            #     q_error_model[:, train_mode_idx, num_particles_idx, :],
            #     num_particles_idx - 0.15 + 0.1 * train_mode_idx, color=color)
            plot_error_bar(
                axs[-1],
                q_error_true[:, train_mode_idx, num_particles_idx, :],
                num_particles_idx - 0.15 + 0.1 * train_mode_idx, color=color)

    axs[0].set_ylabel(r'$p_{\theta}$ error', labelpad=-15)
    # axs[1].set_ylabel(r'$q_\phi$ error to $p_{\theta}$', labelpad=0)
    axs[-1].set_ylabel(r'$q_\phi$ error to $p_{true}$', labelpad=-15)

    axs[0].set_ylim(0, 0.3)
    # axs[1].set_ylim(0, 30)
    axs[-1].set_ylim(0, 40)

    handles = []
    for train_mode_idx, train_mode in enumerate(train_mode_list):
        handles.append(mlines.Line2D([0], [1], color=colors[train_mode_idx]))
    axs[0].legend(handles, list(map(lambda x: x.upper(), train_mode_list)),
                  bbox_to_anchor=(1.07, -0.12), loc='upper center',
                  ncol=len(train_mode_list))

    axs[0].set_xlabel('K', labelpad=-8)
    axs[-1].set_xlabel('K', labelpad=-8)

    for ax in axs:
        ax.set_xticks(range(len(num_particles_list)))
        ax.set_xticklabels(num_particles_list)
        ax.set_yticks([0, ax.get_yticks()[-1]])
        sns.despine(ax=ax, trim=True)

    fig.tight_layout()
    if not os.path.exists('./plots/'):
        os.makedirs('./plots/')
    filename = './plots/errors_end_points.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('saved to {}'.format(filename))


def plot_both():
    p_error, q_error_model, q_error_true = load_errors()
    fig, axss = plt.subplots(nrows=2, ncols=len(num_particles_list) + 1,
                             figsize=(12, 4), sharex='col', sharey='row')

    # colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c']
    colors = ['C4', 'C5', 'C1', 'C6']
    for train_mode_idx, train_mode in enumerate(train_mode_list):
        for num_particles_idx, num_particles in enumerate(num_particles_list):
            print('{} {}'.format(train_mode, num_particles))
            color = colors[train_mode_idx]
            plot_with_error_bars(
                axss[0, num_particles_idx],
                p_error[:, train_mode_idx, num_particles_idx, :], color=color)
            plot_with_error_bars(
                axss[-1, num_particles_idx],
                q_error_true[:, train_mode_idx, num_particles_idx, :],
                color=color)
            plot_error_bar(
                axss[0, -1], p_error[:, train_mode_idx, num_particles_idx, :],
                num_particles_idx - 0.15 + 0.1 * train_mode_idx, color=color)
            plot_error_bar(
                axss[-1, -1],
                q_error_true[:, train_mode_idx, num_particles_idx, :],
                num_particles_idx - 0.15 + 0.1 * train_mode_idx, color=color)

    handles = []
    for train_mode_idx, train_mode in enumerate(train_mode_list):
        handles.append(mlines.Line2D([0], [1], color=colors[train_mode_idx]))
    axss[-1, 2].legend(
        handles, list(map(lambda x: x.upper(), train_mode_list)),
        bbox_to_anchor=(0.5, -0.2), loc='upper center',
        ncol=len(train_mode_list))

    axss[0, 0].set_ylabel(r'$p_{\theta}$ error', labelpad=-17)
    axss[-1, 0].set_ylabel(r'$q_\phi$ error to $p_{true}$', labelpad=-15)

    for ax in axss[0]:
        ax.set_ylim(0, 0.3)
    for ax in axss[-1]:
        ax.set_ylim(0, 40)

    for axs in axss:
        for ax in axs[:-1]:
            ax.set_xticks([0, num_iterations / eval_interval - 1])
            ax.set_xticklabels([1, num_iterations])
            ax.set_yticks([0, ax.get_yticks()[-1]])
            sns.despine(ax=ax, trim=True)

    for ax in axss[-1, :-1]:
        ax.set_xlabel('Iteration', labelpad=-10)

    for ax, num_particles in zip(axss[0], num_particles_list):
        ax.set_title(r'$K = {}$'.format(num_particles))

    for ax in axss[:, -1]:
        ax.set_xticks(range(len(num_particles_list)))
        ax.set_xticklabels(num_particles_list)
        ax.set_yticks([0, ax.get_yticks()[-1]])
        sns.despine(ax=ax, trim=True)
    axss[-1, -1].set_xlabel('K', labelpad=-10)
    axss[0, -1].set_title('Last iteration')

    fig.tight_layout()
    if not os.path.exists('./plots/'):
        os.makedirs('./plots/')
    filename = './plots/both_errors.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('saved to {}'.format(filename))


def plot_production_probs():
    seed = 1
    num_particles = 20
    _, _, true_generative_model = util.init_models(
        './pcfgs/astronomers_pcfg.json')
    true_production_probs = util.get_production_probs(true_generative_model)

    width = 0.2
    fig, ax = plt.subplots(1, 1, figsize=(6, 3), dpi=100)
    ax.bar(np.arange(6) - width, true_production_probs['NP'].numpy(),
           width=width, color='black', label='True')
    for i, (train_mode, color) in enumerate(zip(['ws', 'vimco'],
                                                ['C1', 'C5'])):
                                                # ['#1f78b4', '#b2df8a'])):
        model_folder = util.get_most_recent_model_folder_args_match(
            num_iterations=num_iterations,
            logging_interval=logging_interval,
            eval_interval=eval_interval,
            checkpoint_interval=checkpoint_interval,
            batch_size=batch_size,
            seed=seed,
            train_mode=train_mode,
            num_particles=num_particles,
            exp_levenshtein=exp_levenshtein)
        generative_model, _ = util.load_models(model_folder)
        production_probs = util.get_production_probs(generative_model)
        ax.bar(np.arange(6) + i * width, production_probs['NP'].numpy(),
               width=width, color=color, label=train_mode.upper())

    ax.set_ylim(0, 0.5)
    ax.set_ylabel(r'$P(NP \to \cdot)$')
    ax.set_yticks([0, 0.5])
    ax.set_xticks(np.arange(6))
    ax.set_xticklabels(
        list(map(lambda x: ' '.join(x),
                 true_generative_model.grammar['productions']['NP'])))
    ax.tick_params(axis='x', length=0)
    sns.despine(ax=ax)
    ax.legend()

    fig.tight_layout()
    if not os.path.exists('./plots/'):
        os.makedirs('./plots/')
    filename = './plots/np_probs.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('saved to {}'.format(filename))


def write_posteriors():
    seed = 1
    num_particles = 20
    num_samples = 1000
    num_samples_to_show = 5
    sentence = ['astronomers', 'saw', 'stars', 'with', 'telescopes']

    to_write = ''
    for train_mode in ['ws', 'vimco']:
        to_write += '{}\n'.format(train_mode)
        model_folder = util.get_most_recent_model_folder_args_match(
            num_iterations=num_iterations,
            logging_interval=logging_interval,
            eval_interval=eval_interval,
            checkpoint_interval=checkpoint_interval,
            batch_size=batch_size,
            seed=seed,
            train_mode=train_mode,
            num_particles=num_particles,
            exp_levenshtein=exp_levenshtein)
        _, inference_network = util.load_models(model_folder)
        q_dist = util.get_inference_network_distribution(
            inference_network, sentence, num_samples=num_samples)
        trees = [x[0] for x in q_dist]
        log_weights = torch.cat([x[1].unsqueeze(0) for x in q_dist])
        probs = [x.item() for x in
                 util.exponentiate_and_normalize(log_weights)]
        for i, (tree, prob) in enumerate(zip(trees, probs)):
            if i < num_samples_to_show:
                nltk_tree = util.tree_to_nltk_tree(tree)
                to_write += '{},{}\n'.format(
                    prob, nltk_tree.pformat_latex_qtree())
    if not os.path.exists('./plots/'):
        os.makedirs('./plots/')
    filename = './plots/posteriors.txt'
    with open(filename, 'w') as f:
        f.write(to_write)
    print('saved to {}'.format(filename))


def main():
    plot_errors()
    # plot_errors_end_points()
    # plot_both()
    # plot_production_probs()
    # write_posteriors()


if __name__ == '__main__':
    main()
