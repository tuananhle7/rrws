import util
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib.lines as mlines

# num_iterations = 5000
# logging_interval = 10
# eval_interval = 10
# checkpoint_interval = 100
# batch_size = 10
# seed_list = [1, 2, 3, 4, 5]
# train_mode_list = ['reinforce', 'ws', 'vimco', 'ww']
# num_particles_list = [5, 10, 20, 50, 100]

num_iterations = 2000
logging_interval = 10
eval_interval = 10
checkpoint_interval = 100
batch_size = 2
seed_list = [1, 2, 3, 4, 5, 6]
train_mode_list = ['reinforce', 'ws', 'vimco', 'ww']
num_particles_list = [2, 5, 10, 20, 50]


def plot_with_error_bars(ax, data, **plot_kwargs):
    mid = np.nanmedian(data, axis=0)
    low = np.nanpercentile(data, 10, axis=0)
    high = np.nanpercentile(data, 90, axis=0)
    num_not_nan = np.count_nonzero(~np.isnan(mid))
    ax.plot(np.arange(num_not_nan), mid[:num_not_nan], **plot_kwargs)
    ax.fill_between(np.arange(num_not_nan),
                    low[:num_not_nan], high[:num_not_nan],
                    alpha=0.5, **plot_kwargs)


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
                    num_particles=num_particles)
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
    fig, axss = plt.subplots(nrows=3, ncols=len(num_particles_list),
                             figsize=(12, 6), sharex=True, sharey='row')

    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c']
    for train_mode_idx, train_mode in enumerate(train_mode_list):
        for num_particles_idx, num_particles in enumerate(num_particles_list):
            color = colors[train_mode_idx]
            plot_with_error_bars(
                axss[0, num_particles_idx],
                p_error[:, train_mode_idx, num_particles_idx, :], color=color)
            plot_with_error_bars(
                axss[1, num_particles_idx],
                q_error_model[:, train_mode_idx, num_particles_idx, :],
                color=color)
            plot_with_error_bars(
                axss[2, num_particles_idx],
                q_error_true[:, train_mode_idx, num_particles_idx, :],
                color=color)

    handles = []
    for train_mode_idx, train_mode in enumerate(train_mode_list):
        handles.append(mlines.Line2D([0], [1], color=colors[train_mode_idx]))
    axss[-1, 2].legend(handles, train_mode_list, bbox_to_anchor=(0.5, -0.2),
                       loc='upper center', ncol=len(train_mode_list))

    axss[0, 0].set_ylabel(r'$p_{\theta}$ error', labelpad=-17)
    axss[1, 0].set_ylabel(r'$q_\phi$ error to $p_{\theta}$', labelpad=-5)
    axss[2, 0].set_ylabel(r'$q_\phi$ error to $p_{true}$', labelpad=-15)

    for ax in axss[0]:
        ax.set_ylim(0, 0.3)
    for ax in axss[1]:
        ax.set_ylim(0, 20)
    for ax in axss[-1]:
        ax.set_ylim(0, 40)

    for axs in axss:
        for ax in axs:
            ax.set_xticks([0, num_iterations / eval_interval - 1])
            ax.set_xticklabels([1, num_iterations])
            ax.set_yticks([0, ax.get_yticks()[-1]])
            sns.despine(ax=ax, trim=True)

    for ax in axss[-1]:
        ax.set_xlabel('iteration', labelpad=-10)

    for ax, num_particles in zip(axss[0], num_particles_list):
        ax.set_title(r'$K = {}$'.format(num_particles))

    fig.tight_layout()
    if not os.path.exists('./plots/'):
        os.makedirs('./plots/')
    filename = './plots/errors.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('saved to {}'.format(filename))


def main():
    plot_errors()


if __name__ == '__main__':
    main()
