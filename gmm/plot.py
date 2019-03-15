import util
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.lines as mlines
import torch
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

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
plt.rc('xtick.major', width=0.5)         # set the value globally
plt.rc('ytick.major', width=0.5)         # set the value globally
plt.rc('lines', linewidth=1)             # line thickness
plt.rc('ytick.major', size=2)            # set the value globally
plt.rc('xtick.major', size=2)            # set the value globally
plt.rc('text', usetex=True)
plt.rc('text.latex',
       preamble=[r'\usepackage{amsmath}',
                 r'\usepackage[cm]{sfmath}'])  # for \text command
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['cm']})
plt.rc('axes', titlepad=3)

colors = ['C0', 'C3', 'C4', 'C5', 'C1', 'C6', 'C7']
linestyles = ['dashed', 'dashed', 'dashed', 'dashed', 'solid', 'solid',
              'solid']
labels = ['Concrete', 'RELAX', 'REINFORCE', 'VIMCO', 'WS', 'WW',
          r'$\delta$-WW']
train_mode_list = ['concrete', 'relax', 'reinforce', 'vimco', 'ws', 'ww',
                   'dww']
num_particles_list = [2, 5, 10, 20]
seed_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


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

    num_not_nan = np.count_nonzero(~np.isnan(mid))

    if num_not_nan > 0:
        ax.plot(np.arange(num_not_nan), mid[:num_not_nan], **plot_kwargs)
        ax.fill_between(np.arange(num_not_nan),
                        low[:num_not_nan], high[:num_not_nan],
                        alpha=0.2, **plot_kwargs)


def load_errors():
    len_history = 100
    p_error = np.full((len(seed_list), len(train_mode_list),
                       len(num_particles_list), len_history),
                      np.nan, dtype=np.float)
    q_error = np.full((len(seed_list), len(train_mode_list),
                       len(num_particles_list), len_history),
                      np.nan, dtype=np.float)
    grad_std = np.full((len(seed_list), len(train_mode_list),
                        len(num_particles_list), len_history),
                       np.nan, dtype=np.float)
    for seed_idx, seed in enumerate(seed_list):
        for train_mode_idx, train_mode in enumerate(train_mode_list):
            for num_particles_idx, num_particles in enumerate(
                num_particles_list
            ):
                model_folder = util.get_most_recent_model_folder_args_match(
                    seed=seed,
                    train_mode=train_mode,
                    num_particles=num_particles)
                if model_folder is not None:
                    stats = util.load_object(
                        util.get_stats_path(model_folder))
                    p_error[
                        seed_idx, train_mode_idx,
                        num_particles_idx, :len(stats.p_error_history)
                    ] = stats.p_error_history
                    q_error[
                        seed_idx, train_mode_idx,
                        num_particles_idx, :len(stats.q_error_history)
                    ] = stats.q_error_history
                    grad_std[
                        seed_idx, train_mode_idx,
                        num_particles_idx, :len(stats.grad_std_history)
                    ] = stats.grad_std_history
    return p_error, q_error, grad_std


def plot_errors():
    p_error, q_error, grad_std = load_errors()
    fig, axss = plt.subplots(3, len(num_particles_list), sharex=True,
                             sharey='row')
    fig.set_size_inches(5.5, 3)
    for train_mode_idx, train_mode in enumerate(train_mode_list):
        for num_particles_idx, num_particles in enumerate(num_particles_list):
            print('{} {}'.format(train_mode, num_particles))
            color = colors[train_mode_idx]
            linestyle = linestyles[train_mode_idx]
            plot_with_error_bars(
                axss[0, num_particles_idx],
                p_error[:, train_mode_idx, num_particles_idx, :],
                color=color, linestyle=linestyle)
            plot_with_error_bars(
                axss[1, num_particles_idx],
                q_error[:, train_mode_idx, num_particles_idx, :],
                color=color, linestyle=linestyle)
            plot_with_error_bars(
                axss[2, num_particles_idx],
                grad_std[:, train_mode_idx, num_particles_idx, :],
                color=color, linestyle=linestyle)
    axss[0, 0].set_yscale('log')
    axss[0, 0].set_ylabel(
        r'$\| p_{\theta}(z) - p_{\theta_{\text{true}}}(z) \|$')

    axss[1, 0].set_yscale('log')
    axss[1, 0].set_ylabel(
        'Avg. test\n' +
        r'$\| q_\phi(z | x) - p_{\theta_{\text{true}}}(z | x) \|$')

    axss[2, 0].set_yscale('log')
    axss[2, 0].set_ylabel('Std. of $\phi$ \n gradient est.')

    handles = []
    for train_mode_idx, train_mode in enumerate(train_mode_list):
        handles.append(mlines.Line2D([0], [1], color=colors[train_mode_idx],
                                     linestyle=linestyles[train_mode_idx]))
    axss[-1, 1].legend(
        handles, labels, bbox_to_anchor=(1, -0.2), loc='upper center',
        ncol=len(train_mode_list))

    for i, ax in enumerate(axss[0]):
        ax.set_title('$K = {}$'.format(num_particles_list[i]))
    for i, ax in enumerate(axss[-1]):
        ax.set_xlabel('Iteration')
        ax.xaxis.set_label_coords(0.5, -0.1)

    for axs in axss:
        for ax in axs:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.minorticks_off()
            ax.set_xticks([0, 100])
            ax.set_xticklabels([1, 100000])

    fig.tight_layout(pad=0)
    if not os.path.exists('./plots/'):
        os.makedirs('./plots/')
    filename = './plots/errors.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))


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
            ax.add_artist(mpatches.Rectangle((x, y), width, height, **kwargs))
    return ax


def plot_models():
    saving_iterations = np.arange(100) * 1000
    num_iterations_to_plot = 3
    iterations_to_plot = saving_iterations[np.floor(
        np.linspace(0, 99, num=num_iterations_to_plot)
    ).astype(int)]
    num_test_x = 3
    num_particles_list = [2, 20]
    seed = seed_list[0]
    model_folder = util.get_most_recent_model_folder_args_match(
        seed=seed_list[0],
        train_mode=train_mode_list[0],
        num_particles=num_particles_list[0])
    args = util.load_object(util.get_args_path(model_folder))
    _, _, true_generative_model = util.init_models(args)
    test_xs = np.linspace(0, 19, num=num_test_x) * 10

    nrows = num_iterations_to_plot
    ncols = len(num_particles_list) * (num_test_x + 1)
    fig, axss = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    width = 5.5
    ax_width = width / ncols
    height = nrows * ax_width
    fig.set_size_inches(width, height)
    for iteration_idx, iteration in enumerate(iterations_to_plot):
        axss[iteration_idx, 0].set_ylabel('Iter. {}'.format(iteration))
        for num_particles_idx, num_particles in enumerate(num_particles_list):
            ax = axss[iteration_idx, num_particles_idx * (num_test_x + 1)]
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_ylim(0, 8)
            ax.set_xlim(0, 20)
            if iteration_idx == 0:
                ax.set_title(r'$p_\theta(z)$')

            # true generative model
            i = 0
            plot_hinton(
                ax, true_generative_model.get_latent_params().data.numpy(),
                8 - i, 8 - i - 1, 0, 20, color='black')

            # learned generative models
            for train_mode_idx, train_mode in enumerate(train_mode_list):
                label = labels[train_mode_idx]
                color = colors[train_mode_idx]
                model_folder = util.get_most_recent_model_folder_args_match(
                    seed=seed, train_mode=train_mode,
                    num_particles=num_particles)
                if model_folder is not None:
                    generative_model, _ = util.load_models(
                        model_folder, iteration=iteration)
                    if generative_model is not None:
                        plot_hinton(
                            ax,
                            generative_model.get_latent_params().data.numpy(),
                            8 - train_mode_idx - 1, 8 - train_mode_idx - 2, 0,
                            20, label=label, color=color)

            # inference network
            for test_x_idx, test_x in enumerate(test_xs):
                ax = axss[iteration_idx, num_particles_idx * (num_test_x + 1)
                          + test_x_idx + 1]
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.set_yticks([])
                ax.set_yticklabels([])
                ax.set_ylim(0, 8)
                ax.set_xlim(0, 20)
                test_x_tensor = torch.tensor(test_x, dtype=torch.float,
                                             device=args.device).unsqueeze(0)
                if iteration_idx == 0:
                    ax.set_title(r'$q_\phi(z | x = {0:.0f})$'.format(test_x))

                # true
                plot_hinton(
                    ax, true_generative_model.get_posterior_probs(
                        test_x_tensor)[0].data.numpy(), 8 - i, 8 - i - 1, 0,
                    20, color='black')

                # learned
                for train_mode_idx, train_mode in enumerate(train_mode_list):
                    label = labels[train_mode_idx]
                    color = colors[train_mode_idx]
                    model_folder = \
                        util.get_most_recent_model_folder_args_match(
                            seed=seed, train_mode=train_mode,
                            num_particles=num_particles)
                    if model_folder is not None:
                        _, inference_network = util.load_models(
                            model_folder, iteration=iteration)
                        if inference_network is not None:
                            plot_hinton(
                                ax, inference_network.get_latent_params(
                                    test_x_tensor)[0].data.numpy(),
                                8 - train_mode_idx - 1, 8 - train_mode_idx - 2,
                                0, 20, label=label, color=color)

    for num_particles_idx, num_particles in enumerate(num_particles_list):
        ax = axss[0, num_particles_idx * (num_test_x + 1) +
                  (num_test_x + 1) // 2]
        ax.text(0, 1.25, '$K = {}$'.format(num_particles), fontsize=SMALL_SIZE,
                verticalalignment='bottom', horizontalalignment='center',
                transform=ax.transAxes)

    handles = [mpatches.Rectangle((0, 0), 1, 1, color='black', label='True')]
    for color, label in zip(colors, labels):
        handles.append(mpatches.Rectangle((0, 0), 1, 1,
                       color=color, label=label))
    axss[-1, ncols // 2].legend(bbox_to_anchor=(0, -0.1), loc='upper center',
                                ncol=len(handles), handles=handles)

    for ax in axss[-1]:
        ax.set_xlabel(r'$z$', labelpad=0.5)

    fig.tight_layout(pad=0)
    if not os.path.exists('./plots/'):
        os.makedirs('./plots/')
    filename = './plots/models.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))


def plot_model_movie():
    num_test_x = 5
    num_particles_list = [2, 5, 10, 20]
    seed = seed_list[0]
    model_folder = util.get_most_recent_model_folder_args_match(
        seed=seed_list[0],
        train_mode=train_mode_list[0],
        num_particles=num_particles_list[0])
    args = util.load_object(util.get_args_path(model_folder))
    _, _, true_generative_model = util.init_models(args)
    test_xs = np.linspace(0, 19, num=num_test_x) * 10

    nrows = len(num_particles_list)
    ncols = num_test_x + 1
    width = 5.5
    ax_width = width / ncols
    height = nrows * ax_width
    fig, axss = plt.subplots(nrows, ncols, sharex=True, sharey=True, dpi=300)
    fig.set_size_inches(width, height)

    for num_particles_idx, num_particles in enumerate(num_particles_list):
        axss[num_particles_idx, 0].set_ylabel(
            '$K = {}$'.format(num_particles), fontsize=SMALL_SIZE)

    handles = [mpatches.Rectangle((0, 0), 1, 1, color='black', label='True')]
    for color, label in zip(colors, labels):
        handles.append(mpatches.Rectangle((0, 0), 1, 1,
                       color=color, label=label))
    axss[-1, ncols // 2].legend(bbox_to_anchor=(0, -0.05), loc='upper center',
                                ncol=len(handles), handles=handles)

    axss[0, 0].set_title(r'$p_\theta(z)$')
    for test_x_idx, test_x in enumerate(test_xs):
        axss[0, 1 + test_x_idx].set_title(r'$q_\phi(z | x = {0:.0f})$'.format(
            test_x))
    for ax in axss[-1]:
        ax.set_xlabel(r'$z$', labelpad=0.5)

    for axs in axss:
        for ax in axs:
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_ylim(0, 8)
            ax.set_xlim(0, 20)
    # title = fig.suptitle('Iteration 0')
    t = axss[0, ncols // 2].text(
        0, 1.23, 'Iteration 0',
        horizontalalignment='center', verticalalignment='center',
        transform=axss[0, ncols // 2].transAxes, fontsize=MEDIUM_SIZE)

    fig.tight_layout(pad=0, rect=[0.01, 0.04, 0.99, 0.96])

    def update(frame):
        result = []
        iteration_idx = frame
        iteration = iteration_idx * 1000
        t.set_text('Iteration {}'.format(iteration))
        result.append(t)

        for axs in axss:
            for ax in axs:
                result.append(ax.add_artist(
                    mpatches.Rectangle((0, 0), 20, 8, color='white')))
        for num_particles_idx, num_particles in enumerate(num_particles_list):
            ax = axss[num_particles_idx, 0]

            # true generative model
            i = 0
            plot_hinton(
                ax, true_generative_model.get_latent_params().data.numpy(),
                8 - i, 8 - i - 1, 0, 20, color='black')

            # learned generative models
            for train_mode_idx, train_mode in enumerate(train_mode_list):
                label = labels[train_mode_idx]
                color = colors[train_mode_idx]
                model_folder = util.get_most_recent_model_folder_args_match(
                    seed=seed, train_mode=train_mode,
                    num_particles=num_particles)
                if model_folder is not None:
                    generative_model, _ = util.load_models(
                        model_folder, iteration=iteration)
                    if generative_model is not None:
                        plot_hinton(
                            ax,
                            generative_model.get_latent_params().data.numpy(),
                            8 - train_mode_idx - 1, 8 - train_mode_idx - 2, 0,
                            20, label=label, color=color)

            result += ax.artists

            # inference network
            for test_x_idx, test_x in enumerate(test_xs):
                ax = axss[num_particles_idx, test_x_idx + 1]
                test_x_tensor = torch.tensor(test_x, dtype=torch.float,
                                             device=args.device).unsqueeze(0)

                # true
                plot_hinton(
                    ax, true_generative_model.get_posterior_probs(
                        test_x_tensor)[0].data.numpy(), 8 - i, 8 - i - 1, 0,
                    20, color='black')

                # learned
                for train_mode_idx, train_mode in enumerate(train_mode_list):
                    label = labels[train_mode_idx]
                    color = colors[train_mode_idx]
                    model_folder = \
                        util.get_most_recent_model_folder_args_match(
                            seed=seed, train_mode=train_mode,
                            num_particles=num_particles)
                    if model_folder is not None:
                        _, inference_network = util.load_models(
                            model_folder, iteration=iteration)
                        if inference_network is not None:
                            plot_hinton(
                                ax, inference_network.get_latent_params(
                                    test_x_tensor)[0].data.numpy(),
                                8 - train_mode_idx - 1, 8 - train_mode_idx - 2,
                                0, 20, label=label, color=color)
                result += ax.artists
        return result

    anim = FuncAnimation(fig, update, frames=np.arange(100), blit=True)
    if not os.path.exists('./plots/'):
        os.makedirs('./plots/')
    filename = './plots/model_movie.mp4'
    anim.save(filename, dpi=300)
    print('Saved to {}'.format(filename))


def main():
    plot_errors()
    plot_models()
    plot_model_movie()


if __name__ == '__main__':
    main()
