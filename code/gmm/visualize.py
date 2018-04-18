from util import *
from gmm import *
from matplotlib.animation import FuncAnimation

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
import os


def visualize_linear_layer(linear_layer, ax):
    ax.imshow(np.concatenate(
        [linear_layer.weight.data.numpy(), linear_layer.bias.data.numpy().reshape([-1, 1])],
        axis=1
    ), cmap='Greys', vmin=-3, vmax=3, aspect='equal')
    ax.set_yticks([])
    ax.set_xticks([])
    # ax.set_xlabel('In')
    # ax.set_ylabel('Out')
    return ax


def visualize_mlp_2(mlp, axs=None):
    linear_layers = list(filter(
        None, [m if isinstance(m, nn.Linear) else False for m in mlp.modules()]
    ))

    if axs is None:
        fig, axs = plt.subplots(1, len(linear_layer))

    max_in = -1
    max_out = -1
    for (ax, linear_layer) in zip(axs, linear_layers):
        ax = visualize_linear_layer(linear_layer, ax)
        max_in = max(max_in, linear_layer.weight.size(0))
        max_out = max(max_out, linear_layer.weight.size(1) + 1)

    for ax in axs:
        ax.set_xlim(0, max_out)
        ax.set_ylim(0, max_in)
        ax.set_frame_on(False)

    return axs


def visualize_mlp(mlp, ax=None, vmin=-3, vmax=3, gap=2):
    linear_layers = list(filter(
        None, [m if isinstance(m, nn.Linear) else False for m in mlp.modules()]
    ))

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    total_num_rows = -1
    total_num_cols = -gap
    for linear_layer in linear_layers:
        total_num_rows = max(total_num_rows, linear_layer.weight.size(0))
        total_num_cols += linear_layer.weight.size(1) + 1 + gap

    image = vmin * np.ones([total_num_rows, total_num_cols])
    current_col_offset = 0
    for linear_layer_idx, linear_layer in enumerate(linear_layers):
        weights = np.concatenate(
            [linear_layer.weight.data.numpy(), linear_layer.bias.data.numpy().reshape([-1, 1])],
            axis=1
        )
        num_rows, num_cols = weights.shape
        image[0:num_rows, current_col_offset:(current_col_offset + num_cols)] = weights
        current_col_offset += num_cols + gap

    ax.imshow(image, cmap='Greys', vmin=-3, vmax=3, aspect='equal')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_frame_on(False)

    return ax


def main(args):
    softmax_multiplier = 0.5

    num_iterations = int(np.load('{}/num_iterations_{}.npy'.format(WORKING_DIR, args.uid)))
    logging_interval = int(np.load('{}/logging_interval_{}.npy'.format(WORKING_DIR, args.uid)))
    saving_interval = int(np.load('{}/saving_interval_{}.npy'.format(WORKING_DIR, args.uid)))
    num_mixtures = int(np.load('{}/num_mixtures_{}.npy'.format(WORKING_DIR, args.uid)))

    true_p_mixture_probs = np.load('{}/true_p_mixture_probs_{}.npy'.format(WORKING_DIR, args.uid))
    true_mean_multiplier = float(np.load('{}/true_mean_multiplier_{}.npy'.format(WORKING_DIR, args.uid)))
    true_log_stds = np.load('{}/true_log_stds_{}.npy'.format(WORKING_DIR, args.uid))

    p_init_mixture_probs_pre_softmax = np.random.rand(num_mixtures)
    init_log_stds = np.random.rand(num_mixtures)
    init_mean_multiplier = float(true_mean_multiplier)

    true_generative_network = GenerativeNetwork(np.log(true_p_mixture_probs) / softmax_multiplier, true_mean_multiplier, true_log_stds)

    num_q_layers = 3
    num_experiments = int(args.vimco or args.all) + int(args.reinforce or args.all) + int(args.concrete or args.all) + int(args.ww or args.all) * len(args.ww_probs) + int(args.ws or args.all)
    num_test_x = 5
    test_x = np.linspace(0, num_mixtures - 1, num=num_test_x) * true_mean_multiplier

    if args.visualize_weights or args.visualize_all:
        # Visualize inference network weights
        fig, axs = plt.subplots(num_experiments, 1)
        if num_experiments == 1:
            axs = [axs]
        fig.set_size_inches(num_q_layers * 1.0, num_experiments * 1.3)

        t = axs[0].set_title('Iteration {}'.format(0), color='black')
        ax_idx = 0

        if args.vimco or args.all:
            axs[ax_idx].set_ylabel('vimco')
            ax_idx += 1

        if args.reinforce or args.all:
            axs[ax_idx].set_ylabel('reinforce')
            ax_idx += 1

        if args.concrete or args.all:
            axs[ax_idx].set_ylabel('concrete')
            ax_idx += 1

        if args.ww or args.all:
            for q_mixture_prob_idx, q_mixture_prob in enumerate(args.ww_probs):
                axs[ax_idx].set_ylabel('ww {}'.format(q_mixture_prob))
                ax_idx += 1

        if args.ws or args.all:
            axs[ax_idx].set_ylabel('ws')
            ax_idx += 1

        fig.tight_layout()

        def update(frame):
            iteration = frame

            t.set_text('Iteration {}'.format(iteration))
            result = [t]

            ax_idx = 0

            if args.vimco or args.all:
                filename = '{}/vimco_{}_{}_{}.pt'.format(WORKING_DIR, iteration, args.seed, args.uid)
                iwae_vimco_state_dict = torch.load(filename)
                iwae_vimco = IWAE(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds)
                iwae_vimco.load_state_dict(iwae_vimco_state_dict)
                axs[ax_idx] = visualize_mlp(iwae_vimco.inference_network.mlp, axs[ax_idx])
                result = result + [axs[ax_idx].images[0]]
                ax_idx += 1

            if args.reinforce or args.all:
                filename = '{}/reinforce_{}_{}_{}.pt'.format(WORKING_DIR, iteration, args.seed, args.uid)
                iwae_reinforce_state_dict = torch.load(filename)
                iwae_reinforce = IWAE(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds)
                iwae_reinforce.load_state_dict(iwae_reinforce_state_dict)
                axs[ax_idx] = visualize_mlp(iwae_reinforce.inference_network.mlp, axs[ax_idx])
                result = result + [axs[ax_idx].images[0]]
                ax_idx += 1

            if args.ww or args.all:
                for q_mixture_prob_idx, q_mixture_prob in enumerate(args.ww_probs):
                    filename = '{}/ww{}_{}_{}_{}.pt'.format(WORKING_DIR, str(q_mixture_prob).replace('.', '-'), iteration, args.seed, args.uid)
                    ww_state_dict = torch.load(filename)
                    ww = RWS(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds)
                    ww.load_state_dict(ww_state_dict)
                    axs[ax_idx] = visualize_mlp(ww.inference_network.mlp, axs[ax_idx])
                    result = result + [axs[ax_idx].images[0]]
                    ax_idx += 1

            if args.ws or args.all:
                filename = '{}/ws_{}_{}_{}.pt'.format(WORKING_DIR, iteration, args.seed, args.uid)
                ws_state_dict = torch.load(filename)
                ws = RWS(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds)
                ws.load_state_dict(ws_state_dict)
                axs[ax_idx] = visualize_mlp(ws.inference_network.mlp, axs[ax_idx])
                result = result + [axs[ax_idx].images[0]]
                ax_idx += 1

            return result

        anim = FuncAnimation(fig, update, frames=np.arange(0, num_iterations, saving_interval), blit=True)
        # anim = FuncAnimation(fig, update, frames=np.arange(0, saving_interval * 2, saving_interval), blit=True)
        filename = '{}/visualize_weights_{}_{}.mp4'.format(OUTPUT_DIR, args.seed, args.uid)
        anim.save(filename, dpi=200)
        print('Saved to {}'.format(filename))

    if args.visualize_training or args.visualize_all:
        # Visualize generative and inference networks
        fig, axs = plt.subplots(2, num_test_x, sharex=True, sharey=True)
        fig.set_size_inches(2 * num_test_x, 5)

        ax = axs[0, 0]
        ax.plot(np.arange(num_mixtures), true_generative_network.get_z_params().data.numpy(), label='true', marker='x')

        for ax_idx, ax in enumerate(axs[1]):
            x = Variable(torch.Tensor([test_x[ax_idx]]))
            ax.set_title('$x = {}$'.format(x.data[0]))
            ax.set_xticks(np.arange(num_mixtures))
            ax.set_xlabel('z')
            ax.set_ylim(0, 1)
            ax.plot(np.arange(num_mixtures), true_generative_network.posterior(x).data.numpy()[0], label='true', marker='x')

        axs[1, 0].set_ylabel('$q_{\phi}(z | x)$')

        iteration = 0
        if args.vimco or args.all:
            q_iwae_lines = [None] * num_test_x
            filename = '{}/vimco_{}_{}_{}.pt'.format(WORKING_DIR, iteration, args.seed, args.uid)
            iwae_vimco_state_dict = torch.load(filename)
            iwae_vimco = IWAE(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds)
            iwae_vimco.load_state_dict(iwae_vimco_state_dict)
            p_iwae, = axs[0, 0].plot(np.arange(num_mixtures), iwae_vimco.generative_network.get_z_params().data.numpy(), label='vimco', marker='v')
            for ax_idx, ax in enumerate(axs[1]):
                x = Variable(torch.Tensor([test_x[ax_idx]]))
                q_iwae_lines[ax_idx], = ax.plot(np.arange(num_mixtures), iwae_vimco.inference_network.get_z_params(x).data.numpy()[0], label='vimco', marker='v')

        if args.reinforce or args.all:
            q_iwae_lines = [None] * num_test_x
            filename = '{}/reinforce_{}_{}_{}.pt'.format(WORKING_DIR, iteration, args.seed, args.uid)
            iwae_reinforce_state_dict = torch.load(filename)
            iwae_reinforce = IWAE(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds)
            iwae_reinforce.load_state_dict(iwae_reinforce_state_dict)
            p_iwae, = axs[0, 0].plot(np.arange(num_mixtures), iwae_reinforce.generative_network.get_z_params().data.numpy(), label='reinforce', marker='.')
            for ax_idx, ax in enumerate(axs[1]):
                x = Variable(torch.Tensor([test_x[ax_idx]]))
                q_iwae_lines[ax_idx], = ax.plot(np.arange(num_mixtures), iwae_reinforce.inference_network.get_z_params(x).data.numpy()[0], label='reinforce', marker='.')

        if args.ww or args.all:
            q_ww_lines = [None] * (len(args.ww_probs) * num_test_x)
            p_ww_lines = [None] * len(args.ww_probs)
            for q_mixture_prob_idx, q_mixture_prob in enumerate(args.ww_probs):
                q_mixture_prob_color = str(q_mixture_prob * 0.6)
                filename = '{}/ww{}_{}_{}_{}.pt'.format(WORKING_DIR, str(q_mixture_prob).replace('.', '-'), iteration, args.seed, args.uid)
                ww_state_dict = torch.load(filename)
                ww = RWS(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds)
                ww.load_state_dict(ww_state_dict)
                p_ww_lines[q_mixture_prob_idx], = axs[0, 0].plot(np.arange(num_mixtures), ww.generative_network.get_z_params().data.numpy(), label='ww {}'.format(q_mixture_prob), marker='o', color=q_mixture_prob_color)
                for ax_idx, ax in enumerate(axs[1]):
                    x = Variable(torch.Tensor([test_x[ax_idx]]))
                    q_ww_lines[q_mixture_prob_idx * num_test_x + ax_idx], = ax.plot(np.arange(num_mixtures), ww.inference_network.get_z_params(x).data.numpy()[0], label='ww {}'.format(q_mixture_prob), marker='o', color=q_mixture_prob_color)

        if args.ws or args.all:
            q_ws_lines = [None] * num_test_x
            filename = '{}/ws_{}_{}_{}.pt'.format(WORKING_DIR, iteration, args.seed, args.uid)
            ws_state_dict = torch.load(filename)
            ws = RWS(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds)
            ws.load_state_dict(ws_state_dict)
            p_ws, = axs[0, 0].plot(np.arange(num_mixtures), ws.generative_network.get_z_params().data.numpy(), label='ws', marker='^')
            for ax_idx, ax in enumerate(axs[1]):
                x = Variable(torch.Tensor([test_x[ax_idx]]))
                q_ws_lines[ax_idx], = ax.plot(np.arange(num_mixtures), ws.inference_network.get_z_params(x).data.numpy()[0], label='ws', marker='^')

        axs[0, 0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        axs[0, 0].set_ylabel('$p_{\\theta}(z)$')
        axs[0, 0].set_xticks(np.arange(num_mixtures))
        axs[0, 0].set_ylim(0, 1)

        for ax in axs[0, 1:]:
            ax.remove()

        t = axs[0][0].set_title('Iteration {}'.format(iteration), color='black')

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        def update(frame):
            iteration = frame

            t.set_text('Iteration {}'.format(iteration))
            result = [t]

            if args.vimco or args.all:
                filename = '{}/vimco_{}_{}_{}.pt'.format(WORKING_DIR, iteration, args.seed, args.uid)
                iwae_vimco_state_dict = torch.load(filename)
                iwae_vimco = IWAE(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds)
                iwae_vimco.load_state_dict(iwae_vimco_state_dict)
                p_iwae.set_data(np.arange(num_mixtures), iwae_vimco.generative_network.get_z_params().data.numpy())
                for ax_idx in range(num_test_x):
                    x = Variable(torch.Tensor([test_x[ax_idx]]))
                    q_iwae_lines[ax_idx].set_data(np.arange(num_mixtures), iwae_vimco.inference_network.get_z_params(x).data.numpy()[0])
                result = result + [p_iwae] + q_iwae_lines

            if args.reinforce or args.all:
                filename = '{}/reinforce_{}_{}_{}.pt'.format(WORKING_DIR, iteration, args.seed, args.uid)
                iwae_reinforce_state_dict = torch.load(filename)
                iwae_reinforce = IWAE(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds)
                iwae_reinforce.load_state_dict(iwae_reinforce_state_dict)
                p_iwae.set_data(np.arange(num_mixtures), iwae_reinforce.generative_network.get_z_params().data.numpy())
                for ax_idx in range(num_test_x):
                    x = Variable(torch.Tensor([test_x[ax_idx]]))
                    q_iwae_lines[ax_idx].set_data(np.arange(num_mixtures), iwae_reinforce.inference_network.get_z_params(x).data.numpy()[0])
                result = result + [p_iwae] + q_iwae_lines

            if args.ww or args.all:
                for q_mixture_prob_idx, q_mixture_prob in enumerate(args.ww_probs):
                    filename = '{}/ww{}_{}_{}_{}.pt'.format(WORKING_DIR, str(q_mixture_prob).replace('.', '-'), iteration, args.seed, args.uid)
                    ww_state_dict = torch.load(filename)
                    ww = RWS(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds)
                    ww.load_state_dict(ww_state_dict)
                    p_ww_lines[q_mixture_prob_idx].set_data(np.arange(num_mixtures), ww.generative_network.get_z_params().data.numpy())
                    for ax_idx in range(num_test_x):
                        x = Variable(torch.Tensor([test_x[ax_idx]]))
                        q_ww_lines[q_mixture_prob_idx * num_test_x + ax_idx].set_data(np.arange(num_mixtures), ww.inference_network.get_z_params(x).data.numpy()[0])
                result = result + p_ww_lines + q_ww_lines

            if args.ws or args.all:
                filename = '{}/ws_{}_{}_{}.pt'.format(WORKING_DIR, iteration, args.seed, args.uid)
                ws_state_dict = torch.load(filename)
                ws = RWS(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds)
                ws.load_state_dict(ws_state_dict)
                p_ws.set_data(np.arange(num_mixtures), ws.generative_network.get_z_params().data.numpy())
                for ax_idx in range(num_test_x):
                    x = Variable(torch.Tensor([test_x[ax_idx]]))
                    q_ws_lines[ax_idx].set_data(np.arange(num_mixtures), ws.inference_network.get_z_params(x).data.numpy()[0])
                result = result + [p_ws] + q_ws_lines

            return result

        anim = FuncAnimation(fig, update, frames=np.arange(0, num_iterations, saving_interval), blit=True)
        filename = '{}/visualize_training_{}_{}.mp4'.format(OUTPUT_DIR, args.seed, args.uid)
        anim.save(filename, dpi=200)
        print('Saved to {}'.format(filename))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='GMM open universe')
    parser.add_argument('--uid', type=str, default='', metavar='U',
                        help='run args.uid')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='run seed')
    parser.add_argument('--all', action='store_true', default=False)
    parser.add_argument('--reinforce', action='store_true', default=False)
    parser.add_argument('--vimco', action='store_true', default=False)
    parser.add_argument('--ws', action='store_true', default=False)
    parser.add_argument('--ww', action='store_true', default=False)
    parser.add_argument('--ww-probs', nargs='*', type=float, default=[1.0])
    parser.add_argument('--wsw', action='store_true', default=False)
    parser.add_argument('--wswa', action='store_true', default=False)
    parser.add_argument('--concrete', action='store_true', default=False)
    parser.add_argument('--visualize-weights', action='store_true', default=False)
    parser.add_argument('--visualize-training', action='store_true', default=False)
    parser.add_argument('--visualize-all', action='store_true', default=False)
    args = parser.parse_args()

    main(args)
