from gmm import *
from plot import np_load
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
    ), cmap='Greys', vmin=-7, vmax=7, aspect='equal')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel('In')
    ax.set_ylabel('Out')
    return ax

def visualize_mlp(mlp):
    linear_layers = list(filter(
        None, [m if isinstance(m, nn.Linear) else False for m in mlp.modules()]
    ))
    fig, axs = plt.subplots(1, len(linear_layers))
    for (ax, linear_layer) in zip(axs, linear_layers):
        ax = visualize_linear_layer(linear_layer, ax)

    return fig, axs

def main(args):
    softmax_multiplier = 0.5

    num_iterations = int(np_load('num_iterations'))
    logging_interval = int(np_load('logging_interval'))
    saving_interval = int(np_load('saving_interval'))
    num_mixtures = int(np_load('num_mixtures'))

    true_p_mixture_probs = np_load('true_p_mixture_probs')
    true_mean_multiplier = float(np_load('true_mean_multiplier'))
    true_log_stds = np_load('true_log_stds')

    p_init_mixture_probs_pre_softmax = np.random.rand(num_mixtures)
    init_log_stds = np.random.rand(num_mixtures)
    init_mean_multiplier = float(true_mean_multiplier)


    true_generative_network = GenerativeNetwork(np.log(true_p_mixture_probs) / softmax_multiplier, true_mean_multiplier, true_log_stds)

    fig, axs = plt.subplots(2, num_mixtures, sharex=True, sharey=True)
    fig.set_size_inches(16, 5)

    iteration = 0

    if args.vimco or args.all:
        filename = 'vimco_{}_{}_{}.pt'.format(iteration, SEED, UID)
        iwae_vimco_state_dict = torch.load(filename)
        iwae_vimco = IWAE(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds)
        iwae_vimco.load_state_dict(iwae_vimco_state_dict)

    if args.ww or args.all:
        filename = 'ww_{}_{}_{}.pt'.format(iteration, SEED, UID)
        ww_state_dict = torch.load(filename)
        ww = RWS(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds)
        ww.load_state_dict(ww_state_dict)

    if args.ws or args.all:
        filename = 'ws_{}_{}_{}.pt'.format(iteration, SEED, UID)
        ws_state_dict = torch.load(filename)
        ws = RWS(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds)
        ws.load_state_dict(ws_state_dict)

    ax = axs[0, 0]
    ax.plot(np.arange(num_mixtures), true_generative_network.get_z_params().data.numpy(), label='true', marker='x')

    if args.vimco or args.all:
        l_iwae, = ax.plot(np.arange(num_mixtures), iwae_vimco.generative_network.get_z_params().data.numpy(), label='vimco', marker='v')

    if args.ww or args.all:
        l_ww, = ax.plot(np.arange(num_mixtures), ww.generative_network.get_z_params().data.numpy(), label='ww', marker='o')

    if args.ws or args.all:
        l_ws, = ax.plot(np.arange(num_mixtures), ws.generative_network.get_z_params().data.numpy(), label='ws', marker='^')

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel('$p_{\\theta}(z)$')
    ax.set_xticks(np.arange(num_mixtures))
    ax.set_ylim(0, 1)

    for ax in axs[0, 1:]:
        ax.remove()

    if args.vimco or args.all:
        q_iwae_lines = [None] * num_mixtures
    else:
        q_iwae_lines = []

    if args.ww or args.all:
        q_ww_lines = [None] * num_mixtures
    else:
        q_ww_lines = []

    if args.ws or args.all:
        q_ws_lines = [None] * num_mixtures
    else:
        q_ws_lines = []

    for ax_idx, ax in enumerate(axs[1]):
        x = Variable(torch.Tensor([true_mean_multiplier * ax_idx]))
        ax.set_title('$x = {}$'.format(x.data[0]))
        ax.set_xticks(np.arange(num_mixtures))
        ax.set_xlabel('z')
        ax.set_ylim(0, 1)
        ax.plot(np.arange(num_mixtures), true_generative_network.posterior(x).data.numpy()[0], label='true', marker='x')
        if args.vimco or args.all:
            q_iwae_lines[ax_idx], = ax.plot(np.arange(num_mixtures), iwae_vimco.inference_network.get_z_params(x).data.numpy()[0], label='vimco', marker='v')

        if args.ww or args.all:
            q_ww_lines[ax_idx], = ax.plot(np.arange(num_mixtures), ww.inference_network.get_z_params(x).data.numpy()[0], label='ww', marker='o')

        if args.ws or args.all:
            q_ws_lines[ax_idx], = ax.plot(np.arange(num_mixtures), ws.inference_network.get_z_params(x).data.numpy()[0], label='ws', marker='^')

    axs[1, 0].set_ylabel('$q_{\phi}(z | x)$')

    t = axs[0][0].set_title('Iteration {}'.format(iteration), color='black')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    def update(frame):
        iteration = frame

        t.set_text('Iteration {}'.format(iteration))
        result = [t]

        if args.vimco or args.all:
            filename = 'vimco_{}_{}_{}.pt'.format(iteration, SEED, UID)
            iwae_vimco_state_dict = torch.load(filename)
            iwae_vimco = IWAE(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds)
            iwae_vimco.load_state_dict(iwae_vimco_state_dict)
            l_iwae.set_data(np.arange(num_mixtures), iwae_vimco.generative_network.get_z_params().data.numpy())
            result = result + [l_iwae]

        if args.ww or args.all:
            filename = 'ww_{}_{}_{}.pt'.format(iteration, SEED, UID)
            ww_state_dict = torch.load(filename)
            ww = RWS(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds)
            ww.load_state_dict(ww_state_dict)
            l_ww.set_data(np.arange(num_mixtures), ww.generative_network.get_z_params().data.numpy())
            result = result + [l_ww]

        if args.ws or args.all:
            filename = 'ws_{}_{}_{}.pt'.format(iteration, SEED, UID)
            ws_state_dict = torch.load(filename)
            ws = RWS(p_init_mixture_probs_pre_softmax, init_mean_multiplier, init_log_stds)
            ws.load_state_dict(ws_state_dict)
            l_ws.set_data(np.arange(num_mixtures), ws.generative_network.get_z_params().data.numpy())
            result = result + [l_ws]

        for ax_idx in range(num_mixtures):
            x = Variable(torch.Tensor([true_mean_multiplier * ax_idx]))
            if args.vimco or args.all:
                q_iwae_lines[ax_idx].set_data(np.arange(num_mixtures), iwae_vimco.inference_network.get_z_params(x).data.numpy()[0])

            if args.ww or args.all:
                q_ww_lines[ax_idx].set_data(np.arange(num_mixtures), ww.inference_network.get_z_params(x).data.numpy()[0])

            if args.ws or args.all:
                q_ws_lines[ax_idx].set_data(np.arange(num_mixtures), ws.inference_network.get_z_params(x).data.numpy()[0])

        result = result + q_iwae_lines + q_ww_lines + q_ws_lines

        return result


    anim = FuncAnimation(fig, update, frames=np.arange(0, num_iterations, saving_interval), blit=True)
    filename = 'train.mp4'
    anim.save(filename, dpi=200)
    print('Saved to {}'.format(filename))


def np_load(filename):
    return np.load('{}_{:d}_{}.npy'.format(
        os.path.splitext(filename)[0], SEED, UID
    ))


SEED = 1
UID = ''

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='GMM open universe')
    parser.add_argument('--uid', type=str, default='', metavar='U',
                        help='run UID')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='run seed')
    parser.add_argument('--all', action='store_true', default=False)
    parser.add_argument('--reinforce', action='store_true', default=False)
    parser.add_argument('--vimco', action='store_true', default=False)
    parser.add_argument('--ws', action='store_true', default=False)
    parser.add_argument('--ww', action='store_true', default=False)
    parser.add_argument('--wsw', action='store_true', default=False)
    parser.add_argument('--wswa', action='store_true', default=False)
    args = parser.parse_args()
    SEED = args.seed
    UID = args.uid

    main(args)
