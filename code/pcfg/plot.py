import util
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def plot_losses():
    stats_ws = util.load_object('./trained_models/ws/stats.pkl')
    stats_ww = util.load_object('./trained_models/ww/stats.pkl')
    stats_vimco = util.load_object('./trained_models/vimco/stats.pkl')

    fig, ax = plt.subplots(1, 1, dpi=100, figsize=(4, 3))
    ax.plot(stats_ws.elbo_history, label='ws')
    ax.plot(stats_ww.elbo_history, label='ww')
    ax.plot(stats_vimco.elbo_history, label='vimco')
    ax.set_ylabel(r'ELBO')
    ax.legend()

    for ax in axs:
        ax.set_xlabel('iteration', labelpad=-10)
        ax.set_xticks([0, len(stats_ww.p_error_history) - 1])
        ax.set_xticklabels([
            1, stats_ww.eval_interval * len(stats_ww.p_error_history)])
        sns.despine(ax=ax, trim=True)

    fig.tight_layout()
    if not os.path.exists('./plots/'):
        os.makedirs('./plots/')
    filename = './plots/losses.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('saved to {}'.format(filename))


def plot_eval():
    stats_ws = util.load_object('./trained_models/ws/stats.pkl')
    stats_ww = util.load_object('./trained_models/ww/stats.pkl')
    stats_vimco = util.load_object('./trained_models/vimco/stats.pkl')

    fig, axs = plt.subplots(1, 3, dpi=100, figsize=(12, 3))
    axs[0].plot(stats_ww.p_error_history, label='ww')
    axs[0].plot(stats_ws.p_error_history, label='ws')
    axs[0].plot(stats_vimco.p_error_history, label='vimco')
    axs[0].set_ylabel(r'Error($p_{true}(z, x)$, $p_\theta(z, x)$)')
    axs[0].set_title('generative model error')

    axs[1].plot(stats_ws.q_error_to_model_history, label='ws')
    axs[1].plot(stats_ww.q_error_to_model_history, label='ww')
    axs[1].plot(stats_vimco.q_error_to_model_history, label='vimco')
    axs[1].set_ylabel(r'Error($p_\theta(z | x)$, $q_\phi(z | x)$) + const.')
    axs[1].set_title('inference network error')

    axs[2].plot(stats_ws.q_error_to_true_history, label='ws')
    axs[2].plot(stats_ww.q_error_to_true_history, label='ww')
    axs[2].plot(stats_vimco.q_error_to_true_history, label='vimco')
    axs[2].set_ylabel(r'Error($p_{true}(z | x)$, $q_\phi(z | x)$) + const.')
    axs[2].set_title('inference network error')

    axs[-1].legend()

    for ax in axs:
        ax.set_xlabel('iteration', labelpad=-10)
        ax.set_xticks([0, len(stats_ww.p_error_history) - 1])
        ax.set_xticklabels([
            1, stats_ww.eval_interval * len(stats_ww.p_error_history)])
        sns.despine(ax=ax, trim=True)

    fig.tight_layout()
    if not os.path.exists('./plots/'):
        os.makedirs('./plots/')
    filename = './plots/eval.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('saved to {}'.format(filename))


def main():
    plot_losses()
    plot_eval()


if __name__ == '__main__':
    main()
