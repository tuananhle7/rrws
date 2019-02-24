import util
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def plot_losses():
    stats_ww = util.load_object('./trained_models/ww/stats.pkl')
    stats_vimco = util.load_object('./trained_models/vimco/stats.pkl')

    fig, axs = plt.subplots(1, 4, dpi=100, figsize=(12, 3))
    axs[0].plot(stats_ww.elbo_history, color='C0', label='ww')
    axs[0].plot(stats_vimco.elbo_history, color='C1', label='vimco')
    axs[0].set_ylim(-20)
    axs[0].set_ylabel(r'ELBO')
    axs[0].legend()

    axs[1].plot(stats_ww.wake_phi_loss_history, color='C0')
    axs[1].set_ylabel(r'wake phi loss')

    axs[2].plot(stats_ww.wake_theta_loss_history, color='C0')
    axs[2].set_ylabel(r'wake theta loss')

    axs[3].plot(stats_vimco.loss_history, color='C1')
    axs[3].set_ylabel(r'vimco loss')

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
    stats_ww = util.load_object('./trained_models/ww/stats.pkl')
    stats_vimco = util.load_object('./trained_models/vimco/stats.pkl')

    fig, axs = plt.subplots(1, 3, dpi=100, figsize=(12, 3))
    axs[0].plot(stats_ww.p_error_history, label='ww')
    axs[0].plot(stats_vimco.p_error_history, label='vimco')
    axs[0].set_ylabel(r'Error($p_{true}(z, x)$, $p_\theta(z, x)$)')
    axs[0].set_title('generative model error')

    axs[1].plot(stats_ww.q_error_to_model_history, label='ww')
    axs[1].plot(stats_vimco.q_error_to_model_history, label='vimco')
    axs[1].set_ylabel(r'Error($p_\theta(z | x)$, $q_\phi(z | x)$) + const.')
    axs[1].set_title('inference network error')

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
