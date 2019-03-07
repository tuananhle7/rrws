import os
import torch
import util
import numpy as np
import losses
import matplotlib.pyplot as plt
import seaborn as sns


class OnlineMeanStd():
    def __init__(self):
        self.count = 0
        self.means = None
        self.M2s = None

    def update(self, new_variables):
        if self.count == 0:
            self.count = 1
            self.means = []
            self.M2s = []
            for new_var in new_variables:
                self.means.append(new_var.data)
                self.M2s.append(new_var.data.new(new_var.size()).fill_(0))
        else:
            self.count = self.count + 1
            for new_var_idx, new_var in enumerate(new_variables):
                delta = new_var.data - self.means[new_var_idx]
                self.means[new_var_idx] = self.means[new_var_idx] + delta / \
                    self.count
                delta_2 = new_var.data - self.means[new_var_idx]
                self.M2s[new_var_idx] = self.M2s[new_var_idx] + delta * delta_2

    def means_stds(self):
        if self.count < 2:
            raise ArithmeticError('Need more than 1 value. Have {}'.format(
                self.count))
        else:
            stds = []
            for i in range(len(self.means)):
                stds.append(torch.sqrt(self.M2s[i] / self.count))
            return self.means, stds

    def avg_of_means_stds(self):
        means, stds = self.means_stds()
        num_parameters = np.sum([len(p) for p in means])
        return (np.sum([torch.sum(p) for p in means]) / num_parameters,
                np.sum([torch.sum(p) for p in stds]) / num_parameters)


def get_mean_stds_new(generative_model, inference_network, num_mc_samples,
                      obss, num_particles):
    vimco_grad = OnlineMeanStd()
    vimco_one_grad = OnlineMeanStd()
    reinforce_grad = OnlineMeanStd()
    reinforce_one_grad = OnlineMeanStd()
    two_grad = OnlineMeanStd()
    log_evidence_stats = OnlineMeanStd()
    log_evidence_grad = OnlineMeanStd()
    wake_phi_loss_grad = OnlineMeanStd()
    log_Q_grad = OnlineMeanStd()
    sleep_loss_grad = OnlineMeanStd()

    for mc_sample_idx in range(num_mc_samples):
        util.print_with_time('MC sample {}'.format(mc_sample_idx))
        log_weight, log_q = losses.get_log_weight_and_log_q(
            generative_model, inference_network, obss, num_particles)
        log_evidence = torch.logsumexp(log_weight, dim=1) - \
            np.log(num_particles)
        avg_log_evidence = torch.mean(log_evidence)
        log_Q = torch.sum(log_q, dim=1)
        avg_log_Q = torch.mean(log_Q)
        reinforce_one = torch.mean(log_evidence.detach() * log_Q)
        reinforce = reinforce_one + avg_log_evidence
        vimco_one = 0
        for i in range(num_particles):
            log_weight_ = log_weight[:, util.range_except(num_particles, i)]
            control_variate = torch.logsumexp(
                torch.cat([log_weight_, torch.mean(log_weight_, dim=1,
                                                   keepdim=True)], dim=1),
                dim=1)
            vimco_one = vimco_one + (log_evidence.detach() -
                                     control_variate.detach()) * log_q[:, i]
        vimco_one = torch.mean(vimco_one)
        vimco = vimco_one + avg_log_evidence
        normalized_weight = util.exponentiate_and_normalize(log_weight, dim=1)
        wake_phi_loss = torch.mean(
            -torch.sum(normalized_weight.detach() * log_q, dim=1))

        inference_network.zero_grad()
        generative_model.zero_grad()
        vimco.backward(retain_graph=True)
        vimco_grad.update([param.grad for param in
                           inference_network.parameters()])

        inference_network.zero_grad()
        generative_model.zero_grad()
        vimco_one.backward(retain_graph=True)
        vimco_one_grad.update([param.grad for param in
                               inference_network.parameters()])

        inference_network.zero_grad()
        generative_model.zero_grad()
        reinforce.backward(retain_graph=True)
        reinforce_grad.update([param.grad for param in
                               inference_network.parameters()])

        inference_network.zero_grad()
        generative_model.zero_grad()
        reinforce_one.backward(retain_graph=True)
        reinforce_one_grad.update([param.grad for param in
                                   inference_network.parameters()])

        inference_network.zero_grad()
        generative_model.zero_grad()
        avg_log_evidence.backward(retain_graph=True)
        two_grad.update([param.grad for param in
                         inference_network.parameters()])
        log_evidence_grad.update([param.grad for param in
                                  generative_model.parameters()])

        inference_network.zero_grad()
        generative_model.zero_grad()
        wake_phi_loss.backward(retain_graph=True)
        wake_phi_loss_grad.update([param.grad for param in
                                   inference_network.parameters()])

        inference_network.zero_grad()
        generative_model.zero_grad()
        avg_log_Q.backward(retain_graph=True)
        log_Q_grad.update([param.grad for param in
                           inference_network.parameters()])

        log_evidence_stats.update([avg_log_evidence.unsqueeze(0)])

        sleep_loss = losses.get_sleep_loss(
            generative_model, inference_network, num_particles * len(obss))
        inference_network.zero_grad()
        generative_model.zero_grad()
        sleep_loss.backward()
        sleep_loss_grad.update([param.grad for param in
                                inference_network.parameters()])

    return list(map(
        lambda x: x.avg_of_means_stds(),
        [vimco_grad, vimco_one_grad, reinforce_grad, reinforce_one_grad,
         two_grad, log_evidence_stats, log_evidence_grad, wake_phi_loss_grad,
         log_Q_grad, sleep_loss_grad]))


def get_mean_stds(generative_model, inference_network, num_mc_samples,
                  obss, num_particles, train_mode):
    p_grad_mean_std = OnlineMeanStd()
    q_grad_mean_std = OnlineMeanStd()
    log_evidence_mean_std = OnlineMeanStd()
    for mc_sample_idx in range(num_mc_samples):
        util.print_with_time('MC sample {}'.format(mc_sample_idx))
        inference_network.zero_grad()
        generative_model.zero_grad()
        if train_mode == 'vimco':
            loss, elbo = losses.get_vimco_loss(
                generative_model, inference_network, obss, num_particles)
            log_evidence = elbo
        elif train_mode == 'reinforce':
            loss, elbo = losses.get_reinforce_loss(
                generative_model, inference_network, obss, num_particles)
            log_evidence = elbo
        elif train_mode == 'ww':
            log_weight, log_q = losses.get_log_weight_and_log_q(
                generative_model, inference_network, obss, num_particles)
            loss = losses.get_wake_phi_loss_from_log_weight_and_log_q(
                log_weight, log_q)
            log_evidence = torch.mean(
                torch.logsumexp(log_weight, dim=1) - np.log(num_particles))
        elif train_mode == 'ws':
            loss = losses.get_sleep_loss(
                generative_model, inference_network, num_particles * len(obss))
            log_evidence = torch.tensor(0, dtype=torch.float)
        loss.backward()
        q_grad_mean_std.update([param.grad for param in
                                inference_network.parameters()])
        p_grad_mean_std.update([param.grad for param in
                                generative_model.parameters()])
        log_evidence_mean_std.update([log_evidence.unsqueeze(0)])

    return list(map(
        lambda x: x.avg_of_means_stds(),
        [p_grad_mean_std, q_grad_mean_std, log_evidence_mean_std]))


def old():
    num_iterations = 2000
    logging_interval = 10
    eval_interval = 10
    checkpoint_interval = 100
    batch_size = 2
    pcfg_path = './pcfgs/astronomers_pcfg.json'
    seed = 1
    train_mode = 'vimco'
    num_particles = 50
    exp_levenshtein = True

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
    stats = util.load_object(util.get_stats_filename(model_folder))
    args = util.load_object(util.get_args_filename(model_folder))
    generative_model, inference_network = util.load_models(model_folder)
    _, _, true_generative_model = util.init_models(args.pcfg_path)
    obss = [true_generative_model.sample_obs() for _ in range(args.batch_size)]

    num_mc_samples = 100
    num_particles_list = [2, 5, 10, 20, 50]
    train_mode_list = ['reinforce', 'ws', 'vimco', 'ww']

    p_grad_stats = np.zeros((len(num_particles_list), len(train_mode_list), 2))
    q_grad_stats = np.zeros((len(num_particles_list), len(train_mode_list), 2))
    log_evidence_stats = np.zeros(
        (len(num_particles_list), len(train_mode_list), 2))

    for num_particles_idx, num_particles in enumerate(num_particles_list):
        for train_mode_idx, train_mode in enumerate(train_mode_list):
            util.print_with_time('num_particles = {}, train_mode = {}'.format(
                num_particles, train_mode))
            (p_grad_stats[num_particles_idx, train_mode_idx],
             q_grad_stats[num_particles_idx, train_mode_idx],
             log_evidence_stats[num_particles_idx, train_mode_idx]) = \
                get_mean_stds(generative_model, inference_network,
                              num_mc_samples, obss, num_particles, train_mode)

    util.save_object([p_grad_stats, q_grad_stats, log_evidence_stats],
                     util.get_variance_analysis_filename())


def new_():
    num_iterations = 2000
    logging_interval = 10
    eval_interval = 10
    checkpoint_interval = 100
    batch_size = 2
    pcfg_path = './pcfgs/astronomers_pcfg.json'
    seed = 1
    train_mode = 'vimco'
    num_particles = 50
    exp_levenshtein = True

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
    stats = util.load_object(util.get_stats_filename(model_folder))
    args = util.load_object(util.get_args_filename(model_folder))
    generative_model, inference_network = util.load_models(model_folder)
    _, _, true_generative_model = util.init_models(args.pcfg_path)
    obss = [true_generative_model.sample_obs() for _ in range(args.batch_size)]

    num_mc_samples = 2 # 100
    num_particles_list = [2, 3, 4]  ## [2, 5, 10, 20, 50]

    vimco_grad = np.zeros((len(num_particles_list), 2))
    vimco_one_grad = np.zeros((len(num_particles_list), 2))
    reinforce_grad = np.zeros((len(num_particles_list), 2))
    reinforce_one_grad = np.zeros((len(num_particles_list), 2))
    two_grad = np.zeros((len(num_particles_list), 2))
    log_evidence_stats = np.zeros((len(num_particles_list), 2))
    log_evidence_grad = np.zeros((len(num_particles_list), 2))
    wake_phi_loss_grad = np.zeros((len(num_particles_list), 2))
    log_Q_grad = np.zeros((len(num_particles_list), 2))
    sleep_loss_grad = np.zeros((len(num_particles_list), 2))

    for i, num_particles in enumerate(num_particles_list):
        util.print_with_time('num_particles = {}'.format(num_particles))
        (vimco_grad[i], vimco_one_grad[i], reinforce_grad[i],
         reinforce_one_grad[i], two_grad[i], log_evidence_stats[i],
         log_evidence_grad[i], wake_phi_loss_grad[i], log_Q_grad[i],
         sleep_loss_grad[i]) = get_mean_stds_new(
            generative_model, inference_network, num_mc_samples, obss,
            num_particles)

    util.save_object([
        vimco_grad, vimco_one_grad, reinforce_grad,  reinforce_one_grad,
        two_grad, log_evidence_stats, log_evidence_grad, wake_phi_loss_grad,
        log_Q_grad, sleep_loss_grad],
        './variance_analysis/data_new.pkl')


def plot():
    num_particles_list = [2, 5, 10, 20, 50]
    [vimco_grad, vimco_one_grad, reinforce_grad, reinforce_one_grad,
     two_grad, log_evidence_stats, log_evidence_grad, wake_phi_loss_grad,
     log_Q_grad, sleep_loss_grad] = util.load_object(
        './variance_analysis/data_new.pkl')

    fig, axss = plt.subplots(2, 9, figsize=(18, 4), dpi=100, sharex=True,
                             sharey='row')
    for i, stats in enumerate(
        [vimco_grad, vimco_one_grad, reinforce_grad, reinforce_one_grad,
         two_grad, log_evidence_grad, wake_phi_loss_grad, log_Q_grad,
         sleep_loss_grad]):
        for j in range(2):
            axss[j, i].plot(stats[:, j], color='black')

    axss[0, 0].set_ylabel('mean')
    axss[1, 0].set_ylabel('std')

    for ax in axss[0]:
        ax.set_yticks([ax.get_yticks()[0], ax.get_yticks()[-1]])

    for ax in axss[1]:
        ax.set_yscale('log')
    #     ax.set_yticks([0, ax.get_yticks()[-1]])
        # ax.set_yticks([ax.get_yticks()[0], ax.get_yticks()[-1]])
        # ax.set_yticks([1e-2, 1e4])
        ax.set_xlabel('K')

    for axs in axss:
        for ax in axs:
            ax.set_xticks(range(len(num_particles_list)))
            ax.set_xticklabels(num_particles_list)
            sns.despine(ax=ax, trim=True)

    for ax, title in zip(axss[0], [
        r'$g_{VIMCO}$', r'$g_{VIMCO}^1$', r'$g_{REINFORCE}$',
        r'$g_{REINFORCE}^1$', r'$g^2$', r'$\nabla_{\theta} \log Z_K$',
        r'$\nabla_{\phi}$ wake-$\phi$ loss', r'$\nabla_{\phi} \log Q$',
        r'$\nabla_{\phi}$ sleep loss'
    ]):
        ax.set_title(title)

    fig.tight_layout()
    if not os.path.exists('./plots/'):
        os.makedirs('./plots/')
    filename = './plots/variance_analysis.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('saved to {}'.format(filename))


def main():
    # old()
    new_()
    # plot()


if __name__ == '__main__':
    main()
