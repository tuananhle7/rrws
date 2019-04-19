import os
import torch
import util
import numpy as np
import losses
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def get_mean_stds(generative_model, inference_network, num_mc_samples, obs,
                  num_particles):
    vimco_grad = util.OnlineMeanStd()
    vimco_one_grad = util.OnlineMeanStd()
    reinforce_grad = util.OnlineMeanStd()
    reinforce_one_grad = util.OnlineMeanStd()
    two_grad = util.OnlineMeanStd()
    log_evidence_stats = util.OnlineMeanStd()
    log_evidence_grad = util.OnlineMeanStd()
    wake_phi_loss_grad = util.OnlineMeanStd()
    log_Q_grad = util.OnlineMeanStd()
    sleep_loss_grad = util.OnlineMeanStd()

    for mc_sample_idx in range(num_mc_samples):
        util.print_with_time('MC sample {}'.format(mc_sample_idx))
        log_weight, log_q = losses.get_log_weight_and_log_q(
            generative_model, inference_network, obs, num_particles)
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
            generative_model, inference_network, num_particles * len(obs))
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


def main():
    num_mixtures = 20
    temp = np.arange(num_mixtures) + 5
    true_p_mixture_probs = temp / np.sum(temp)
    softmax_multiplier = 0.5
    args = argparse.Namespace(
        init_mixture_logits=np.array(list(reversed(
            2 * np.arange(num_mixtures)))),
        softmax_multiplier=softmax_multiplier,
        device=torch.device('cpu'),
        num_mixtures=num_mixtures,
        relaxed_one_hot=False,
        temperature=None,
        true_mixture_logits=np.log(true_p_mixture_probs) / softmax_multiplier)
    batch_size = 2
    generative_model, inference_network, true_generative_model = \
        util.init_models(args)
    obs = true_generative_model.sample_obs(batch_size)

    num_mc_samples = 100
    num_particles_list = [2, 5, 10, 20, 50, 100]

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
         sleep_loss_grad[i]) = get_mean_stds(
            generative_model, inference_network, num_mc_samples, obs,
            num_particles)

    util.save_object([
        vimco_grad, vimco_one_grad, reinforce_grad,  reinforce_one_grad,
        two_grad, log_evidence_stats, log_evidence_grad, wake_phi_loss_grad,
        log_Q_grad, sleep_loss_grad],
        './variance_analysis/data.pkl')


if __name__ == '__main__':
    main()
