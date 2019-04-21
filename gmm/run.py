import torch
import util
import numpy as np
import train
import models


def run(args):
    # set up args
    args.device = None
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    args.num_mixtures = 20
    args.init_mixture_logits = np.array(list(reversed(
        2 * np.arange(args.num_mixtures))))
    args.softmax_multiplier = 0.5
    if args.train_mode == 'concrete':
        args.relaxed_one_hot = True
        args.temperature = 3
    else:
        args.relaxed_one_hot = False
        args.temperature = None
    temp = np.arange(args.num_mixtures) + 5
    true_p_mixture_probs = temp / np.sum(temp)
    args.true_mixture_logits = \
        np.log(true_p_mixture_probs) / args.softmax_multiplier
    util.print_with_time(str(args))

    # save args
    model_folder = util.get_model_folder()
    args_filename = util.get_args_path(model_folder)
    util.save_object(args, args_filename)

    # init models
    util.set_seed(args.seed)
    generative_model, inference_network, true_generative_model = \
        util.init_models(args)
    if args.train_mode == 'relax':
        control_variate = models.ControlVariate(args.num_mixtures)

    # train
    if args.train_mode == 'ws':
        train_callback = train.TrainWakeSleepCallback(
            model_folder, true_generative_model,
            args.batch_size * args.num_particles, args.logging_interval,
            args.checkpoint_interval, args.eval_interval)
        train.train_wake_sleep(generative_model, inference_network,
                               true_generative_model, args.batch_size,
                               args.num_iterations, args.num_particles,
                               train_callback)
    elif args.train_mode == 'ww':
        train_callback = train.TrainWakeWakeCallback(
            model_folder, true_generative_model, args.num_particles,
            args.logging_interval, args.checkpoint_interval,
            args.eval_interval)
        train.train_wake_wake(generative_model, inference_network,
                              true_generative_model, args.batch_size,
                              args.num_iterations, args.num_particles,
                              train_callback)
    elif args.train_mode == 'dww':
        train_callback = train.TrainDefensiveWakeWakeCallback(
            model_folder, true_generative_model, args.num_particles, 0.2,
            args.logging_interval, args.checkpoint_interval,
            args.eval_interval)
        train.train_defensive_wake_wake(
            0.2, generative_model, inference_network, true_generative_model,
            args.batch_size, args.num_iterations, args.num_particles,
            train_callback)
    elif args.train_mode == 'reinforce' or args.train_mode == 'vimco':
        train_callback = train.TrainIwaeCallback(
            model_folder, true_generative_model, args.num_particles,
            args.train_mode, args.logging_interval, args.checkpoint_interval,
            args.eval_interval)
        train.train_iwae(args.train_mode, generative_model, inference_network,
                         true_generative_model, args.batch_size,
                         args.num_iterations, args.num_particles,
                         train_callback)
    elif args.train_mode == 'concrete':
        train_callback = train.TrainConcreteCallback(
            model_folder, true_generative_model, args.num_particles,
            args.num_iterations, args.logging_interval,
            args.checkpoint_interval, args.eval_interval)
        train.train_iwae(args.train_mode, generative_model, inference_network,
                         true_generative_model, args.batch_size,
                         args.num_iterations, args.num_particles,
                         train_callback)
    elif args.train_mode == 'relax':
        train_callback = train.TrainRelaxCallback(
            model_folder, true_generative_model, args.num_particles,
            args.logging_interval, args.checkpoint_interval,
            args.eval_interval)
        train.train_relax(generative_model, inference_network, control_variate,
                          true_generative_model, args.batch_size,
                          args.num_iterations, args.num_particles,
                          train_callback)

    # save models and stats
    util.save_models(generative_model, inference_network, model_folder)
    if args.train_mode == 'relax':
        util.save_control_variate(control_variate, model_folder)
    stats_filename = util.get_stats_path(model_folder)
    util.save_object(train_callback, stats_filename)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train-mode', default='ww',
                        help='dww, ww, ws, reinforce, vimco, concrete or '
                             'relax')
    parser.add_argument('--num-iterations', type=int, default=100000,
                        help=' ')
    parser.add_argument('--logging-interval', type=int, default=1000,
                        help=' ')
    parser.add_argument('--eval-interval', type=int, default=1000,
                        help=' ')
    parser.add_argument('--checkpoint-interval', type=int, default=1000,
                        help=' ')
    parser.add_argument('--batch-size', type=int, default=100,
                        help=' ')
    parser.add_argument('--num-particles', type=int, default=2,
                        help=' ')
    parser.add_argument('--seed', type=int, default=1, help=' ')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    args = parser.parse_args()
    run(args)
