import util
import train


def main(args):
    print(args)

    # init models
    if args.load_model:
        _, _, true_generative_model = util.init_models(args.pcfg_path)
        generative_model, inference_network = util.load_models()
    else:
        generative_model, inference_network, true_generative_model = \
            util.init_models(args.pcfg_path)

    # train
    if args.train_mode == 'sleep':
        train_sleep_callback = train.TrainSleepCallback(args.logging_interval)
        train.train_sleep(true_generative_model, inference_network,
                          args.num_samples, args.num_iterations,
                          train_sleep_callback)
        model_folder = './trained_models/sleep'
    elif args.train_mode == 'ww':
        model_folder = './trained_models/ww'
        train_wake_wake_callback = train.TrainWakeWakeCallback(
            args.pcfg_path, model_folder, true_generative_model,
            args.logging_interval, args.checkpoint_interval,
            args.eval_interval)
        train.train_wake_wake(generative_model, inference_network,
                              true_generative_model, args.batch_size,
                              args.num_iterations, args.num_particles,
                              train_wake_wake_callback)
        stats_filename = util.get_stats_filename(model_folder)
        util.save_object(train_wake_wake_callback, stats_filename)
    elif args.train_mode == 'ws':
        model_folder = './trained_models/ws'
        train_wake_sleep_callback = train.TrainWakeSleepCallback(
            args.pcfg_path, model_folder, true_generative_model,
            args.logging_interval, args.checkpoint_interval,
            args.eval_interval)
        train.train_wake_sleep(generative_model, inference_network,
                               true_generative_model, args.batch_size,
                               args.num_iterations, args.num_particles,
                               train_wake_sleep_callback)
        stats_filename = util.get_stats_filename(model_folder)
        util.save_object(train_wake_sleep_callback, stats_filename)
    elif args.train_mode == 'reinforce' or args.train_mode == 'vimco':
        algorithm = args.train_mode
        model_folder = './trained_models/{}'.format(algorithm)
        train_iwae_callback = train.TrainIwaeCallback(
            args.pcfg_path, model_folder, true_generative_model,
            args.logging_interval, args.checkpoint_interval,
            args.eval_interval)
        train.train_iwae(algorithm, generative_model, inference_network,
                         true_generative_model, args.batch_size,
                         args.num_iterations, args.num_particles,
                         train_iwae_callback)
        stats_filename = util.get_stats_filename(model_folder)
        util.save_object(train_iwae_callback, stats_filename)

    # save models
    util.save_models(generative_model, inference_network, args.pcfg_path,
                     model_folder)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--load-model', dest='load_model', action='store_true',
                        help=' ')
    parser.add_argument('--train-mode', default='ww',
                        help=' ')
    parser.add_argument('--num-iterations', type=int, default=2000,
                        help=' ')
    parser.add_argument('--logging-interval', type=int, default=10,
                        help=' ')
    parser.add_argument('--eval-interval', type=int, default=10,
                        help=' ')
    parser.add_argument('--checkpoint-interval', type=int, default=100,
                        help=' ')
    parser.add_argument('--batch-size', type=int, default=5,
                        help=' ')
    parser.add_argument('--num-particles', type=int, default=20,
                        help=' ')
    parser.add_argument('--pcfg-path', default='./pcfgs/astronomers_pcfg.json',
                        help=' ')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='only needed for pure sleep')
    args = parser.parse_args()
    main(args)
