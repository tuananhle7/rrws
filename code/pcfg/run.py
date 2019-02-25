import util
import train


def get_models(load_model_folder, pcfg_path):
    """Load/initialize models.

    Args:
        load_model_folder: string; if '' then will initialize new models
        pcfg_path: path to a json spec of a pcfg

    Returns: generative_model, inference_network, true_generative_model
    """

    if load_model_folder == '':
        generative_model, inference_network, true_generative_model = \
            util.init_models(args.pcfg_path)
    else:
        _, _, true_generative_model = util.init_models(args.pcfg_path)
        generative_model, inference_network = util.load_models()

    return generative_model, inference_network, true_generative_model


def run(args):
    util.print_with_time(str(args))

    # save args
    model_folder = util.get_model_folder()
    args_filename = util.get_args_filename(model_folder)
    util.save_object(args, args_filename)

    # init models
    util.set_seed(args.seed)
    generative_model, inference_network, true_generative_model = get_models(
        args.load_model_folder, args.pcfg_path)

    # train
    if args.train_mode == 'ww':
        train_callback = train.TrainWakeWakeCallback(
            args.pcfg_path, model_folder, true_generative_model,
            args.logging_interval, args.checkpoint_interval,
            args.eval_interval)
        train.train_wake_wake(generative_model, inference_network,
                              true_generative_model, args.batch_size,
                              args.num_iterations, args.num_particles,
                              train_callback)
    elif args.train_mode == 'ws':
        train_callback = train.TrainWakeSleepCallback(
            args.pcfg_path, model_folder, true_generative_model,
            args.logging_interval, args.checkpoint_interval,
            args.eval_interval)
        train.train_wake_sleep(generative_model, inference_network,
                               true_generative_model, args.batch_size,
                               args.num_iterations, args.num_particles,
                               train_callback)
    elif args.train_mode == 'reinforce' or args.train_mode == 'vimco':
        train_callback = train.TrainIwaeCallback(
            args.pcfg_path, model_folder, true_generative_model,
            args.logging_interval, args.checkpoint_interval,
            args.eval_interval)
        train.train_iwae(args.train_mode, generative_model, inference_network,
                         true_generative_model, args.batch_size,
                         args.num_iterations, args.num_particles,
                         train_callback)

    # save models and stats
    util.save_models(generative_model, inference_network, args.pcfg_path,
                     model_folder)
    stats_filename = util.get_stats_filename(model_folder)
    util.save_object(train_callback, stats_filename)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--load-model-folder', default='',
                        help='if specified, will train loaded model')
    parser.add_argument('--train-mode', default='ww',
                        help='ww, ws, reinforce or vimco')
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
    parser.add_argument('--seed', type=int, default=1, help=' ')
    parser.add_argument('--pcfg-path', default='./pcfgs/astronomers_pcfg.json',
                        help=' ')
    args = parser.parse_args()
    run(args)
