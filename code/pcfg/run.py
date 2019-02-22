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
        train_sleep_callback = train.TrainSleepCallback(
            args.logging_interval)
        train.train_sleep(true_generative_model, inference_network,
                          args.num_samples, args.num_iterations,
                          train_sleep_callback)
    elif args.train_mode == 'ww':
        train_wake_wake_callback = train.TrainWakeWakeCallback(
            args.logging_interval)
        train.train_wake_wake(generative_model, inference_network,
                              true_generative_model, args.batch_size,
                              args.num_iterations, args.num_particles,
                              train_wake_wake_callback)

    # save models
    util.save_models(generative_model, inference_network, args.pcfg_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-model', dest='load_model', action='store_true')
    parser.add_argument('--train-mode', default='ww')
    parser.add_argument('--num-iterations', type=int, default=10)
    parser.add_argument('--logging-interval', type=int, default=2)
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-particles', type=int, default=3)
    parser.add_argument('--pcfg-path', default='./pcfgs/astronomers_pcfg.json')
    args = parser.parse_args()
    main(args)
