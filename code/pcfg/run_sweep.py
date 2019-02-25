import subprocess


def main():
    num_iterations = 100
    logging_interval = 1
    eval_interval = 1
    checkpoint_interval = 5
    batch_size = 2
    for seed in [1, 2, 3, 4, 5]:
        for train_mode in ['ws', 'vimco', 'ww']:
            for num_particles in [2, 3, 4, 5, 6]:
                subprocess.call(
                    'qsub -F \"{} {} {} {} {} {} {}\" run.sh'.format(
                        train_mode, num_iterations, logging_interval,
                        eval_interval, checkpoint_interval, batch_size,
                        num_particles), shell=True)


if __name__ == '__main__':
    main()
