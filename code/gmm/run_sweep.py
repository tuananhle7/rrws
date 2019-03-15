import subprocess


def main():
    train_modes = ['ws', 'ww', 'dww', 'vimco', 'reinforce', 'concrete',
                   'relax']
    num_particles_list = [2, 5, 10, 20]
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for seed in seeds:
        for train_mode in train_modes:
            for num_particles in num_particles_list:
                subprocess.call(
                    'sbatch run.sh {} {} {}'.format(
                        train_mode, num_particles, seed), shell=True)


if __name__ == '__main__':
    main()
