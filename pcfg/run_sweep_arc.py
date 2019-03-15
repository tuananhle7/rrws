import subprocess


def main():
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for train_mode in ['ws', 'vimco', 'ww', 'reinforce', 'relax']:
            for num_particles in [2, 5, 10, 20, 50]:
                subprocess.call(
                    'sbatch run_arc.sh {} {} {}'.format(
                        train_mode, num_particles, seed), shell=True)


if __name__ == '__main__':
    main()
