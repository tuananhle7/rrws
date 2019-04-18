import subprocess


def main():
    subprocess.call('sbatch variance_analysis.sh', shell=True)


if __name__ == '__main__':
    main()
