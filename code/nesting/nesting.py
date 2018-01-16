import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import scipy as sp


def lognormexp(values):
    return values - sp.special.logsumexp(values)


def get_posterior_params(obs, prior_mean, prior_std, obs_std):
    posterior_var = 1 / (1 / prior_std**2 + 1 / obs_std**2)
    posterior_std = np.sqrt(posterior_var)
    posterior_mean = posterior_var * (prior_mean / prior_std**2 + obs / obs_std**2)

    return posterior_mean, posterior_std


def get_expectation_of_quadratic(mean, std):
    return std**2 + mean**2


def get_empirical_expectation(obs, prior_mean, prior_std, obs_std, f, num_samples):
    x = np.random.normal(prior_mean, prior_std, size=num_samples)
    log_weights = scipy.stats.norm.logpdf(obs, loc=x, scale=obs_std)
    normalized_weights = np.exp(lognormexp(log_weights))

    return np.sum(normalized_weights * f(x))


def quadratic(x):
    return x**2


def get_expectation_1(prior_std_1, obs_std_1, prior_mean_2, prior_std_2, obs_std_2, obs_1, obs_2):
    expectation_2 = get_expectation_of_quadratic(*get_posterior_params(obs_2, prior_mean_2, prior_std_2, obs_std_2))
    return get_expectation_of_quadratic(*get_posterior_params(obs_1, expectation_2, prior_std_1, obs_std_1))


def get_empirical_expectation_1(prior_std_1, obs_std_1, prior_mean_2, prior_std_2, obs_std_2,
                                    obs_1, obs_2,
                                num_samples_1, num_samples_2):
    empirical_expectation_2 = get_empirical_expectation(obs_2, prior_mean_2, prior_std_2, obs_std_2, quadratic, num_samples_2)
    return get_empirical_expectation(obs_1, empirical_expectation_2, prior_std_1, obs_std_1, quadratic, num_samples_1)


def get_empirical_expectation_1_compiled(prior_std_1, obs_std_1, prior_mean_2, prior_std_2, obs_std_2,
                                         obs_1, obs_2,
                                         num_samples_1, num_samples_2):
    posterior_mean_2, posterior_std_2 = get_posterior_params(obs_2, prior_mean_2, prior_std_2, obs_std_2)
    x_2 = np.random.normal(posterior_mean_2, posterior_std_2, size=num_samples_2)

    empirical_expectation_2_compiled = np.mean(quadratic(x_2))
    return get_empirical_expectation(obs_1, empirical_expectation_2_compiled, prior_std_1, obs_std_1, quadratic, num_samples_1)


def main():
    prior_std_1 = 1
    obs_std_1 = 1

    prior_mean_2 = 0
    prior_std_2 = 1
    obs_std_2 = 1

    obs_1 = 7.2
    obs_2 = 4.5

    expectation_1 = get_expectation_1(prior_std_1, obs_std_1, prior_mean_2, prior_std_2, obs_std_2, obs_1, obs_2)

    num_samples_2_list = np.arange(100, 100001, 100)
    num_samples_1 = 100

    empirical_expectation_1_list_1 = np.array([
        get_empirical_expectation_1(
            prior_std_1, obs_std_1, prior_mean_2, prior_std_2, obs_std_2,
            obs_1, obs_2,
            num_samples_1, num_samples_2
        ) for num_samples_2 in num_samples_2_list
    ])

    num_samples_1_list = np.arange(100, 100001, 100)
    num_samples_2 = 100

    empirical_expectation_1_list_2 = np.array([
        get_empirical_expectation_1(
            prior_std_1, obs_std_1, prior_mean_2, prior_std_2, obs_std_2,
            obs_1, obs_2,
            num_samples_1, num_samples_2
        ) for num_samples_1 in num_samples_1_list
    ])

    ##############

    num_samples_2_list = np.arange(100, 100001, 100)
    num_samples_1 = 100

    empirical_expectation_1_list_1_compiled = np.array([
        get_empirical_expectation_1_compiled(
            prior_std_1, obs_std_1, prior_mean_2, prior_std_2, obs_std_2,
            obs_1, obs_2,
            num_samples_1, num_samples_2
        ) for num_samples_2 in num_samples_2_list
    ])

    num_samples_1_list = np.arange(100, 100001, 100)
    num_samples_2 = 100

    empirical_expectation_1_list_2_compiled = np.array([
        get_empirical_expectation_1_compiled(
            prior_std_1, obs_std_1, prior_mean_2, prior_std_2, obs_std_2,
            obs_1, obs_2,
            num_samples_1, num_samples_2
        ) for num_samples_1 in num_samples_1_list
    ])

    ############

    fig, axs = plt.subplots(1, 4, sharey=True)
    fig.set_size_inches(11, 2.5)

    axs[0].plot(num_samples_2_list, empirical_expectation_1_list_1, color='black', marker='o', linestyle='None', markersize=1, label='empirical', alpha=0.5)
    axs[0].axhline(y=expectation_1, color='black', label='true')
    axs[0].set_title('$N_1 = {}$'.format(num_samples_1))
    axs[0].set_xlabel('$N_2$')


    axs[1].plot(num_samples_1_list, empirical_expectation_1_list_2, color='black', marker='o', linestyle='None', markersize=1, label='empirical', alpha=0.5)
    axs[1].axhline(y=expectation_1, color='black', label='true')
    axs[1].set_title('$N_2 = {}$'.format(num_samples_2))
    axs[1].set_xlabel('$N_1$')

    axs[2].plot(num_samples_2_list, empirical_expectation_1_list_1_compiled, color='black', marker='o', linestyle='None', markersize=1, label='empirical', alpha=0.5)
    axs[2].axhline(y=expectation_1, color='black', label='true')
    axs[2].set_title('$N_1 = {}$ (compiled)'.format(num_samples_1))
    axs[2].set_xlabel('$N_2$')


    axs[3].plot(num_samples_1_list, empirical_expectation_1_list_2_compiled, color='black', marker='o', linestyle='None', markersize=1, label='empirical', alpha=0.5)
    axs[3].axhline(y=expectation_1, color='black', label='true')
    axs[3].set_title('$N_2 = {}$ (compiled)'.format(num_samples_2))
    axs[3].set_xlabel('$N_1$')

    axs[3].legend()
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


    fig.tight_layout()
    filename = 'nesting.pdf'
    fig.savefig(filename)
    print('Saved to {}'.format(filename))

if __name__ == '__main__':
    main()
