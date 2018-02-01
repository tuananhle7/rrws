import hmmlearn.hmm as hmm
import numpy as np


def read_model(filename):
    model = np.genfromtxt(filename, delimiter=',')
    num_states = np.shape(model)[1]
    initial_probs = model[0]
    transition_probs = model[1:(num_states + 1)]
    obs_means = model[num_states + 1]
    obs_vars = model[num_states + 2]

    return initial_probs, transition_probs, obs_means, obs_vars


def make_hmm(initial_probs, transition_probs, obs_means, obs_vars):
    num_states = len(initial_probs)
    my_hmm = hmm.GaussianHMM(n_components=num_states, covariance_type='full')
    my_hmm.startprob_ = initial_probs
    my_hmm.transmat_ = transition_probs
    my_hmm.means_ = np.reshape(obs_means, (-1, 1))
    my_hmm.covars_ = np.reshape(obs_vars, (-1, 1, 1))

    return my_hmm


def generate_from_prior(
    num_samples,
    num_timesteps,
    initial_probs,
    transition_probs,
    obs_means,
    obs_vars
):
    my_hmm = make_hmm(initial_probs, transition_probs, obs_means, obs_vars)
    latent_states = np.zeros([num_samples, num_timesteps])
    observations = np.zeros([num_samples, num_timesteps])

    for sample_idx in range(num_samples):
        obs, lat = my_hmm.sample(n=num_timesteps)
        latent_states[sample_idx, :] = lat.reshape(-1)
        observations[sample_idx, :] = obs.reshape(-1)

    return latent_states, observations
