# Hidden markov model

- `model.csv` contains the model specification in the form of

```
<initial_probability_1>,...,<initial_probability_{num_states}>
<transition_probability_1_1>,...,<transition_probability_1_{num_states}>
...
<transition_probability_{num_states}_1>,...,<transition_probability_{num_states}_{num_states}>
<emission_mean_1>,...,<emission_mean_{num_states}>
<emission_variance_1>,...,<emission_variance_{num_states}>
```
where `<transition_probability_j_k>` is the probability of transitioning from state `j` to state `k`.
