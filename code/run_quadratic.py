import argparse
import cdae.util as util
import cdae.models.quadratic as quadratic
import cdae.train as train

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed', help='random seed', default=4, type=int)
parser.add_argument(
    '--device',
    help='selected CUDA device (-1: all, 0: 1st device, 1: 2nd device, etc.)',
    default=-1,
    type=int
)
parser.add_argument('--cuda', help='use CUDA', action='store_true')
parser.add_argument('--visdom', help='use Visdom for visualizations', action='store_true')
parser.add_argument('--num-data', help='number of data points to train', default=1000, type=int)
parser.add_argument(
    '--train-iterations',
    help='number of training iterations',
    default=10,
    type=int
)
parser.add_argument('--train-batch-size', help='batch size', default=1000, type=int)
parser.add_argument('--generative-epochs-per-iteration', help='random seed', default=40, type=int)
parser.add_argument('--inference-epochs-per-iteration', help='random seed', default=40, type=int)
opt = parser.parse_args()

util.init(opt)

# Initialize models
generative_model = quadratic.QuadraticGenerativeModel(a=1, b=0, c=0)
generative_network = quadratic.QuadraticGenerativeNetwork()
inference_network = quadratic.QuadraticInferenceNetwork()

# Plot test performance
quadratic.plot_quadratic_generative_comparison(
    generative_model,
    generative_network,
    num_data=1000,
    filename='./figures/generative_start.pdf'
)

quadratic.plot_quadratic_comparison(
    generative_model,
    generative_network,
    filename='./figures/quadratic_start.pdf'
)

quadratic.plot_quadratic_inference_comparison(
    generative_model,
    inference_network,
    num_inference_network_samples=1000,
    num_importance_particles=1000,
    y_test=25,
    filename='./figures/inference_start.pdf'
)

# Training
generative_network_objective,\
    inference_network_objective,\
    num_generative_epochs,\
    num_inference_epochs = train.train_cdae(
        generative_model,
        generative_network,
        inference_network,
        num_iterations=opt.train_iterations,
        batch_size=opt.train_batch_size,
        num_data=opt.num_data,
        generative_epochs_per_iteration=opt.generative_epochs_per_iteration,
        inference_epochs_per_iteration=opt.inference_epochs_per_iteration
    )
train.plot_cdae_train(
    generative_network_objective,
    inference_network_objective,
    num_generative_epochs,
    num_inference_epochs,
    filename='./figures/training.pdf'
)

# Test performance
quadratic.plot_quadratic_generative_comparison(
    generative_model,
    generative_network,
    num_data=1000,
    filename='./figures/generative.pdf'
)

quadratic.plot_quadratic_comparison(
    generative_model,
    generative_network,
    filename='./figures/quadratic.pdf'
)

quadratic.plot_quadratic_inference_comparison(
    generative_model,
    inference_network,
    num_inference_network_samples=1000,
    num_importance_particles=1000,
    y_test=25,
    filename='./figures/inference.pdf'
)
