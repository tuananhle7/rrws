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
generative_network_small = quadratic.QuadraticGenerativeNetworkSmall()
generative_network_large = quadratic.QuadraticGenerativeNetworkLarge()
inference_network_forward_dependence = quadratic.QuadraticInferenceNetworkForwardDependence()
inference_network_reverse_dependence = quadratic.QuadraticInferenceNetworkReverseDependence()
inference_network_independent = quadratic.QuadraticInferenceNetworkIndependent()

for generative_network, generative_network_name in zip(
    [generative_network_large, generative_network_small], ['gen_large', 'gen_small']
):
    for inference_network, inference_network_name in zip(
        [
            inference_network_forward_dependence,
            inference_network_reverse_dependence,
            inference_network_independent
        ],
        ['inf_fwd', 'inf_rev', 'inf_ind']
    ):
        filename_prefix = generative_network_name + '_' + inference_network_name + '_'

        # Plot test performance
        quadratic.plot_quadratic_generative_comparison(
            generative_model,
            generative_network,
            num_data=1000,
            filename='./figures/quadratic/{}generative_start.pdf'.format(filename_prefix)
        )

        if generative_network_name == 'gen_small':
            quadratic.plot_quadratic_comparison(
                generative_model,
                generative_network,
                filename='./figures/quadratic/{}quadratic_start.pdf'.format(filename_prefix)
            )

        quadratic.plot_quadratic_inference_comparison(
            generative_model,
            inference_network,
            num_inference_network_samples=1000,
            num_importance_particles=1000,
            y_test=25,
            filename='./figures/quadratic/{}inference_start.pdf'.format(filename_prefix)
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
            filename='./figures/quadratic/{}training.pdf'.format(filename_prefix)
        )

        # Test performance
        quadratic.plot_quadratic_generative_comparison(
            generative_model,
            generative_network,
            num_data=1000,
            filename='./figures/quadratic/{}generative.pdf'.format(filename_prefix)
        )

        if generative_network_name == 'gen_small':
            quadratic.plot_quadratic_comparison(
                generative_model,
                generative_network,
                filename='./figures/quadratic/{}quadratic.pdf'.format(filename_prefix)
            )

        quadratic.plot_quadratic_inference_comparison(
            generative_model,
            inference_network,
            num_inference_network_samples=1000,
            num_importance_particles=1000,
            y_test=25,
            filename='./figures/quadratic/{}inference.pdf'.format(filename_prefix)
        )
