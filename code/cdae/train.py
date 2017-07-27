import cdae.models as models
import cdae.util as util
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


def train_cdae(
    generative_model,
    generative_network,
    inference_network,
    num_iterations,
    batch_size,
    num_data,
    generative_epochs_per_iteration,
    inference_epochs_per_iteration,
    num_generative_epochs=None,
    num_inference_epochs=None
):
    """
    Trains generative_network and inference_network using CDAE algorithm.
    """

    inference_network_optim = optim.Adam(inference_network.parameters())
    generative_network_optim = optim.Adam(generative_network.parameters())

    if num_generative_epochs is None:
        num_generative_epochs = np.repeat(generative_epochs_per_iteration, num_iterations)
    if num_inference_epochs is None:
        num_inference_epochs = np.repeat(inference_epochs_per_iteration, num_iterations)

    generative_model_dataset = models.GenerativeModelDataset(
        generative_model,
        infinite_data=False,
        num_data=num_data
    )
    generative_model_dataloader = torch.utils.data.DataLoader(
        generative_model_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    generative_network_objective = []
    inference_network_objective = []
    if util.vis is not None:
        util.vis.close()
        generative_network_objective_line = util.vis.line(
            X=np.array([0]),
            Y=np.array([0]),
            opts=dict(
                xlabel='Epoch',
                ylabel='Objective (to maximize)',
                title='Generative network objective'
            )
        )

        inference_network_objective_line = util.vis.line(
            X=np.array([0]),
            Y=np.array([0]),
            opts=dict(
                xlabel='Epoch',
                ylabel='Objective (to minimize)',
                title='Inference network objective'
            )
        )

    for i in range(num_iterations):
        util.logger.info('Iteration {}'.format(i))

        # Step 1
        for epoch in range(num_generative_epochs[i]):
            temp_generative_network_objective = []
            for _, observes in enumerate(generative_model_dataloader):
                latents = inference_network.sample(*observes)

                generative_network_optim.zero_grad()
                logpdf_generative_network = generative_network.forward(
                    *map(Variable, latents),
                    *map(Variable, observes)
                )
                temp_generative_network_objective += logpdf_generative_network.data.numpy().tolist()
                utility = torch.mean(logpdf_generative_network)
                loss = -utility  # we want to maximize
                loss.backward()
                generative_network_optim.step()
            generative_network_objective.append(np.mean(temp_generative_network_objective))
            if util.vis is not None:
                util.vis.line(
                    X=np.arange(len(generative_network_objective)),
                    Y=np.nan_to_num(np.array(generative_network_objective)),
                    update='replace',
                    win=generative_network_objective_line
                )
            util.logger.info(
                'Generative network step | Epoch {0} | Objective {1}'.format(
                    epoch,
                    generative_network_objective[-1]
                )
            )

        # Step 2
        for epoch in range(num_inference_epochs[i]):
            temp_inference_network_objective = []
            for batch_size in util.chunk(num_data, batch_size):
                latents, observes = generative_network.sample(batch_size)

                inference_network_optim.zero_grad()
                logpdf_inference_network = inference_network.forward(
                    *map(Variable, latents),
                    *map(Variable, observes)
                )
                temp_inference_network_objective += (-logpdf_inference_network) \
                    .data.numpy().tolist()
                loss = torch.mean(-logpdf_inference_network)
                loss.backward()
                inference_network_optim.step()
            inference_network_objective.append(np.mean(temp_inference_network_objective))
            if util.vis is not None:
                util.vis.line(
                    X=np.arange(len(inference_network_objective)),
                    Y=np.nan_to_num(np.array(inference_network_objective)),
                    update='replace',
                    win=inference_network_objective_line
                )
            util.logger.info(
                'Inference network step | Epoch {0} | Objective {1}'.format(
                    epoch,
                    inference_network_objective[-1]
                )
            )

    return \
        generative_network_objective,\
        inference_network_objective,\
        num_generative_epochs,\
        num_inference_epochs


def plot_cdae_train(
    generative_network_objective,
    inference_network_objective,
    num_generative_epochs,
    num_inference_epochs,
    filename
):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(10, 4)
    ax[0].plot(
        np.arange(len(generative_network_objective)),
        np.nan_to_num(generative_network_objective)
    )
    ax[0].plot(
        np.cumsum(num_generative_epochs) - 1,
        np.nan_to_num(np.array(generative_network_objective)[np.cumsum(num_generative_epochs) - 1]),
        linestyle='None',
        marker='x',
        markersize=5,
        markeredgewidth=1
    )
    ax[0].set_xlim([0, len(generative_network_objective) - 1])
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Objective (to maximize)')
    ax[0].set_title('Generative network objective')

    ax[1].plot(
        np.arange(len(inference_network_objective)),
        np.nan_to_num(inference_network_objective)
    )
    ax[1].plot(
        np.cumsum(num_inference_epochs) - 1,
        np.nan_to_num(np.array(inference_network_objective)[np.cumsum(num_inference_epochs) - 1]),
        linestyle='None',
        marker='x',
        markersize=5,
        markeredgewidth=1
    )
    ax[1].set_xlim([0, len(inference_network_objective) - 1])
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Objective (to minimize)')
    ax[1].set_title('Inference network objective')

    fig.savefig(filename, bbox_inches='tight')
