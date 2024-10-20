#!/usr/bin/env python

import argparse
import os
import random
import sys
from dataclasses import dataclass


import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

THIS_DIR = os.path.dirname(os.path.abspath("__file__"))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath("__file__")))
sys.path.append(PARENT_DIR)

import data_gen
import modules_Stable as module
import plots_helper as plot
import utils
from integrator import Mid_point


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("No GPU found!")
else:
    print("Great, a GPU is there")
print("=" * 50)


# For reproducibility
utils.reproducibility_seed(seed=42)

# Plotting setting
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

plt.rc("font", size=20)  # controls default text size
plt.rc("axes", titlesize=20)  # fontsize of the title
plt.rc("axes", labelsize=20)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=20)  # fontsize of the x tick labels
plt.rc("ytick", labelsize=20)  # fontsize of the y tick labels
plt.rc("legend", fontsize=15)  # fontsize of the legend

# Define the parameters
@dataclass
class Parameters:
    """It contain necessary parameters for this example."""

    t_train = 10  # Training final time
    t_test = 50  # Testing final time
    Nt = 100  # Number initial conditions of training samples
    Ntest = 25  # Number initial conditions of testing samples

    canonical_dim = 2  # canonical dimension
    latent_dim = 4  # latent canonical dimensional
    hidden_dim = 8  # number of neurons in a hidden layer
    smpling_intl = [-1.5, 1.5]  # sampling interval
    sample_size = 20  # number of samples in a given training time interval
    batch_size = 64  # batch size
    max_potential = 4  # max potential to generate training data
    lr = 3e-3  # Learning rate
    encoder: str = "MLP"  # type of neural network
    confi_model: str = None  # model configuration
    epoch: int = None  # number of epochs which are externally controlled
    path: str = (
        None  # path where the results will be save and it is also externally controlled
    )


# Prepare learned models for integration
def learned_model(t, x):
    """It yields time-derivative of x at time t.
    It is obtained throught the time-derivative of Hamiltonian function.

    Args:
        t (float): time
        x (float): state variable containing position and momenta.

    Returns:
        float: time-derivative of x
    """
    x = torch.tensor(
        x.reshape(-1, params.latent_dim), dtype=torch.float64, requires_grad=True
    ).to(device)
    y = hnn.vector_field(x)
    y = y.detach()
    return y.cpu().numpy()


def Jacobian_learned_model(x):
    """It gives the Jacobian of the right-hand of the differential equations
    that has canonical structure.

    Args:
        x (float): state variable containing position and momenta.

    Returns:
        float: Jacobian of the right-hand of the differential equation
    """
    x = torch.tensor(
        x.reshape(-1, params.latent_dim), dtype=torch.float64, requires_grad=True
    ).to(device)
    y = hnn.vector_field_jacobian(x)
    return y.detach().cpu().numpy()


if __name__ == "__main__":
    params = Parameters()

    # Taking external inputs

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--confi_model",
        type=str,
        default="cubic",
        choices={"linear", "quad", "cubic", "linear_nostability", "linear_opinf"},
        help="Enforcing model hypothesis",
    )

    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")

    args = parser.parse_args()

    params.confi_model = args.confi_model
    params.epoch = args.epochs

    params.path = "./../Results/LV/" + params.confi_model + "/"

    if not os.path.exists(params.path):
        os.makedirs(params.path)
        print("The new directory is created as " + params.path)

    # Generating data
    dyn_model = data_gen.LVExample()

    # Obtaining the data
    x_train, t_train = data_gen.get_data(params, dyn_model)

    # Create data loaders. Note that the data contains both state and derivative information.
    train_dl = data_gen.data_loader(x_train, t_train, dyn_model.vector_field, params)

    # Define the NN models, optimizer

    models = module.network_models(params)

    optim = torch.optim.Adam(
        [
            {
                "params": models["ae"].parameters(),
                "lr": params.lr,
                "weight_decay": 1e-5,
            },
            {
                "params": models["hnn"].parameters(),
                "lr": params.lr,
                "weight_decay": 1e-4,
            },
        ]
    )

    # Obtain Hamiltonian function, autoencoder, transformation and loss error
    models, err_t = module.train(models, train_dl, optim, params)

    # Extracting autoencoder and hnn (hamiltonian)
    autoencoder, hnn = models["ae"], models["hnn"]

    # Plotting the loss decay
    plot.loss_plot(err_t)
    plt.savefig(params.path + "loss_plot.pdf")

    # Testing the models for different initial conditions

    utils.reproducibility_seed(seed=100)
    color_idx, method_name = utils.define_color_method(params)

    learned_sol = []
    ground_truth_sol = []
    track_err = []

    for i in range(params.Ntest):

        t_span = [0, params.t_test]
        t = np.linspace(t_span[0], t_span[1], 10 * params.Nt)

        # define initial condition
        y0_test = np.random.uniform(params.smpling_intl[0], params.smpling_intl[1], 2)

        while (
            np.abs(dyn_model.potential(y0_test[0], y0_test[1])) > params.max_potential
        ):
            y0_test = np.random.uniform(
                params.smpling_intl[0], params.smpling_intl[1], 2
            )

        y0_test = torch.tensor(y0_test, dtype=torch.float64)
        encoded_initial = autoencoder.encode(y0_test.to(device)).detach().cpu()

        print("Testing the models for initial conditon:", y0_test.numpy())
        print("Encoded initial condition:              ", encoded_initial.numpy())
        print("=" * 50)

        # ground truth solution
        gt_sol = Mid_point(dyn_model.vector_field, dyn_model.jacobian, y0_test, t)
        ground_truth_sol.append(gt_sol)

        # Integrating the learned model
        latent_sol = Mid_point(
            learned_model, Jacobian_learned_model, encoded_initial.numpy(), t
        )
        latent_sol = torch.tensor(latent_sol, dtype=torch.float64).to(device)
        decoded_latent_sol = autoencoder.decode(latent_sol.T).detach().T.cpu().numpy()
        learned_sol.append(decoded_latent_sol)

        plot.plotting_saving_plots(
            t,
            gt_sol,
            decoded_latent_sol,
            color_idx=color_idx,
            method_name=method_name,
            idx=i,
            path=params.path,
        )

        temp_err = gt_sol - decoded_latent_sol
        track_err.append(temp_err)

    learned_sol_stack = np.stack(learned_sol, axis=2)
    ground_truth_sol_stack = np.stack(ground_truth_sol, axis=2)
    # Plotting some trajectories in the phase-space
    plt = plot.phase_flow(
        learned_sol_stack[..., :10],
        ground_truth_sol_stack[..., :10],
        (3.5, 3),
        color_idx,
        method_name,
    )
    plt.ylim(-4, 2)
    plt.xlim(-2, 2.5)
    plt.savefig(params.path + "LV_phase.png", dpi=300)
    plt.savefig(params.path + "LV_phase.pdf")

    savemat(
        params.path + "sol_trajectories.mat",
        {
            "learned_sol": learned_sol_stack,
            "ground_truth_sol": ground_truth_sol_stack,
            "t": t,
            "err": track_err,
        },
    )
