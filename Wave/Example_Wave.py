#!/usr/bin/env python

import argparse
import os
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
from scipy.io import savemat
from torch.utils.data import DataLoader, TensorDataset

THIS_DIR = os.path.dirname(os.path.abspath("__file__"))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath("__file__")))
sys.path.append(PARENT_DIR)

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

# Plotting setting
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

plt.rc("font", size=20)  # controls default text size
plt.rc("axes", titlesize=20)  # fontsize of the title
plt.rc("axes", labelsize=20)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=20)  # fontsize of the x tick labels
plt.rc("ytick", labelsize=20)  # fontsize of the y tick labels
plt.rc("legend", fontsize=15)  # fontsize of the legend

## Define the parameters
@dataclass
class Parameters:
    """It contain necessary parameters for this example."""

    canonical_dim = 3  # canonical dimension
    train_index = 600  # number of pts for training
    latent_dim = 6  # latent canonical dimensional
    hidden_dim = 12  # number of neurons in a hidden layer
    batch_size = 32  # batch size
    lr = 3e-3  # Learning rate
    encoder = "MLP"
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


def plot_singular_values():
    """It is a helper function for plotting singular values."""
    plt.rc("font", size=30)  # controls default text size

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.semilogy(_S / _S[0])
    ax.set(
        ylabel="singular values (rel)", xlabel="$k$", xlim=(-1, 50), ylim=(1e-6, 2e0)
    )
    ax.grid()
    plt.tight_layout()
    fig.savefig(params.path + "svd_plot.png", dpi=300)
    fig.savefig(params.path + "svd_plot.pdf")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    Ssum = np.zeros_like(_S)
    Ssum[0] = _S[0]
    for i in range(len(_S[1:])):
        Ssum[i] = sum(_S[: i + 1])

    # plt.plot(_S)
    ax.plot(list(range(1, len(_S) + 1)), Ssum / sum(_S))
    ax.plot([3, 3], [0.05, 1.0], "g--")
    ax.plot([-1, 50], [sum(_S[:3]) / sum(_S), sum(_S[:3]) / sum(_S)], "g--")

    ax.set(ylabel="energy captured", xlabel="$k$", xlim=(-1, 20), ylim=(0.05, 1.05))
    ax.set_xticks([0, 3, 5, 10, 15, 20])
    ax.set_yticks([0.25, 0.527, 0.75, 1.0])
    ax.grid()
    plt.tight_layout()
    fig.savefig(params.path + "energy_plot.png", dpi=300)
    fig.savefig(params.path + "energy_plot.pdf")


def POD_coeffs_plots():
    """It is a helper function to plot POD coefficients
    (grouth truth as well as learned ones).
    """
    fig1, ax1 = plt.subplots(1, 1, figsize=(5, 3))
    fig2, ax2 = plt.subplots(1, 1, figsize=(5, 3))

    for k in range(data.shape[-1]):
        ax1.plot(t, data[:, k].cpu())
        ax2.plot(t, decoded_latent_sol[:, k].cpu())
        ax1.set_ylim(-4.5, 4.5)
        ax2.set_ylim(-4.5, 4.5)

    ax1.set(xlabel="time", ylabel="POD coeffs")
    ax2.set(xlabel="time", ylabel="POD coeffs")

    fig1.savefig(
        params.path + f"pod_coeffs_ground_truth_{k}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    fig1.savefig(
        params.path + f"pod_coeffs_ground_truth_{k}.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )

    fig2.savefig(
        params.path + f"pod_coeffs_learned_{k}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    fig2.savefig(
        params.path + f"pod_coeffs_learned_{k}.pdf", bbox_inches="tight", pad_inches=0.1
    )


def plot_decoded_solution(save_plots=True):
    """It is a helper function to plot solutions on the full grid.

    Args:
        save_plots (bool, optional): It indicates whether to save plots or not. Defaults to True.
    """

    def mycolorbar(ax):
        fbar = fig.colorbar(
            im,
            ax=ax,
            orientation="horizontal",
            pad=0.3,
            format="%.1e",
            ticks=[
                _min,
                (2 / 3) * _min + (1 / 3) * _max,
                (1 / 3) * _min + (2 / 3) * _max,
                _max,
            ],
        )
        tick_font_size = 13
        fbar.ax.tick_params(labelsize=tick_font_size)

    fig, axes = plt.subplots(1, 4, figsize=(18, 3), sharex=True, sharey=True)

    _min, _max = np.min(q_data.T[:, ::1].numpy()), np.max(q_data.T[:, ::1].numpy())
    im = axes[0].pcolormesh(
        T[:, ::1], X[:, ::1], q_data.T[:, ::1], vmin=_min, vmax=_max
    )

    axes[0].set(title="ground truth", xlabel="time", ylabel="$q$")
    mycolorbar(axes[0])

    _min, _max = np.min(q_pod.T[:, ::1].numpy()), np.max(q_pod.T[:, ::1].numpy())
    im = axes[1].pcolormesh(T[:, ::1], X[:, ::1], q_pod.T[:, ::1], vmin=_min, vmax=_max)

    axes[1].set(title=" linear-decoder", xlabel="time")
    mycolorbar(axes[1])

    _min, _max = np.min(q_rec_quad.T[:, ::1].numpy()), np.max(
        q_rec_quad.T[:, ::1].numpy()
    )
    im = axes[2].pcolormesh(
        T[:, ::1], X[:, ::1], q_rec_quad.T[:, ::1], vmin=_min, vmax=_max
    )

    axes[2].set(title=" quad-decoder", xlabel="time")
    mycolorbar(axes[2])

    _min, _max = np.min(q_rec.T[:, ::1].numpy()), np.max(q_rec.T[:, ::1].numpy())
    im = axes[3].pcolormesh(T[:, ::1], X[:, ::1], q_rec.T[:, ::1], vmin=_min, vmax=_max)

    mycolorbar(axes[3])
    axes[3].set(title=" convo-decoder", xlabel="time")
    plt.subplots_adjust(wspace=0.3, hspace=0)

    if save_plots:
        plt.savefig(
            params.path + f"q_compare_{i}.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )
        # plt.savefig(
        #     params.path + f"q_compare_{i}.pdf", bbox_inches="tight", pad_inches=0.1
        # )

    fig, axes = plt.subplots(1, 4, figsize=(18, 3), sharex=True, sharey=True)

    _min, _max = np.min(p_data.T[:, ::1].numpy()), np.max(p_data.T[:, ::1].numpy())
    im = axes[0].pcolormesh(
        T[:, ::1], X[:, ::1], p_data.T[:, ::1], vmin=_min, vmax=_max
    )

    axes[0].set(title="ground truth", xlabel="time", ylabel="$p$")
    mycolorbar(axes[0])

    _min, _max = np.min(p_pod.T[:, ::1].numpy()), np.max(p_pod.T[:, ::1].numpy())
    im = axes[1].pcolormesh(T[:, ::1], X[:, ::1], p_pod.T[:, ::1], vmin=_min, vmax=_max)

    axes[1].set(title=" linear-decoder", xlabel="time")
    mycolorbar(axes[1])

    _min, _max = np.min(p_rec_quad.T[:, ::1].numpy()), np.max(
        p_rec_quad.T[:, ::1].numpy()
    )
    im = axes[2].pcolormesh(
        T[:, ::1], X[:, ::1], p_rec_quad.T[:, ::1], vmin=_min, vmax=_max
    )

    axes[2].set(title=" quad-decoder", xlabel="time")
    mycolorbar(axes[2])

    _min, _max = np.min(p_rec.T[:, ::1].numpy()), np.max(p_rec.T[:, ::1].numpy())
    im = axes[3].pcolormesh(T[:, ::1], X[:, ::1], p_rec.T[:, ::1], vmin=_min, vmax=_max)

    mycolorbar(axes[3])
    axes[3].set(title=" convo-decoder", xlabel="time")
    plt.subplots_adjust(wspace=0.3, hspace=0)

    if save_plots:
        plt.savefig(
            params.path + f"p_compare_{i}.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )
        # plt.savefig(
        #     params.path + f"p_compare_{i}.pdf", bbox_inches="tight", pad_inches=0.1
        # )


if __name__ == "__main__":
    utils.reproducibility_seed(seed=100)

    params = Parameters()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--confi_model",
        type=str,
        default="linear_opinf",
        choices={"linear", "quad", "cubic", "linear_nostability", "linear_opinf"},
        help="Enforcing model hypothesis",
    )

    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")

    args = parser.parse_args()

    params.confi_model = args.confi_model
    params.epoch = args.epochs

    params.path = "./../Results/Wave/" + params.confi_model + "/"

    if not os.path.exists(params.path):
        os.makedirs(params.path)
        print("The new directory is created as " + params.path)

    color_idx, method_name = utils.define_color_method(params)

    #####################
    ##### Preparing data
    #####################
    data_mat = scipy.io.loadmat("./wave_data_multiple.mat")

    t = data_mat["t"]
    x = data_mat["x"]
    [X, T] = np.meshgrid(x, t)

    freqs = data_mat["freqs"]
    # spliting of the data into training and testing
    idxs = list(np.arange(0, len(freqs[0])))
    testing_idxs = list([2, 5, 7])
    train_idxs = list(set(idxs) - set(testing_idxs))

    data_training = data_mat["z"][train_idxs]
    mean_vec = data_training.mean(axis=(0, 2)).reshape(1, -1, 1)

    data_training = data_training - mean_vec

    dzdt_training = data_mat["dz"][train_idxs]

    data_testing = data_mat["z"][testing_idxs] - mean_vec
    dzdt_testing = data_mat["dz"][testing_idxs]

    Xd = np.hstack(data_training)

    Ns = 256
    _U, _S, _ = np.linalg.svd(np.hstack((Xd[:Ns], Xd[Ns:])), full_matrices=False)

    reduced_order = 2 * params.canonical_dim

    # determining projection matrices
    V_proj = np.block(
        [
            [_U[:, : reduced_order // 2], np.zeros((Ns, reduced_order // 2))],
            [np.zeros((Ns, reduced_order // 2)), _U[:, : reduced_order // 2]],
        ]
    )

    print(f"Energy: {np.sum(_S[:reduced_order//2])/np.sum(_S)}")
    print(
        f"size of V: {V_proj.shape} | r: {reduced_order} | params.canonical_dim: {params.canonical_dim}"
    )

    data_training_red = np.zeros(
        (data_training.shape[0], reduced_order, data_training.shape[2])
    )
    data_training_red_deri = np.zeros(
        (data_training.shape[0], reduced_order, data_training.shape[2])
    )

    data_testing_red = np.zeros(
        (data_testing.shape[0], reduced_order, data_testing.shape[2])
    )
    data_testing_red_deri = np.zeros(
        (data_testing.shape[0], reduced_order, data_testing.shape[2])
    )

    for i in range(data_training.shape[0]):
        data_training_red[i] = (V_proj.T) @ data_training[i]
        data_training_red_deri[i] = (V_proj.T) @ dzdt_training[i]

    for i in range(data_testing.shape[0]):
        data_testing_red[i] = (V_proj.T) @ data_testing[i]
        data_testing_red_deri[i] = (V_proj.T) @ dzdt_testing[i]

    # plot singular values
    plot_singular_values()

    data_training_red_deri_5pt = np.zeros(
        (data_training.shape[0], reduced_order, data_training.shape[2])
    )

    for i in range(data_training.shape[0]):
        for j in range(reduced_order):
            data_training_red_deri_5pt[i][j] = np.array(
                utils.compute_derivative(t[0, 1] - t[0, 0], data_training_red[i, j])
            )

    z_train = (
        torch.tensor(data_training_red[..., : params.train_index])
        .permute(0, 2, 1)
        .reshape(-1, reduced_order)
        .requires_grad_()
    )
    dzdt = (
        torch.tensor(data_training_red_deri_5pt[..., : params.train_index])
        .permute(0, 2, 1)
        .reshape(-1, reduced_order)
    )

    train_ds = TensorDataset((z_train).to(device), (dzdt).to(device))
    train_dl = DataLoader(
        train_ds, batch_size=params.batch_size, shuffle=True, num_workers=0
    )

    ########################################
    ######## Define the NN models, optimizer
    ########################################

    params.canonical_dim = reduced_order
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
                "weight_decay": 1e-5,
            },
        ]
    )

    # Obtain Hamiltonian function, autoencoder, transformation and loss error
    models, err_t = module.train(models, train_dl, optim, params)

    # Plotting the loss decay
    plot.loss_plot(err_t)

    # Extracting autoencoder and hnn (hamiltonian)
    autoencoder, hnn = models["ae"], models["hnn"]

    learned_sols = []
    track_err = []

    # Initial condition
    for i, _ in enumerate(data_testing_red):
        data = torch.tensor(data_testing_red[i]).T

        y0_test = data[i, :]
        y0_test = y0_test.unsqueeze(dim=0)

        initial = autoencoder.encode(y0_test.to(device)).detach().cpu()
        initial = initial.numpy()
        print("Encoded initial condition", initial)

        # Integrating the learned model...
        t = np.linspace(0, 60, 1201).T.squeeze()

        latent_sol = Mid_point(learned_model, Jacobian_learned_model, initial, t)

        latent_sol = torch.tensor(latent_sol, dtype=torch.float64).to(device)

        decoded_latent_sol = autoencoder.decode(latent_sol.T).detach()
        learned_sols.append(decoded_latent_sol)

        POD_coeffs_plots()

        err = torch.log10(((data.cpu() - decoded_latent_sol.cpu()) ** 2).mean())
        track_err.append(err.item())

    #########################################################
    # A comparison of decoders which aim to produce full-domain solutions
    # using the POD coordinates
    #########################################################

    # loading the trained decoders
    decoder = module.Decoder(latent_dim=reduced_order).double().to(device)
    decoder.load_state_dict(torch.load("./Wave_decoder.pkl"))

    decoder_quad = module.DecoderQuad(latent_dim=reduced_order).double().to(device)
    decoder_quad.load_state_dict(torch.load("./Wave_decoder_quad.pkl"))

    mean_vec_tensor = torch.from_numpy(mean_vec[..., 0]).to(device)
    with torch.no_grad():
        for i, (x, xr_learned) in enumerate(
            zip(
                torch.from_numpy(data_testing).permute(0, 2, 1),
                learned_sols,
            )
        ):
            x = x.to(device) + mean_vec_tensor
            x_rec = decoder(xr_learned.to(device)) + mean_vec_tensor
            x_rec_quad = decoder_quad(xr_learned.to(device)) + mean_vec_tensor

            x_pod = (
                xr_learned.to(device) @ torch.tensor(V_proj.T).double().to(device)
                + mean_vec_tensor
            )

            err_pod = ((x - x_pod) ** 2).mean().item()
            err_quad = ((x - x_rec_quad) ** 2).mean().item()
            err_conv = ((x - x_rec) ** 2).mean().item()

            print([err_pod, err_quad, err_conv])

            q_data = x[:, :256].T.cpu()
            p_data = x[:, 256:].T.cpu()
            q_rec = x_rec[:, :256].T.cpu()
            p_rec = x_rec[:, 256:].T.cpu()

            q_rec_quad = x_rec_quad[:, :256].T.cpu()
            p_rec_quad = x_rec_quad[:, 256:].T.cpu()

            q_pod = x_pod[:, :256].T.cpu()
            p_pod = x_pod[:, 256:].T.cpu()

            plot_decoded_solution(save_plots=True)

    savemat(params.path + "err.mat", {"err": track_err})  # save test error

    try:
        L = scipy.io.loadmat("./../Results/Wave/linear/" + "err.mat")["err"]
        Q = scipy.io.loadmat("./../Results/Wave/quad/" + "err.mat")["err"]
        C = scipy.io.loadmat("./../Results/Wave/cubic/" + "err.mat")["err"]
        O = scipy.io.loadmat("./../Results/Wave/linear_opinf/" + "err.mat")["err"]

        fig, ax = plt.subplots(1, 1, figsize=(5, 3))

        ax.plot(
            [0, 1, 2],
            L[0],
            marker="d",
            markersize=10,
            color=colors[1],
            label="s-linear-embs",
        )
        ax.plot(
            [0, 1, 2],
            Q[0],
            "--",
            marker="s",
            markersize=10,
            color=colors[2],
            label="quad-embs",
        )
        ax.plot(
            [0, 1, 2],
            C[0],
            marker="<",
            markersize=10,
            color=colors[3],
            label="s-cubic-embs",
        )
        ax.plot(
            [0, 1, 2],
            O[0],
            marker="o",
            markersize=10,
            color=colors[4],
            label="OpInf-Ham",
        )

        ax.set(ylabel="error (log-scale)", xlabel="test cases")
        labels = ["T1", "T2", "T3"]
        ticks = np.arange(len(labels))
        plt.xticks(ticks, labels)
        plt.legend(
            loc="center", bbox_to_anchor=(1.0, 0.5), fancybox=True, shadow=True, ncol=1
        )

        plt.savefig(
            params.path + "/../" + "testing_err.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.savefig(
            params.path + "/../" + "testing_err.pdf",
            bbox_inches="tight",
            pad_inches=0.1,
        )

    except FileNotFoundError:
        print("Not all experiments runs for wave!")
