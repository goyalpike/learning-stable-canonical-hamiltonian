#!/usr/bin/env python

import argparse
import os
import random
import sys
from dataclasses import dataclass

import scipy.io
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

THIS_DIR = os.path.dirname(os.path.abspath("__file__"))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath("__file__")))
sys.path.append(PARENT_DIR)

import modules_Stable as module
import utils


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("No GPU found!")
else:
    print("Great, a GPU is there")
    # get_ipython().system('nvidia-smi')
print("=" * 50)

# For reproducibility
utils.reproducibility_seed(seed=100)


# Define the parameters
@dataclass
class Parameters:
    """It contain necessary parameters for this example."""
    canonical_dim: int = 3
    train_index: int = 500 # taking first 500 points for training and the rest of testing
    batch_size: int = 64
    lr: float = 1e-3  # Learning rate
    path: str = "./../Results/Wave/" + "/" # Path
    epoch: int = None # Number of epochs
    confi_decoder: str = None # configuration of decoder


params = Parameters()

params.path = "./../Results/Wave/" + "/"

if not os.path.exists(params.path):
    os.makedirs(params.path)
    print("The new directory is created as " + params.path)

# Taking external inputs
parser = argparse.ArgumentParser()

parser.add_argument(
    "--confi_decoder",
    type=str,
    default="cnn",
    choices={"quad", "cnn"},
    help="Enforcing decoder hypothesis",
)
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")

args = parser.parse_args()

params.confi_decoder = args.confi_decoder
params.epoch = args.epochs

######################
# Setting up the data
######################
data_mat = scipy.io.loadmat("./wave_data_multiple.mat")

freqs = data_mat["freqs"]

# Splitting of the data into training and testing
idxs = list(np.arange(0, len(freqs[0])))
testing_idxs = list([2, 5, 7])
train_idxs = list(set(idxs) - set(testing_idxs))


data_training = data_mat["z"][train_idxs]
mean_vec = data_training.mean(axis=(0, 2)).reshape(1, -1, 1)
data_training = data_training - mean_vec
data_testing = data_mat["z"][testing_idxs]


Xd = np.hstack(data_training)

_U, _S, _ = np.linalg.svd(np.hstack((Xd[:256], Xd[256:])))

reduced_order = 2 * params.canonical_dim

# We do block projection via co-tangent lifting.
# Note that the projection matrix is only determined using the training data.
V_proj = np.block(
    [
        [_U[:, : reduced_order // 2], np.zeros((256, reduced_order // 2))],
        [np.zeros((256, reduced_order // 2)), _U[:, : reduced_order // 2]],
    ]
)

print(f"Energy: {np.sum(_S[:reduced_order//2])/np.sum(_S)}")
print(f"size of V: {V_proj.shape}")

data_training_red = np.zeros((data_training.shape[0], reduced_order, data_training.shape[2]))
data_training_red_deri = np.zeros((data_training.shape[0], reduced_order, data_training.shape[2]))

data_testing_red = np.zeros((data_testing.shape[0], reduced_order, data_training.shape[2]))
data_testing_red_deri = np.zeros((data_testing.shape[0], reduced_order, data_training.shape[2]))


for i in range(data_training.shape[0]):
    data_training_red[i] = (V_proj.T) @ data_training[i]

for i in range(data_testing.shape[0]):
    data_testing_red[i] = (V_proj.T) @ data_testing[i]

# Defining data loaders
z_train_full = (
    torch.tensor(data_training[..., : params.train_index])
    .permute(0, 2, 1)
    .reshape(-1, 512)
)
z_train_red = (
    torch.tensor(data_training_red[..., : params.train_index])
    .permute(0, 2, 1)
    .reshape(-1, reduced_order)
)

train_ds = TensorDataset(z_train_full.to(device), z_train_red.to(device))
train_dl = DataLoader(
    train_ds, batch_size=params.batch_size, shuffle=True, num_workers=0
)

z_testing_full = torch.tensor(data_testing).permute(0, 2, 1)
z_testing_red = torch.tensor(data_testing_red).permute(0, 2, 1)

test_ds = TensorDataset(z_testing_full.to(device), z_testing_red.to(device))
test_dl = DataLoader(
    test_ds, batch_size=params.batch_size, shuffle=False, num_workers=0
)


#########################################################
########## Model Configuration and trainng ##############
#########################################################

if params.confi_decoder == "quad":
    print("Decoder is quadratic!")
    decoder = (
        module.DecoderQuad(latent_dim=reduced_order).double().to(device)
    )  # Defining dataloader

    optim = torch.optim.Adam(
        [
            {"params": decoder.parameters(), "lr": 1 * params.lr, "weight_decay": 1e-5},
        ]
    )

    decoder, err = module.train_decoder(
        decoder, optim, train_dl, params, torch.tensor(V_proj).double().to(device)
    )

    # Saving decoder
    torch.save(decoder.state_dict(), "./Wave_decoder_quad.pkl")
    decoder.load_state_dict(torch.load("./Wave_decoder_quad.pkl"))

else:
    print("Decoder is based on CNN!")

    decoder = module.Decoder(latent_dim=reduced_order).double().to(device)
    optim = torch.optim.Adam(
        [
            {"params": decoder.parameters(), "lr": 1 * params.lr, "weight_decay": 1e-5},
        ]
    )

    decoder, err = module.train_decoder(
        decoder, optim, train_dl, params, torch.tensor(V_proj).double().to(device)
    )

    torch.save(decoder.state_dict(), "./Wave_decoder.pkl")
    decoder.load_state_dict(torch.load("./Wave_decoder.pkl"))


#########################################################
########## Testing the learned decoder ##################
#########################################################
t = data_mat["t"]
x = data_mat["x"][:, ::1]
[X, T] = np.meshgrid(x, t)

with torch.no_grad():
    for x, z in iter(test_ds):
        x_rec = decoder(z)
        x_pod = z @ torch.tensor(V_proj.T).double().to(device)
        q_data = x[:, :256].T.cpu()
        p_data = x[:, 256:].T.cpu()
        q_rec = x_rec[:, :256].T.cpu()
        p_rec = x_rec[:, 256:].T.cpu()

        q_pod = x_pod[:, :256].T.cpu()
        p_pod = x_pod[:, 256:].T.cpu()

        fig = plt.figure(figsize=(24, 5))
        s = 1
        plt.subplot(231)
        plt.pcolormesh(T[:, ::s], X[:, ::s], q_data.T[:, ::s])
        plt.title("Ground truth")
        plt.colorbar()
        plt.xlabel("$t$")
        plt.ylabel("$x$")

        plt.subplot(232)
        plt.pcolormesh(T[:, ::s], X[:, ::s], q_rec.T[:, ::s])
        plt.title("Reconstructed")
        plt.colorbar()
        plt.xlabel("$t$")
        plt.ylabel("$x$")

        plt.subplot(233)
        plt.pcolormesh(T[:, ::s], X[:, ::s], q_pod.T[:, ::s])
        plt.title("Reconstructed")
        plt.colorbar()
        plt.xlabel("$t$")
        plt.ylabel("$x$")

        plt.subplot(234)
        plt.pcolormesh(T[:, ::s], X[:, ::s], p_data.T[:, ::s])
        plt.title("Ground truth")
        plt.colorbar()
        plt.xlabel("$t$")
        plt.ylabel("$x$")

        plt.subplot(235)
        plt.pcolormesh(T[:, ::s], X[:, ::s], p_rec.T[:, ::s])
        plt.title("Reconstructed")
        plt.colorbar()
        plt.xlabel("$t$")
        plt.ylabel("$x$")

        plt.subplot(236)
        plt.pcolormesh(T[:, ::s], X[:, ::s], p_pod.T[:, ::s])
        plt.title("Reconstructed")
        plt.colorbar()
        plt.xlabel("$t$")
        plt.ylabel("$x$")
        plt.tight_layout()
