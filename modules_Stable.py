import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import utils as util

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def jacobian(y, x):
    """It aims to compute the derivate of the output with respect to input.

    Args:
        y (float): Output
        x (float): Input

    Returns:
        dy/dx (float): Derivative of output w.r.t. input
    """
    batchsize = x.shape[0]
    dim = y.shape[1]
    res = torch.zeros(x.shape[0], y.shape[1], x.shape[1]).to(x)
    for i in range(dim):
        res[:, i, :] = torch.autograd.grad(
            y[:, i], x, grad_outputs=torch.ones(batchsize).to(x), create_graph=True
        )[0].reshape(res[:, i, :].shape)
    return res


def batch_mtxproduct(y, x):
    """It does batch wise matrix-matrix product. The first dimension is the batch.

    Args:
        y (float): A matrix of size a x b x c
        x (float): A matrix of size a x c x e

    Returns:
        y*x (float): Batch wise product of y and x
    """
    x = x.unsqueeze(dim=-1)
    return torch.einsum("abc,ace->abe", [y, x]).view(y.size(0), y.size(1))


def permutation_tensor(n):
    """Generating symplectic matrix

    Args:
        n (int): dimension of the system

    Returns:
        float (matrix): Symplectic matrix
    """
    J = None
    J = torch.eye(n)
    J = torch.cat([J[n // 2 :], -J[: n // 2]])

    return J


class AutoEncoderLinear(nn.Module):
    """It is meant to have linear encoder and decoder. Precisely, they both are identity.
    It is inhreited from nn.Module.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def encode(self, x):
        """Encoder function

        Args:
            x (float): state

        Returns:
            float: encoded state
        """
        return x

    def decode(self, z):
        """Decoder function

        Args:
            z (float): encoded state

        Returns:
            float: decoded state
        """
        return z

    def forward(self, x):
        """Classical auto-encoder

        Args:
            x (float): state

        Returns:
            float: state via auto-encoder
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat


class AutoEncoderMLP(nn.Module):
    """It is meant to have nonlinear encoder and decoder. Precisely, here we have hardcoded three hidden layers.
    It is inhreited from nn.Module.
    """

    def __init__(self, dim, hidden_dim, latent_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = torch.nn.Linear(hidden_dim, latent_dim)

        self.linear5 = torch.nn.Linear(latent_dim, hidden_dim)
        self.linear6 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear7 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear8 = torch.nn.Linear(hidden_dim, dim)

        self.linear_quad = torch.nn.Linear(latent_dim * 2, dim)

    def encode(self, x):
        """Encoder function

        Args:
            x (float): state

        Returns:
            float: encoded state
        """
        h = self.linear1(x)
        h = h + F.silu(self.linear2(h))
        h = h + F.silu(self.linear3(h))
        return self.linear4(h)

    def decode(self, z):
        """Decoder function

        Args:
            z (float): encoded state

        Returns:
            float: decoded state
        """
        h = self.linear5(z)
        h = h + F.silu(self.linear6(h))
        h = h + F.silu(self.linear7(h))
        return self.linear8(h)

    def forward(self, x):
        """Classical auto-encoder

        Args:
            x (float): state

        Returns:
            float: state via auto-encoder
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat


class HNNCubic(nn.Module):
    """It defines Hamiltonian of the encoded variables.
    We have implemented the quartic Hamiltonian function as follows:
    H(x) = x^TQx + [x; x \otimes x]^T Q_1 [x; x \otimes x]
    As a result, it determine quartic Hamiltonian are bounded from below and radially unbounded.
    """

    def __init__(self, dim):
        """To make an instance of HNN with quartic function.

        Args:
            dim (int): dimensional of the latent (encoded) space
        """
        super().__init__()
        # self.layers = torch.nn.ModuleList()
        self.dim = dim
        self.J = permutation_tensor(dim).double().to(device)  # Symplectic matrix

        self.fc_linear = nn.Linear(dim, dim, bias=True)
        self.fc_quad = nn.Linear(2 * dim, 2 * dim, bias=True)
        self.shift = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x):
        """It determine quartic Hamiltonian are bounded from below and radially unbounded.

        Args:
            x (float): state

        Returns:
            float: Hamiltonian function
        """
        # Following two lines are for x^TQx
        result_linear = self.fc_linear(x)  #
        result_linear = (result_linear**2).sum(dim=-1, keepdims=True)

        # The rest of lines are for [x; x \otimes x]^T Q_1 [x; x \otimes x]
        x2 = (x - self.shift) ** 2
        x_all = torch.cat((x, x2), dim=-1)
        result_quad = self.fc_quad(x_all)
        result_quad = (result_quad**2).sum(dim=-1, keepdims=True)

        return result_quad + result_linear

    def vector_field(self, x):
        """it defines vector field

        Args:
            x (float): state

        Returns:
            float: vector field
        """
        H = self.forward(x)  # Hamiltonian
        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        return dH @ self.J.t()

    def vector_field_jacobian(self, x):
        """it computes jacobian of the right-hand-side of the equation.
        It is needed while solving differential equations using
        Symplectic integrator later during the inference.

        Args:
            x (float): state

        Returns:
            float: Jacobian of J*H(x) wrt x
        """
        vf = self.vector_field(x)
        return jacobian(vf, x)


class HNNQuad(nn.Module):
    """It defines Hamiltonian of the encoded variables.
    We have implemented the quartic Hamiltonian function as follows:
    H(x) = x^TQx + x^TQ1kron(x,x)
    """

    def __init__(self, dim):
        """To make an instance of HNN with quartic function.

        Args:
            dim (int): dimensional of the latent (encoded) space
        """
        super().__init__()
        self.dim = dim
        self.J = permutation_tensor(dim).double().to(device)
        self.fc_quad = nn.Linear(dim**3 + dim**2 + dim, 1)

    def forward(self, x):
        """It gives systems with cubic Hamiltonian.

        Args:
            x (float): state

        Returns:
            float: Hamiltonian function
        """
        x2 = util.kron(x, x)
        x3 = util.kron(x, x2)
        x_all = torch.cat((x, x2, x3), dim=-1)
        return self.fc_quad(x_all)  #

    def vector_field(self, x):
        """it defines vector field

        Args:
            x (float): state

        Returns:
            float: vector field
        """
        H = self.forward(x)  # Hamiltonian
        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        return dH @ self.J.t()

    def vector_field_jacobian(self, x):
        """it computes jacobian of the right-hand-side of the equation.
        It is needed while solving differential equations using
        Symplectic integrator later during the inference.

        Args:
            x (float): state

        Returns:
            float: Jacobian of J*H(x) wrt x
        """
        vf = self.vector_field(x)
        return jacobian(vf, x)


class HNNLinear(torch.nn.Module):
    """It defines Hamiltonian of the encoded variables.
    We have implemented the quadratic Hamiltonian function as follows:
    H(x) = x^TQx
    As a result, it determine quartic Hamiltonian are bounded from below and radially unbounded.
    """

    def __init__(self, dim):
        """To make an instance of HNN with quartic function.

        Args:
            dim (int): dimensional of the latent (encoded) space
        """
        super().__init__()
        self.dim = dim
        self.J = permutation_tensor(dim).double().to(device)  # Symplectic matrix
        self.fc_linear = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        """It determine quartic Hamiltonian are bounded from below and radially unbounded.

        Args:
            x (float): state

        Returns:
            float: Hamiltonian function
        """
        # Following two lines are for x^TQx
        result = self.fc_linear(x)  #
        result = (result**2).sum(dim=-1, keepdims=True)
        return result

    def vector_field(self, x):
        """it defines vector field

        Args:
            x (float): state

        Returns:
            float: vector field
        """
        H = self.forward(x)  # Hamiltonian
        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        return dH @ self.J.t()

    def vector_field_jacobian(self, x):
        """it computes jacobian of the right-hand-side of the equation.
        It is needed while solving differential equations using
        Symplectic integrator later during the inference.

        Args:
            x (float): state

        Returns:
            float: Jacobian of J*H(x) wrt x
        """
        vf = self.vector_field(x)
        return jacobian(vf, x)


class HNNLinearNoStability(nn.Module):
    """It defines Hamiltonian of the encoded variables.
    We have implemented the quadratic Hamiltonian function as follows:
    H(x) = a^Tkron(x,x)
    As a result, it determine quadratic Hamiltonian are not necessarily
      bounded from below and radially unbounded.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.J = permutation_tensor(dim).double().to(device)  # Symplectic matrix
        self.fc = nn.Linear(dim**2 + dim, 1)

    def forward(self, x):
        """It determines systems with quadratic Hamiltonian.

        Args:
            x (float): state

        Returns:
            float: Hamiltonian function
        """
        x2 = util.kron(x, x)
        x_all = torch.cat((x, x2), dim=-1)
        return self.fc(x_all)  #

    def vector_field(self, x):
        """it defines vector field

        Args:
            x (float): state

        Returns:
            float: vector field
        """
        H = self.forward(x)  # Hamiltonian
        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        return dH @ self.J.t()

    def vector_field_jacobian(self, x):
        """it computes jacobian of the right-hand-side of the equation.
        It is needed while solving differential equations using
        Symplectic integrator later during the inference.

        Args:
            x (float): state

        Returns:
            float: Jacobian of J*H(x) wrt x
        """
        vf = self.vector_field(x)
        return jacobian(vf, x)


def network_models(params):
    """It configures type of autoencoder and Hamiltonian function based on a given configuation.

    Args:
        params (dataclass): contains parameters

    Raises:
        ValueError: type of configuration is not found in setting, then it raises an error

    Returns:
        models: models containing autoencoder and Hamiltonian function
    """
    AE_HNN_CONFIG = {
        "linear": (
            AutoEncoderMLP,
            HNNLinear,
            "Nonlinear autoencoder and linear system with gaurantee stability!",
        ),
        "linear_nostability": (
            AutoEncoderMLP,
            HNNLinearNoStability,
            "Nonlinear autoencoder and linear system with NO-gaurantee stability!",
        ),
        "quad": (
            AutoEncoderMLP,
            HNNQuad,
            "Nonlinear autoencoder and quadratic system with NO-gaurantee stability!",
        ),
        "cubic": (
            AutoEncoderMLP,
            HNNCubic,
            "Nonlinear autoencoder and cubic system with gaurantee stability!",
        ),
        "linear_opinf": (
            AutoEncoderLinear,
            HNNLinearNoStability,
            "Linear autoencoder and linear system with NO-gaurantee stability!",
        ),
    }

    if params.confi_model in AE_HNN_CONFIG:
        ae_fun, hnn_fuc, print_str = AE_HNN_CONFIG[params.confi_model]
        print(print_str)
        models = {
            "hnn": hnn_fuc(dim=params.latent_dim).double().to(device),
            "ae": ae_fun(
                dim=params.canonical_dim,
                hidden_dim=params.hidden_dim,
                latent_dim=params.latent_dim,
            )
            .double()
            .to(device),
        }

    else:
        raise ValueError(
            f" '{params.confi_model}' configuration is not found! Kindly provide suitable model configuration."
        )

    return models


def train(models, train_dl, optim, params):
    """It does training to learn vector field for the systems that are canonical

    Args:
        models (nn.module): models containing autoencoder and Hamiltonian networks
        train_dl (dataloder): training data
        optim (optimizer): optmizer to update parameters of neural networks
        params (dataclass): contains necessary parameter e.g., number of epochs

    Returns:
        (model, loss): trained model and loss as training progresses
    """
    models["ae"].train()
    models["hnn"].train()

    scheduler = torch.optim.lr_scheduler.StepLR(
        optim, step_size=1500 * len(train_dl), gamma=0.1
    )

    print("Training begins!")

    mse_loss = nn.MSELoss()

    err_t = []
    for i in range(params.epoch):

        for x, dxdt in train_dl:
            z = models["ae"].encode(x)
            x_hat = models["ae"].decode(z)
            loss_ae = 1.0 * mse_loss(x_hat, x)  # Encoder loss

            dzdx = util.jacobian_bcw(z, x)

            dzdt = batch_mtxproduct(dzdx, dxdt)
            dzdt_hat = models["hnn"].vector_field(z)
            # Hamiltonian neural network loss
            loss_h = mse_loss(dzdt_hat, dzdt)

            ### Extra constraints (symplectic transformation)
            J = util.tensor_J(len(x), params.latent_dim).to(device)
            J_can = util.tensor_J(len(x), params.canonical_dim).to(device)

            dzdxT = torch.transpose(dzdx, 1, 2)
            Can = util.can_tp(dzdxT, J)
            Can = util.can_tp(Can, dzdx)
            loss_can = mse_loss(Can, J_can)

            ###########
            if params.confi_model in {"linear", "linear_opinf"}:
                loss = 1e-1 * loss_ae + loss_h + loss_can
            elif params.confi_model == "linear_nostability":
                loss = 1e-1 * loss_ae + loss_h
            else:
                loss = (
                    1e-1 * loss_ae
                    + loss_h
                    + loss_can
                    + 1e-4 * (models["hnn"].fc_quad.weight).abs().mean()
                )
            loss.backward()
            optim.step()
            scheduler.step()
            optim.zero_grad()
            err_t.append(loss.item())

        if (i + 1) % 200 == 0:
            lr = optim.param_groups[0]["lr"]
            print(
                f"Epoch {i+1}/{params.epoch} | loss_HNN: {loss_h.item():.2e}| loss_can: {loss_can.item():.2e} | loss_AE: {loss_ae.item():.2e} | learning rate: {lr:.2e}"
            )
    return models, err_t


#############################################
### High-dimensional example ################
#############################################
class Decoder(torch.nn.Module):
    """Convolution-based decoder.
    It aims to construct full solutions using a convolution neural network.

    Args:
        latent_dim (int): latent space dimension
    """

    def __init__(self, latent_dim):
        super().__init__()

        # Decoder
        self.linear1 = nn.Linear(in_features=2 * latent_dim, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=512)
        self.deconv1 = nn.ConvTranspose1d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.deconv2 = nn.ConvTranspose1d(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.deconv3 = nn.ConvTranspose1d(
            in_channels=16,
            out_channels=8,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.deconv4 = nn.ConvTranspose1d(
            in_channels=8,
            out_channels=2,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )

        self.upsample = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)

    def decode(self, z):
        #     print('Decoding')
        z2 = z**2
        z_all = torch.cat((z, z2), dim=-1)

        h = self.linear1(z_all)
        h = F.selu_(h)
        h = self.linear2(h)
        h = F.selu_(h)
        h = h.reshape(h.shape[0], 64, 8)
        h = self.upsample(h)
        h = self.deconv1(h)
        h = F.selu_(h)
        h = self.deconv2(h)
        h = F.selu_(h)
        h = self.deconv3(h)
        h = F.selu_(h)
        h = self.deconv4(h)
        h = nn.Flatten()(h)
        return h

    def forward(self, z):
        return self.decode(z)


class DecoderQuad(nn.Module):
    """Quadratic decoder. It aims to construct full solutions using a quadratic ansatz.

    Args:
        latent_dim (int): latent space dimension
    """

    def __init__(self, latent_dim):
        super().__init__()

        # Decoder
        self.linear = nn.Linear(
            in_features=latent_dim + latent_dim**2, out_features=512
        )

    def decode(self, z):
        z2 = util.kron(z, z)
        z_all = torch.cat((z, z2), dim=-1)

        h = self.linear(z_all)
        return h

    def forward(self, z):
        return self.decode(z)


def train_decoder(decoder, optim, train_dl, params, V_proj):
    """Training of decoder that aims to reconstruct spatial solutions using OD coordindates.

    Args:
        decoder (nn.Module): decoder
        optim (optimizer): optimizer
        train_dl (dataloader): training data
        params (dataclass): contains necessary parameters for training
        V_proj (float): POD projection matrix

    Returns:
        decoder, err_t: trained decoder and error as training progresses
    """

    scheduler = torch.optim.lr_scheduler.StepLR(
        optim, step_size=250 * len(train_dl), gamma=0.1
    )

    print("training started")
    mse_loss = nn.MSELoss()

    err_t = []
    for i in range(params.epoch):
        for x, z in train_dl:
            x_hat = decoder(z)  # reconstion using decoder
            x_pod = z @ V_proj.T  # reconstruction using POD matrix
            loss_ae = 0.5 * mse_loss(x_hat, x)  # Encoder loss
            loss_ae += 0.5 * (x_hat - x).abs().mean()

            with torch.no_grad():
                loss_pod = 0.5 * mse_loss(x_pod, x)  # Encoder loss
                loss_pod += 0.5 * (x_pod - x).abs().mean()

            loss = loss_ae
            loss.backward()
            optim.step()
            scheduler.step()
            optim.zero_grad()
            err_t.append(loss.item())

        if (i + 1) % 100 == 0:
            lr = optim.param_groups[0]["lr"]
            print(
                f"Epoch {i+1}/{params.epoch} | loss_AE: {loss_ae.item():.2e} | loss_POD: {loss_pod.item():.2e} | learning rate: {lr:.2e}"
            )
    return decoder, err_t
