"""
This file includes a set of specific artificial neural networks designed on
top of Convolution + Recurrent Neural networks (LSTM).
Ref. https://github.com/wohlert/generative-query-network-pytorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

SCALE=4

class Conv2dLSTMCell(nn.Module):
    """
    2d convolutional long short-term memory (LSTM) cell.
    Functionally equivalent to nn.LSTMCell with the
    difference being that nn.Kinear layers are replaced
    by nn.Conv2D layers.

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param kernel_size: size of image kernel
    :param stride: length of kernel stride
    :param padding: number of pixels to pad with
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv2dLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        kwargs = dict(kernel_size=kernel_size, stride=stride, padding=padding)

        self.forget = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.input  = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.output = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.state  = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, input, states):
        """
        Send input through the cell.

        :param input: input to send through
        :param states: (hidden, cell) pair of internal state
        :return new (hidden, cell) pair
        """
        (hidden, cell) = states

        forget_gate = torch.sigmoid(self.forget(input))
        input_gate  = torch.sigmoid(self.input(input))
        output_gate = torch.sigmoid(self.output(input))
        state_gate  = torch.tanh(self.state(input))

        # Update internal cell state
        cell = forget_gate * cell + input_gate * state_gate
        hidden = output_gate * torch.tanh(cell)

        return hidden, cell


class RecurrentAutoEncoder2D(nn.Module):
    """
    Network similar to a convolutional variational
    autoencoder that refines the generated image
    over a number of iterations.

    :param x_dim: number of channels in input
    :param v_dim: dimensions of viewpoint
    :param r_dim: dimensions of representation
    :param z_dim: latent channels
    :param h_dim: hidden channels in LSTM
    :param L: Number of refinements of density
    """
    def __init__(self, x_dim, v_dim, r_dim, z_dim=64, h_dim=128, L=12):
        super(RecurrentAutoEncoder2D, self).__init__()
        self.L = L
        self.z_dim = z_dim
        self.h_dim = h_dim

        # Core computational units
        self.inference_core = Conv2dLSTMCell(h_dim + x_dim + v_dim + r_dim, h_dim, kernel_size=5, stride=1, padding=2)
        self.generator_core = Conv2dLSTMCell(v_dim + r_dim + z_dim, h_dim, kernel_size=5, stride=1, padding=2)

        # Inference, prior
        self.posterior_density = nn.Conv2d(h_dim, 2*z_dim, kernel_size=5, stride=1, padding=2)
        self.prior_density     = nn.Conv2d(h_dim, 2*z_dim, kernel_size=5, stride=1, padding=2)

        # Generative density
        self.observation_density = nn.Conv2d(h_dim, x_dim, kernel_size=1, stride=1, padding=0)

        # Up/down-sampling primitives
        self.upsample   = nn.ConvTranspose2d(h_dim, h_dim, kernel_size=SCALE, stride=SCALE, padding=0)
        self.downsample = nn.Conv2d(x_dim, x_dim, kernel_size=SCALE, stride=SCALE, padding=0)

    def forward(self, x, v, r):
        """
        Attempt to reconstruct x with corresponding
        viewpoint v and context representation r.

        :param x: image to send through
        :param v: viewpoint of image
        :param r: representation for image
        :return reconstruction of x and kl-divergence
        """
        batch_size, _, h, w = x.size()
        kl = 0

        # Increase dimensions
        v = v.view(batch_size, -1, 1, 1).repeat(1, 1, h//SCALE, w//SCALE)
        if r.size(2) != h//SCALE:
            r = r.repeat(1, 1, h//SCALE, w//SCALE)

        # Reset hidden state
        hidden_g = x.new_zeros((batch_size, self.h_dim, h//SCALE, w//SCALE))
        hidden_i = x.new_zeros((batch_size, self.h_dim, h//SCALE, w//SCALE))

        # Reset cell state
        cell_g = x.new_zeros((batch_size, self.h_dim, h//SCALE, w//SCALE))
        cell_i = x.new_zeros((batch_size, self.h_dim, h//SCALE, w//SCALE))

        u = x.new_zeros((batch_size, self.h_dim, h, w))

        x = self.downsample(x)

        for _ in range(self.L):
            # Prior factor (eta π network)
            o = self.prior_density(hidden_g)
            p_mu, p_std = torch.split(o, self.z_dim, dim=1)
            prior_distribution = Normal(p_mu, F.softplus(p_std))

            # Inference state update
            hidden_i, cell_i = self.inference_core(torch.cat([hidden_g, x, v, r], dim=1), [hidden_i, cell_i])

            # Posterior factor (eta e network)
            o = self.posterior_density(hidden_i)
            q_mu, q_std = torch.split(o, self.z_dim, dim=1)
            posterior_distribution = Normal(q_mu, F.softplus(q_std))

            # Posterior sample
            z = posterior_distribution.rsample()

            # Calculate u
            hidden_g, cell_g = self.generator_core(torch.cat([z, v, r], dim=1), [hidden_g, cell_g])
            u = self.upsample(hidden_g) + u

            # Calculate KL-divergence
            kl += kl_divergence(posterior_distribution, prior_distribution)

        x_mu = self.observation_density(u)

        return torch.sigmoid(x_mu), kl

    def sample(self, x_shape, v, r):
        """
        Sample from the prior distribution to generate
        a new image given a viewpoint and representation

        :param x_shape: (height, width) of image
        :param v: viewpoint
        :param r: representation (context)
        """
        h, w = x_shape
        batch_size = v.size(0)

        # Increase dimensions
        v = v.view(batch_size, -1, 1, 1).repeat(1, 1, h//SCALE, w//SCALE)
        if r.size(2) != h//SCALE:
            r = r.repeat(1, 1, h//SCALE, w//SCALE)

        # Reset hidden and cell state for generator
        hidden_g = v.new_zeros((batch_size, self.h_dim, h//SCALE, w//SCALE))
        cell_g = v.new_zeros((batch_size, self.h_dim, h//SCALE, w//SCALE))

        u = v.new_zeros((batch_size, self.h_dim, h, w))

        for _ in range(self.L):
            o = self.prior_density(hidden_g)
            p_mu, p_log_std = torch.split(o, self.z_dim, dim=1)
            prior_distribution = Normal(p_mu, F.softplus(p_log_std))

            # Prior sample
            z = prior_distribution.sample()

            # Calculate u
            hidden_g, cell_g = self.generator_core(torch.cat([v, r, z], dim=1), [hidden_g, cell_g])
            u = self.upsample(hidden_g) + u

        x_mu = self.observation_density(u)

        return torch.sigmoid(x_mu)
