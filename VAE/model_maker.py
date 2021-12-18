"""
This is the module where the model is defined. It uses the nn.Module as backbone to create the network structure
"""
# Own modules

# Built in
import math
# Libs
import numpy as np

# Pytorch module
import torch.nn as nn
import torch.nn.functional as F
import torch


class VAE(nn.Module):
    def __init__(self, flags):
        """
        This part is to define the modules involved in the
        :param flags:
        """
        super(VAE, self).__init__()
        self.z_dim = flags.dim_z
        # print("self.z_dim = ", self.z_dim)
        # For Decoder
        self.linears_d = nn.ModuleList([])
        self.bn_linears_d = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear_d[0:-1]):               # Excluding the last one as we need intervals
            self.linears_d.append(nn.Linear(fc_num, flags.linear_d[ind + 1]))
            self.bn_linears_d.append(nn.BatchNorm1d(flags.linear_d[ind + 1]))
        # For Encoder
        self.linears_e = nn.ModuleList([])
        self.bn_linears_e = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear_e[0:-1]):  # Excluding the last one as we need intervals
            self.linears_e.append(nn.Linear(fc_num, flags.linear_e[ind + 1]))
            self.bn_linears_e.append(nn.BatchNorm1d(flags.linear_e[ind + 1]))
        # Re-parameterization
        # self.zmean_layer = nn.Linear(flags.linear_e[-1], self.z_dim)
        # self.z_log_var_layer = nn.Linear(flags.linear_e[-1], self.z_dim)
        # For Spectra Encoder
        self.linears_se = nn.ModuleList([])
        self.bn_linears_se = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear_se[0:-1]):               # Excluding the last one as we need intervals
            self.linears_se.append(nn.Linear(fc_num, flags.linear_se[ind + 1]))
            self.bn_linears_se.append(nn.BatchNorm1d(flags.linear_se[ind + 1]))
        # Conv Layer definitions here
        self.convs_se = nn.ModuleList([])
        in_channel = 1                                                  # Initialize the in_channel number
        for ind, (out_channel, kernel_size, stride) in enumerate(zip(flags.conv_out_channel_se,
                                                                     flags.conv_kernel_size_se,
                                                                     flags.conv_stride_se)):
            if stride == 2:     # We want to double the number
                pad = int(kernel_size/2 - 1)
            elif stride == 1:   # We want to keep the number unchanged
                pad = int((kernel_size - 1)/2)
            else:
                Exception("Now only support stride = 1 or 2, contact Ben")
            self.convs_se.append(nn.Conv1d(in_channel, out_channel, kernel_size,
                                           stride=stride, padding=pad))
            in_channel = out_channel  # Update the out_channel
        if self.convs_se:                       # In case that there is no conv layers
            self.convs_se.append(nn.Conv1d(in_channel, out_channels=1, kernel_size=1, stride=1, padding=0))

    def encoder(self, G, S_enc):
        """
        The forward function which defines how the network is connected
        :param S_enc:  The encoded spectra input
        :param G: Geometry output
        :return: Z_mean, Z_log_var: the re-parameterized mean and variance of the
        """
        out = torch.cat((G, S_enc), dim=-1)  # initialize the out
        # For the linear part
        for ind, (fc, bn) in enumerate(zip(self.linears_e, self.bn_linears_e)):
            if ind != len(self.linears_d) - 1:
                # print("decoder layer", ind)
                # out = F.relu(fc(out))
                out = F.relu(bn(fc(out)))  # ReLU + BN + Linear
            else:
                out = fc(out)
        z_mean, z_log_var = torch.chunk(out, 2, dim=1)
        # z_mean = self.zmean_layer(out)
        # z_log_var = self.z_log_var_layer(out)
        return z_mean, z_log_var

    def reparameterize(self, mu, logvar):
        """
        The reparameterization trick for training a probablistic model
        :param mu: The z_mean vector for mean
        :param logvar:  The z_log_var vector for log of variance
        :return: The combined z value
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, S_enc):
        """
        The forward function which defines how the network is connected
        :param S_enc:  The encoded spectra input
        :return: G: Geometry output
        """
        #print("size of z:", z.size())
        #print("size of S_enc", S_enc.size())
        out = torch.cat((z, S_enc), dim=-1)                                                         # initialize the out
        #print("size of cated out:", out.size())
        # For the linear part
        for ind, (fc, bn) in enumerate(zip(self.linears_d, self.bn_linears_d)):
            # print(out.size())
            if ind != len(self.linears_d) - 1:
                #print("decoder layer", ind)
                # out = F.relu(fc(out))
                out = F.relu(bn(fc(out)))  # ReLU + BN + Linear
            else:
                out = fc(out)
        return out
        #return torch.tanh(out)
        #return torch.sigmoid(out)

    def spectra_encoder(self, S):
        """
        The backward function defines how the backward network is connected
        :param S: The 300-d input spectrum
        :return: S_enc: The n-d output encoded spectrum
        """
        if not self.convs_se:                # If there is no conv_se layer, there is no SE then
            return S
        out = S.unsqueeze(1)
        # For the Conv Layers
        for ind, conv in enumerate(self.convs_se):
            out = conv(out)

        out = out.squeeze()
        # For the linear part
        for ind, (fc, bn) in enumerate(zip(self.linears_se, self.bn_linears_se)):
            out = F.relu(bn(fc(out)))
        S_enc = out
        return S_enc

    def forward(self, G, S):
        """
        The forward training funciton for VAEs
        :param G: The input geometry
        :param S: The input spectra
        :return: The output geometry, the mean and variance (log) of the latent variable
        """
        S_enc = self.spectra_encoder(S)                             # Encode the spectra into smaller dimension
        z_mean, z_log_var = self.encoder(G, S_enc)                  # Encoder the spectra & geometry pair into latent
        if self.training:
            z = self.reparameterize(z_mean, z_log_var)              # reparameterize it into z from mean and var
        else:
            z = z_mean
        G_out = self.decode(z, S_enc)                               # Decode the geometry out
        return G_out, z_mean, z_log_var

    def inference(self, S):
        """
        The inference function
        :param S: Input spectra
        :return: The output geometry
        """
        z = torch.randn([S.size(0), self.z_dim])
        if torch.cuda.is_available():
            z = z.cuda()
        return self.decode(z, self.spectra_encoder(S))






"""
class Decoder(nn.Module):
    def __init__(self, flags):
        super(Decoder, self).__init__()
        ""
        This part is the Decoder model layers definition:
        ""
        # Linear Layer and Batch_norm Layer definitions here
        self.linears_d = nn.ModuleList([])
        self.bn_linears_d = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear_d[0:-1]):               # Excluding the last one as we need intervals
            self.linears_d.append(nn.Linear(fc_num, flags.linear_d[ind + 1]))
            self.bn_linears_d.append(nn.BatchNorm1d(flags.linear_d[ind + 1]))

    def forward(self, z, S_enc):
        ""
        The forward function which defines how the network is connected
        :param S_enc:  The encoded spectra input
        :return: G: Geometry output
        ""
        out = torch.concatenate(z, S_enc)                                                         # initialize the out
        # For the linear part
        for ind, (fc, bn) in enumerate(zip(self.linears_d, self.bn_linears_d)):
            # print(out.size())
            out = F.relu(bn(fc(out)))                                   # ReLU + BN + Linear
        return out


class Encoder(nn.Module):
    def __init__(self, flags):
        super(Encoder, self).__init__()
        ""
        This part is the Decoder model layers definition:
        ""
        # Linear Layer and Batch_norm Layer definitions here
        self.linears_e = nn.ModuleList([])
        self.bn_linears_e = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear_e[0:-1]):  # Excluding the last one as we need intervals
            self.linears_e.append(nn.Linear(fc_num, flags.linear_e[ind + 1]))
            self.bn_linears_e.append(nn.BatchNorm1d(flags.linear_e[ind + 1]))

        # Re-parameterization
        self.zmean_layer = nn.Linear(flags.linear_e[-1], flags.dim_latent_z)
        self.z_log_var_layer = nn.Linear(flags.linear_e[-1], flags.dim_latent_z)

    def forward(self, G, S_enc):
        ""
        The forward function which defines how the network is connected
        :param S_enc:  The encoded spectra input
        :param G: Geometry output
        :return: Z_mean, Z_log_var: the re-parameterized mean and variance of the
        ""
        out = torch.concatenate(G, S_enc)  # initialize the out
        # For the linear part
        for ind, (fc, bn) in enumerate(zip(self.linears_e, self.bn_linears_e)):
            # print(out.size())
            out = F.relu(bn(fc(out)))  # ReLU + BN + Linear
        z_mean = self.zmean_layer(out)
        z_log_var = self.z_log_var_layer(out)
        return z_mean, z_log_var

class SpectraEncoder(nn.Module):
    def __init__(self, flags):
        super(SpectraEncoder, self).__init__()
        ""
        This part if the backward model layers definition:
        ""
        # Linear Layer and Batch_norm Layer definitions here
        self.linears_se = nn.ModuleList([])
        self.bn_linears_se = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear_se[0:-1]):               # Excluding the last one as we need intervals
            self.linears_se.append(nn.Linear(fc_num, flags.linear_se[ind + 1]))
            self.bn_linears_se.append(nn.BatchNorm1d(flags.linear_se[ind + 1]))

        # Conv Layer definitions here
        self.convs_se = nn.ModuleList([])
        in_channel = 1                                                  # Initialize the in_channel number
        for ind, (out_channel, kernel_size, stride) in enumerate(zip(flags.conv_out_channel_se,
                                                                     flags.conv_kernel_size_se,
                                                                     flags.conv_stride_se)):
            if stride == 2:     # We want to double the number
                pad = int(kernel_size/2 - 1)
            elif stride == 1:   # We want to keep the number unchanged
                pad = int((kernel_size - 1)/2)
            else:
                Exception("Now only support stride = 1 or 2, contact Ben")
            self.convs_se.append(nn.Conv1d(in_channel, out_channel, kernel_size,
                                          stride=stride, padding=pad))
            in_channel = out_channel  # Update the out_channel

    def forward(self, S):
        ""
        The backward function defines how the backward network is connected
        :param S: The 300-d input spectrum
        :return: S_enc: The n-d output encoded spectrum
        ""
        out = S.unsqueeze(1)
        # For the Conv Layers
        for ind, conv in enumerate(self.convs_se):
            out = conv(out)

        out = out.squeeze()
        # For the linear part
        for ind, (fc, bn) in enumerate(zip(self.linears_se, self.bn_linears_se)):
            out = F.relu(bn(fc(out)))
        S_enc = out
        return S_enc

"""
