
import torch.nn as nn

#ToDO Fill in the __ values
class SmoothMLPMixer(nn.Module):
    def __init__(self, n_class, config):
        super().__init__()

        self.latent_dim = config.latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, self.latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, self.latent_dim)