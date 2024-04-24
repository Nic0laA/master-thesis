import torch
from torch import nn
from torch.functional import F

import swyft.lightning as sl

from torchvision.models import efficientnet_v2_s

import logging
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)


class InferenceNetwork(sl.SwyftModule):
    def __init__(self, conf):
        super().__init__()
        self.one_d_only = conf["one_d_only"]
        self.batch_size = conf["training_batch_size"]
        self.noise_shuffling = conf["shuffling"]
        self.num_params = len(conf["priors"]["int_priors"].keys()) + len(
            conf["priors"]["ext_priors"].keys()
        )
        self.marginals = conf["marginals"]
        
        # Replace U-Net with LSTM
        self.effnet_t = efficientnet_v2_s()
        self.effnet_f = efficientnet_v2_s()

        self.flatten = nn.Flatten(1)
        self.linear_t = LinearCompression()
        self.linear_f = LinearCompression()

        self.logratios_1d = sl.LogRatioEstimator_1dim(
            num_features=32, num_params=int(self.num_params), varnames="z_total"
        )
        
        if not self.one_d_only:
            self.linear_t_2d = LinearCompression_2d()
            self.linear_f_2d = LinearCompression_2d()
            self.logratios_2d = sl.LogRatioEstimator_Ndim(
                num_features=256, marginals=self.marginals, varnames="z_total"
            )
            
        self.optimizer_init = sl.AdamOptimizerInit(lr=conf["learning_rate"])

    def forward(self, A, B):
        if self.noise_shuffling and A["d_t"].size(0) != 1:
            noise_shuffling = torch.randperm(self.batch_size)
            d_t = A["d_t"] + A["n_t"][noise_shuffling]
            d_f_w = A["d_f_w"] + A["n_f_w"][noise_shuffling]
        else:
            d_t = A["d_t"] + A["n_t"]
            d_f_w = A["d_f_w"] + A["n_f_w"]
        z_total = B["z_total"]

        # Pass data through LSTM instead of U-Net
        d_t = self.lstm_t(d_t)
        d_f_w = self.lstm_f(d_f_w)

        features_t = self.linear_t(self.flatten(d_t))
        features_f = self.linear_f(self.flatten(d_f_w))
        features = torch.cat([features_t, features_f], dim=1)
        logratios_1d = self.logratios_1d(features, z_total)
        return logratios_1d
    
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x):
        return self.conv(x)
    
class LSTMNet(nn.Module):
    
    def __init__(
        self, 
        input_dim, 
        hidden_dim, 
        output_dim
    ):
        super(LSTMNet, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        self.outc = OutConv(16, output_dim)

    def forward(self, x):
        
        # LSTM processing
        x, _ = self.lstm(x)
        # Reshape for ConvTranspose1d
        x = x.contiguous().view(-1, 2 * self.hidden_dim, self.seq_len)
        x = self.up_sample(x)
        # Further reshape if necessary to match the desired output dimensions
        x = F.interpolate(x, size=self.seq_len)  # Adjust size if necessary
        x = self.final_conv(x)
        # Match the output shape to U-Net's output
        x = x.view(-1, self.seq_len, 1)  # Adjust the shape to [batch, seq_len, output_dim]
        
        
        
        
        f = self.outc(x)
        return f

    

class LinearCompression(nn.Module):
    def __init__(self):
        super(LinearCompression, self).__init__()
        self.sequential = nn.Sequential(
            nn.LazyLinear(1024),
            nn.ReLU(),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(16),
        )

    def forward(self, x):
        return self.sequential(x)


class LinearCompression_2d(nn.Module):
    def __init__(self):
        super(LinearCompression_2d, self).__init__()
        self.sequential = nn.Sequential(
            nn.LazyLinear(1024),
            nn.ReLU(),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.LazyLinear(128),
        )

    def forward(self, x):
        return self.sequential(x)
    
    