import torch
from torch import nn
from torch.functional import F
from toolz.dicttoolz import valmap

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import swyft.lightning as sl

# from swyft_module import SwyftModule

import logging
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)


class InferenceNetwork(sl.SwyftModule):
    def __init__(self, **conf):
        super().__init__()
        self.one_d_only = conf["one_d_only"]
        self.batch_size = conf["training_batch_size"]
        self.noise_shuffling = conf["shuffling"]
        self.num_params = len(conf["priors"]["int_priors"].keys()) + len(
            conf["priors"]["ext_priors"].keys()
        )
        self.marginals = conf["marginals"]
        self.unet_t = Unet(
            n_in_channels=len(conf["ifo_list"]),
            n_out_channels=1,
            sizes=(16, 32, 64, 128, 256),
            down_sampling=(8, 8, 8, 8),
        )
        self.unet_f = Unet(
            n_in_channels=2 * len(conf["ifo_list"]),
            n_out_channels=1,
            sizes=(16, 32, 64, 128, 256),
            down_sampling=(2, 2, 2, 2),
        )

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
        
        self.save_roc_path = conf["save_path"]

    def forward(self, A, B):
        
        # print ("A")
        # print (type(A))
        # print (A['d_t'].shape)
        # print (A['d_f_w'].shape)
# 
        # print ("B")
        # print (type(B))
        # print (B['d_t'].shape)
        # print (B['d_f_w'].shape)
        
        if self.noise_shuffling and A["d_t"].size(0) != 1:
            noise_shuffling = torch.randperm(self.batch_size)
            d_t = A["d_t"] + A["n_t"][noise_shuffling]
            d_f_w = A["d_f_w"] + A["n_f_w"][noise_shuffling]
        else:
            d_t = A["d_t"] + A["n_t"]
            d_f_w = A["d_f_w"] + A["n_f_w"]
        z_total = B["z_total"]

        # print (d_t.shape)
        # print (d_f_w.shape)

        d_t = self.unet_t(d_t)
        d_f_w = self.unet_f(d_f_w)

        # print (d_t.shape)
        # print (d_f_w.shape)
        
        features_t = self.linear_t(self.flatten(d_t))
        features_f = self.linear_f(self.flatten(d_f_w))
        features = torch.cat([features_t, features_f], dim=1)

        
        logratios_1d = self.logratios_1d(features, z_total)
        
        # print (features)
        # print (z_total)
        # print (logratios_1d)
        
        return logratios_1d
        
    def validation_step(self, batch, batch_idx):
        loss = self._calc_loss(batch, randomized=False)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.save_roc_curve(batch, batch_idx)
        
        return loss
        
    def save_roc_curve(self, batch, batch_idx):
        
        # print (f'Saving roc curve to {self.save_roc_path}')
        
        intrinsic_variables = [
            'mass_ratio', 'chirp_mass', 'theta_jn', 'phase', 'tilt_1', 'tilt_2', 
            'a_1', 'a_2', 'phi_12', 'phi_jl'
        ]
        extrinsic_variables = ['luminosity_distance', 'dec', 'ra', 'psi', 'geocent_time']
        
        if isinstance(
            batch, list
        ):  # multiple dataloaders provided, using second one for contrastive samples
            A = batch[0]
            B = batch[1]
        else:  # only one dataloader provided, using same samples for constrative samples
            A = batch
            B = valmap(lambda z: torch.roll(z, 1, dims=0), A)

        # Concatenate positive samples and negative (contrastive) examples
        x = A
        z = {}
        for key in B:
            z[key] = torch.cat([A[key], B[key]])

        num_pos = len(list(x.values())[0])  # Number of positive examples
        num_neg = len(list(z.values())[0]) - num_pos  # Number of negative examples

        out = self(x, z)  # Evaluate network

        logratios = self._get_logratios(
            out
        )  # Generates concatenated flattened list of all estimated log ratios
        if logratios is not None:
            y = torch.zeros_like(logratios)
            y[:num_pos, ...] = 1
            
        # Use soft-max to convert logratios to probabilities
        probabilities = nn.functional.softmax(logratios, dim=1)
        
        plt.figure()
        for i,name in enumerate(intrinsic_variables):
            
            # Compute ROC curve
            fpr, tpr, thresholds = roc_curve(y.cpu().numpy()[:,i], probabilities.detach().cpu().numpy()[:,i])
            roc_auc = auc(fpr, tpr)  # Calculate area under the curve

            # Plot ROC curve
            plt.plot(fpr, tpr, lw=1, label=f'{name} (area = {roc_auc :0.2f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(bbox_to_anchor=(1.7, 1), loc="upper right")
        plt.savefig(self.save_roc_path + '_int.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure()
        for i, name in enumerate(extrinsic_variables):
            
            # Compute ROC curve
            fpr, tpr, thresholds = roc_curve(y.cpu().numpy()[:,i+10], probabilities.detach().cpu().numpy()[:,i])
            roc_auc = auc(fpr, tpr)  # Calculate area under the curve

            # Plot ROC curve
            plt.plot(fpr, tpr, lw=1, label=f'{name} (area = {roc_auc :0.2f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(bbox_to_anchor=(1.7, 1), loc="upper right")
        plt.savefig(self.save_roc_path + '_ext.png', dpi=300, bbox_inches='tight')
        plt.close()
        
# 1D Unet implementation below
class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        mid_channels=None,
        padding=1,
        bias=False,
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                mid_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                mid_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, down_sampling=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(down_sampling), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super().__init__()
        self.up = nn.ConvTranspose1d(
            in_channels, in_channels // 2, kernel_size=kernel_size, stride=stride
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_signal_length = x2.size()[2] - x1.size()[2]

        x1 = F.pad(
            x1, [diff_signal_length // 2, diff_signal_length - diff_signal_length // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(
        self,
        n_in_channels,
        n_out_channels,
        sizes=(16, 32, 64, 128, 256),
        down_sampling=(2, 2, 2, 2),
    ):
        super(Unet, self).__init__()
        self.inc = DoubleConv(n_in_channels, sizes[0])
        self.down1 = Down(sizes[0], sizes[1], down_sampling[0])
        self.down2 = Down(sizes[1], sizes[2], down_sampling[1])
        self.down3 = Down(sizes[2], sizes[3], down_sampling[2])
        self.down4 = Down(sizes[3], sizes[4], down_sampling[3])
        self.up1 = Up(sizes[4], sizes[3])
        self.up2 = Up(sizes[3], sizes[2])
        self.up3 = Up(sizes[2], sizes[1])
        self.up4 = Up(sizes[1], sizes[0])
        self.outc = OutConv(sizes[0], n_out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
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


