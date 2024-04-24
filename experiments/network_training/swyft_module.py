from toolz.dicttoolz import valmap
import numpy as np
import torch
from torch.nn import functional as F

import pytorch_lightning as pl

import yaml

from swyft.lightning.data import *
from swyft.plot.mass import get_empirical_z_score
from swyft.lightning.utils import (
    OptimizerInit,
    AdamOptimizerInit,
    SwyftParameterError,
    _collection_mask,
    _collection_flatten,
)

import scipy
from scipy.ndimage import gaussian_filter1d, gaussian_filter

class SwyftModule(pl.LightningModule):
    r"""This is the central Swyft LightningModule for handling the training of logratio estimators.

    Derived classes are supposed to overwrite the `forward` method in order to implement specific inference tasks.

    The attribute `optimizer_init` points to the optimizer initializer (default is `AdamOptimizerInit`).

    .. note::

       The forward method takes as arguments the sample batches `A` and `B`,
       which typically include all sample variables.  Joined samples correspond to
       A=B, whereas marginal samples correspond to samples A != B.

    Example usage:

    .. code-block:: python

       class MyNetwork(swyft.SwyftModule):
           def __init__(self):
               self.optimizer_init = AdamOptimizerInit(lr = 1e-4)
               self.mlp = swyft.LogRatioEstimator_1dim(4, 4)

           def forward(A, B);
               x = A['x']
               z = A['z']
               logratios = self.mlp(x, z)
               return logratios
    """

    def __init__(self):
        super().__init__()
        self.optimizer_init = AdamOptimizerInit()

    def configure_optimizers(self):
        return self.optimizer_init(self.parameters())

    def _get_logratios(self, out):
        if isinstance(out, dict):
            out = {k: v for k, v in out.items() if k[:4] != "aux_"}
            logratios = torch.cat(
                [val.logratios.flatten(start_dim=1) for val in out.values()], dim=1
            )
        elif isinstance(out, list) or isinstance(out, tuple):
            out = [v for v in out if hasattr(v, "logratios")]
            if out == []:
                return None
            logratios = torch.cat(
                [val.logratios.flatten(start_dim=1) for val in out], dim=1
            )
        elif isinstance(out, swyft.LogRatioSamples):
            logratios = out.logratios.flatten(start_dim=1)
        else:
            logratios = None
        return logratios

    def _calc_loss(self, batch, randomized=True):
        """Calcualte batch-averaged loss summed over ratio estimators.

        Note: The expected loss for an untrained classifier (with f = 0) is
        subtracted.  The initial loss is hence usually close to zero.
        """
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
        loss_tot = 0

        logratios = self._get_logratios(
            out
        )  # Generates concatenated flattened list of all estimated log ratios
        if logratios is not None:
            y = torch.zeros_like(logratios)
            y[:num_pos, ...] = 1
            pos_weight = torch.ones_like(logratios[0]) * num_neg / num_pos
            loss = F.binary_cross_entropy_with_logits(
                logratios, y, reduction="none", pos_weight=pos_weight
            )
            num_ratios = loss.shape[1]
            loss = loss.sum() / num_neg  # Calculates batched-averaged loss
            loss = loss - 2 * np.log(2.0) * num_ratios
            loss_tot += loss

        aux_losses = self._get_aux_losses(out)
        if aux_losses is not None:
            loss_tot += aux_losses.sum()

        return loss_tot

    def _get_aux_losses(self, out):
        flattened_out = _collection_flatten(out)
        filtered_out = [v for v in flattened_out if isinstance(v, swyft.AuxLoss)]
        if len(filtered_out) == 0:
            return None
        else:
            losses = torch.cat([v.loss.unsqueeze(-1) for v in filtered_out], dim=1)
            return losses

    def training_step(self, batch, batch_idx):
        loss = self._calc_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calc_loss(batch, randomized=False)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._calc_loss(batch, randomized=False)
        self.log("test_loss", loss, on_epoch=True, on_step=False)
        return loss

    def predict_step(self, batch, *args, **kwargs):
        A = batch[0]
        B = batch[1]
        return self(A, B)