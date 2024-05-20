
import swyft.lightning as sl
import torch
from torch import nn

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

import numpy as np
import swyft.lightning as sl
from toolz.dicttoolz import valmap
import torch.nn.functional as F

from typing import Optional, Any
import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer
from einops.layers.torch import Rearrange

# Transformer model 
# https://github.com/gzerveas/mvts_transformer/blob/master/src/models/ts_transformer.py

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))

# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))

class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None, is_causal: bool = False) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src

class TSTransformerEncoderClassiregressor(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """

    def __init__(
        self, 
        feat_dim, 
        max_len, 
        d_model, 
        n_heads, 
        num_layers, 
        dim_feedforward, 
        num_classes,
        patch_size=16,
        dropout=0.0, 
        pos_encoding='fixed', 
        activation='gelu', 
        norm='BatchNorm', 
        freeze=False
        ):
        super(TSTransformerEncoderClassiregressor, self).__init__()

        assert (max_len % patch_size) == 0

        num_patches = max_len // patch_size
        patch_dim = feat_dim * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
        )

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.output_layer = self.build_output_module(d_model, num_patches, num_classes)

    def build_output_module(self, d_model, max_len, num_classes):
        output_layer = nn.Linear(d_model * max_len, num_classes)
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
        # add F.log_softmax and use NLLoss
        return output_layer

    def forward(self, X):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        
        inp = self.to_patch_embedding(X)
        inp = inp.permute(1, 0, 2)
        # inp = self.project_inp(inp) * math.sqrt(self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        padding_masks = torch.ones(inp.shape[1], inp.shape[0]).to(torch.bool).to(device)
        
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)       
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity             
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)
        
        # Output
        output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.output_layer(output)  # (batch_size, num_classes)
        
        return output

# Define Inference network

class ViTInferenceNetwork(sl.SwyftModule):
    
    def __init__(
        self, 
        batch_size, 
        learning_rate,
        ):
        super().__init__()
        
        self.batch_size = batch_size
        self.noise_shuffling = True
        self.num_params = 15
        self.marginals = (0,1),
        self.include_noise = True

        self.ViT_t = TSTransformerEncoderClassiregressor(
            feat_dim = 3,
            max_len = 8192,
            num_classes = 16,
            d_model = 128,
            n_heads = 4,
            num_layers = 6,
            dim_feedforward = 1024,
            dropout = 0,
            pos_encoding = 'fixed',
            activation = 'gelu',
            norm = 'BatchNorm', 
            freeze = False,
        )
        
        self.ViT_f = TSTransformerEncoderClassiregressor(
            feat_dim = 6,
            max_len = 4096,
            num_classes = 16,
            d_model = 128,
            n_heads = 4,
            num_layers = 6,
            dim_feedforward = 1024,
            dropout = 0,
            pos_encoding = 'fixed',
            activation = 'gelu',
            norm = 'BatchNorm', 
            freeze = False,
        )
        
        self.logratios_1d = sl.LogRatioEstimator_1dim(
            num_features=32, num_params=int(self.num_params), varnames="z_total"
        )
            
        self.optimizer_init = sl.AdamOptimizerInit(lr=learning_rate)

    def forward(self, A, B):        
                
        if self.include_noise:
                   
            if self.noise_shuffling and A["d_t"].size(0) != 1:
                noise_shuffling = torch.randperm(self.batch_size)
                d_t = A["d_t"] + A["n_t"][noise_shuffling]
                d_f_w = A["d_f_w"] + A["n_f_w"][noise_shuffling]
            else:
                d_t = A["d_t"] + A["n_t"]
                d_f_w = A["d_f_w"] + A["n_f_w"]
        
        else:
            d_t = A["d_t"]
            d_f_w = A["d_f_w"]
        
        z_total = B["z_total"]

        features_t = self.ViT_t(d_t)
        features_f = self.ViT_f(d_f_w[:,:,:-1])
        
        features = torch.cat([features_t, features_f], dim=1)
        
        logratios_1d = self.logratios_1d(features, z_total)
        
        return logratios_1d

    def _calc_loss_per_feature(self, batch, randomized=True):
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
        loss_tot = torch.zeros(15).to(self.device)

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

            loss = loss.sum(axis=0) / num_neg  # Calculates batched-averaged loss
            loss = loss - 2 * np.log(2.0)
            loss_tot += loss

        return loss_tot

    def training_step(self, batch, batch_idx):
        loss = self._calc_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        
        loss_per_feature = self._calc_loss_per_feature(batch)
        for i in range(15):
            self.log(f"train_loss_feature_{i}", loss_per_feature[i], on_step=True, on_epoch=False)
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calc_loss(batch, randomized=False)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        loss_per_feature = self._calc_loss_per_feature(batch)
        for i in range(15):
            self.log(f"val_loss_feature_{i}", loss_per_feature[i], on_step=False, on_epoch=True)
        
        return loss