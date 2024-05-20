#!/usr/bin/env python
# coding: utf-8

# In[99]:


# Load simulation data

import zarr
import swyft.lightning as sl

data_directory = '/scratch-shared/scur2012/peregrine_data/bhardwaj2023'

run_name = 'lowSNR'
rnd_id = 1

data_dir = f'/scratch-shared/scur2012/peregrine_data/bhardwaj2023/simulations_{run_name}_R{rnd_id}'
# simulation_results = zarr.convenience.open(simulation_store_path)

data_dir = '/scratch-shared/scur2012/peregrine_data/tmnre_experiments/peregrine_copy_highSNR_v3/simulations/round_1'


# In[100]:


# Data loader function

def load_data(data_dir, batch_size=64, train_test_split=0.9):
    
    zarr_store = sl.ZarrStore(f"{data_dir}")
    
    train_data = zarr_store.get_dataloader(
        num_workers=8,
        batch_size=batch_size,
        idx_range=[0, int(train_test_split * len(zarr_store.data.z_int))],
        on_after_load_sample=False
    )

    val_data = zarr_store.get_dataloader(
        num_workers=8,
        batch_size=batch_size,
        idx_range=[
            int(train_test_split * len(zarr_store.data.z_int)),
            len(zarr_store.data.z_int) - 1,
        ],
        on_after_load_sample=None
    )
    
    return train_data, val_data


# In[101]:


# Transformer model 
# https://github.com/gzerveas/mvts_transformer/blob/master/src/models/ts_transformer.py

from typing import Optional, Any
import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer
from einops.layers.torch import Rearrange

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


# In[102]:


# Define Inference network
import numpy as np
import swyft.lightning as sl
from toolz.dicttoolz import valmap
import torch.nn.functional as F

class ViTInferenceNetwork(sl.SwyftModule):
    
    def __init__(
        self, 
        batch_size, 
        learning_rate,
        d_model,
        n_heads,
        num_layers,
        dim_feedforward,
        dropout,
        pos_encoding,
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
            d_model = d_model,
            n_heads = n_heads,
            num_layers = num_layers,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            pos_encoding = pos_encoding,
            activation = 'gelu',
            norm = 'BatchNorm', 
            freeze = False,
        )
        
        self.ViT_f = TSTransformerEncoderClassiregressor(
            feat_dim = 6,
            max_len = 4096,
            num_classes = 16,
            d_model = d_model,
            n_heads = n_heads,
            num_layers = num_layers,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            pos_encoding = pos_encoding,
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
    


# In[103]:


# Define training function
import os
import tempfile

import ray
from ray import train, tune
from ray.train import Checkpoint, CheckpointConfig, RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import CombinedStopper,MaximumIterationStopper,TrialPlateauStopper
from ray.tune import Tuner, TuneConfig
from ray.tune.search.hyperopt import HyperOptSearch

import torch.optim as optim

def train_transformer_mvts(config, data_dir=None):
    
    print (config)
    
    net = ViTInferenceNetwork(        
        batch_size = config['batch_size'], 
        learning_rate = config['learning_rate'],
        d_model = config['d_model'],
        n_heads = config['n_heads'], 
        num_layers = config['num_layers'],
        dim_feedforward = config['dim_feedforward'],
        dropout = config['dropout'],
        pos_encoding = config['pos_encoding'],
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=config["learning_rate"])

    checkpoint = train.get_checkpoint()

    if checkpoint:
        
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
            start_epoch = checkpoint_dict["epoch"]
            net.load_state_dict(checkpoint_dict["model_state"])
        
    else:
        start_epoch = 0

    trainloader, valloader = load_data(data_dir, batch_size=config['batch_size'], train_test_split=0.8)

    for epoch in range(start_epoch, config["max_num_epochs"]):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
                
        for i, data in enumerate(trainloader, 0):
            
            batch = {key:data[key].to(device) for key in data}

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            loss = net.training_step(batch, 0)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                
                batch = {key:data[key].to(device) for key in data}

                loss = net.validation_step(batch, 0)
                val_loss += loss.cpu().numpy()
                val_steps += 1


        metrics = {"loss": val_loss / val_steps}
        with tempfile.TemporaryDirectory() as tempdir:
            torch.save(
                {"epoch": epoch, "model_state": net.state_dict()},
                os.path.join(tempdir, "checkpoint.pt"),
            )
            train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir))
        
    print("Finished Training")


# In[104]:


import os
from functools import partial

def main(num_samples=10, max_num_epochs=10):
    
    data_dir = os.path.abspath(f"/scratch-shared/scur2012/peregrine_data/bhardwaj2023/simulations_{run_name}_R{rnd_id}")
        
    config = dict(
        batch_size = tune.choice([32,64]),
        learning_rate = tune.loguniform(1e-5, 1e-3),
        d_model = tune.choice([128]),
        n_heads = tune.choice([4,8]),
        num_layers = tune.choice([6]),
        dim_feedforward = tune.choice([1024]),
        dropout = tune.choice([0]),
        pos_encoding = tune.choice(['fixed','learnable']),
        max_num_epochs = max_num_epochs,
    )
    
    first_guess = [{
        "batch_size": 32,
        "learning_rate": 1e-4,
        "d_model": 128,
        "n_heads": 4,
        "num_layers": 6,
        "dim_feedforward": 1024,
        "dropout": 0,
        "pos_encoding": 'fixed',
    }]

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=2500,
        grace_period=1,
        reduction_factor=2,
        time_attr="time_total_s"
    )
        
    tune_config = TuneConfig(
        max_concurrent_trials=1,
        num_samples=num_samples,
        search_alg=HyperOptSearch(
            points_to_evaluate=first_guess, 
            metric='loss', 
            mode='min', 
            n_initial_points=10),
        scheduler=scheduler,
    )
    
    stopper = CombinedStopper(
        MaximumIterationStopper(max_iter=max_num_epochs),
        TrialPlateauStopper(metric="loss"),
    )
    
    run_config = RunConfig(
        name="transformer_mvts",
        storage_path='/home/scur2012/Thesis/master-thesis/experiments/tuning/ray_results',
        checkpoint_config=CheckpointConfig(checkpoint_score_attribute='loss', checkpoint_score_order='min',num_to_keep=1),
        log_to_file=True,
        stop=stopper,
    )
    
    # Create Tuner
    trainable_with_cpu_gpu = tune.with_resources(partial(train_transformer_mvts, data_dir=data_dir), {"cpu": 18, "gpu": 1})
    tuner = Tuner(
        trainable_with_cpu_gpu,
        # Add some parameters to tune
        param_space=config,
        # Specify tuning behavior
        tune_config=tune_config,
        # Specify run behavior
        run_config=run_config,
    )
    
    # Run tuning job
    results = tuner.fit()
    print(results.get_best_result(metric="loss", mode="min").config)


# In[105]:


main(num_samples=20, max_num_epochs=10)


# In[ ]:


import sys
sys.exit()

net = TSTransformerEncoderClassiregressor(
    feat_dim = 3,
    max_len = 8192,
    num_classes = 16,
    d_model = 128,
    n_heads = 16, 
    num_layers = 3,
    dim_feedforward = 256,
    dropout = 0.1,
    pos_encoding = 'learnable',
    activation = 'gelu',
    norm = 'BatchNorm', 
    freeze = False,
    )

trainloader, valloader = load_data(data_dir, batch_size=5, train_test_split=0.8)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for i, data in enumerate(trainloader, 0):

    net.to(device)
    # forward + backward + optimize
    out = net(data['d_t'].to(device))
    
    print (i)



# In[ ]:


max_len = 8192
patch_size = 16

num_patches = max_len // patch_size
patch_dim = 3 * patch_size


to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p = 16),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, 128),
            nn.LayerNorm(128),
        )

