#!/usr/bin/env python
# coding: utf-8

# In[84]:


# Load simulation data

import zarr
import swyft.lightning as sl

data_directory = '/scratch-shared/scur2012/peregrine_data/bhardwaj2023'

run_name = 'lowSNR'
rnd_id = 1

data_dir = f'/scratch-shared/scur2012/peregrine_data/bhardwaj2023/simulations_{run_name}_R{rnd_id}'
# simulation_results = zarr.convenience.open(simulation_store_path)


# In[85]:


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


# In[86]:


# Transformer model 
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_1d.py

import torch
from torch import nn

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, seq_len, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert (seq_len % patch_size) == 0

        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, series):
        
        x = self.to_patch_embedding(series)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)
        x, ps = pack([cls_tokens, x], 'b * d')
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        cls_tokens, _ = unpack(x, ps, 'b * d')

        return self.mlp_head(cls_tokens)



# In[87]:


# Define Inference network
import numpy as np
import swyft.lightning as sl
from toolz.dicttoolz import valmap
import torch.nn.functional as F

class ViTInferenceNetwork(sl.SwyftModule):
    
    def __init__(self, batch_size, learning_rate):
        super().__init__()
        
        self.batch_size = batch_size
        self.noise_shuffling = True
        self.num_params = 15
        self.marginals = (0,1),
        self.include_noise = True

        self.ViT_t = ViT(
            seq_len = 8192,
            channels = 3,
            patch_size = 16,
            num_classes = 16,
            dim = 1024,
            depth = 6,
            heads = 8,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        
        self.ViT_f = ViT(
            seq_len = 4096,
            channels = 6,
            patch_size = 16,
            num_classes = 16,
            dim = 1024,
            depth = 6,
            heads = 8,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
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
    
    
class ViTInferenceNetwork_t(sl.SwyftModule):
    
    def __init__(
        self, 
        batch_size, 
        learning_rate, 
        patch_size, 
        num_classes, 
        dim, 
        depth, 
        heads,
        mlp_dim,
        dropout,
        emb_dropout
        ):
        super().__init__()
        
        self.batch_size = batch_size
        self.noise_shuffling = True
        self.num_params = 15
        self.marginals = (0,1),
        self.include_noise = True

        self.ViT_t = ViT(
            seq_len = 8192,
            channels = 3,
            patch_size = patch_size,
            num_classes = num_classes,
            dim = dim,
            depth = depth,
            heads = heads,
            mlp_dim = mlp_dim,
            dropout = dropout,
            emb_dropout = emb_dropout,
        )
               
        self.logratios_1d = sl.LogRatioEstimator_1dim(
            num_features=num_classes, num_params=int(self.num_params), varnames="z_total"
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

        features = self.ViT_t(d_t)        
        logratios_1d = self.logratios_1d(features, z_total)
        
        return logratios_1d
    
class ViTInferenceNetwork_f(sl.SwyftModule):
    
    def __init__(
        self, 
        batch_size, 
        learning_rate, 
        patch_size, 
        num_classes, 
        dim, 
        depth, 
        heads,
        mlp_dim,
        dropout,
        emb_dropout
        ):
        super().__init__()
        
        self.batch_size = batch_size
        self.noise_shuffling = True
        self.num_params = 15
        self.marginals = (0,1),
        self.include_noise = True

        self.ViT_f = ViT(
            seq_len = 4096,
            channels = 6,
            patch_size = patch_size,
            num_classes = num_classes,
            dim = dim,
            depth = depth,
            heads = heads,
            mlp_dim = mlp_dim,
            dropout = dropout,
            emb_dropout = emb_dropout,
        )
               
        self.logratios_1d = sl.LogRatioEstimator_1dim(
            num_features=num_classes, num_params=int(self.num_params), varnames="z_total"
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

        features = self.ViT_f(d_f_w[:,:,:-1])        
        logratios_1d = self.logratios_1d(features, z_total)
        
        return logratios_1d


# In[88]:


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

def train_transformer_t(config, data_dir=None):
    
    print (config)
    
    net = ViTInferenceNetwork_t(        
        batch_size = config['batch_size'], 
        learning_rate = config['learning_rate'], 
        patch_size = config['patch_size'], 
        num_classes = config['num_classes'], 
        dim = config['dim'], 
        depth = config['depth'], 
        heads = config['heads'],
        mlp_dim = config['mlp_dim'],
        dropout = config['dropout'],
        emb_dropout = config['emb_dropout']
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

    for epoch in range(start_epoch, 10):  # loop over the dataset multiple times
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
            #if i % 20 == 19:  # print every 20 mini-batches
            #    print(
            #        "[%d, %5d] loss: %.6f"
            #        % (epoch + 1, i + 1, running_loss / epoch_steps)
            #    )
            #    running_loss = 0.0

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


# In[89]:


import os
from functools import partial

def main(num_samples=10, max_num_epochs=10):
    
    data_dir = os.path.abspath(f"/scratch-shared/scur2012/peregrine_data/bhardwaj2023/simulations_{run_name}_R{rnd_id}")
        
    config = {
        "batch_size": tune.choice([32,64]),
        "learning_rate": tune.choice([1.2e-4, 1.6e-4]),
        # "train_test_split": tune.choice([0.8]),
        # "patch_size": tune.choice([16]),
        # "num_classes": tune.choice([16]),                      
        # "dim": tune.choice([1024]),
        # "depth": tune.choice([6]),
        # "heads": tune.choice([8]),                          
        # "mlp_dim": tune.choice([2048]),                          
        # "dropout": tune.choice([0.1]),                          
        # "emb_dropout": tune.choice([0.1]),
        # 
        # "batch_size": tune.choice([16, 32, 64, 128]),
        # "train_test_split": tune.choice([0.8]),
        "patch_size": tune.choice([8, 12, 16]),
        "num_classes": tune.choice([16, 20, 24, 28, 32]),
        "dim": tune.choice([256, 512, 1024, 2048]),
        "depth": tune.choice([4, 5, 6, 7, 8, 9, 10]),
        "heads": tune.choice([6, 7, 8, 9, 10]),
        "mlp_dim": tune.choice([512, 1024, 2048]),
        "dropout": tune.choice([0, 0.1, 0.2]),
        "emb_dropout": tune.choice([0, 0.1, 0.2]),
    }
    
    first_guess = [{
        "batch_size": 64,
        "learning_rate": 1.6e-4,
        "patch_size": 16,
        "num_classes": 32,
        "dim": 512,
        "depth": 6,
        "heads": 8,
        "mlp_dim": 1024,
        "dropout": 0,
        "emb_dropout": 0.1,
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
        search_alg=HyperOptSearch(points_to_evaluate=first_guess, metric='loss', mode='min'),
        scheduler=scheduler,
    )
    
    stopper = CombinedStopper(
        MaximumIterationStopper(max_iter=max_num_epochs),
        TrialPlateauStopper(metric="loss"),
    )
    
    run_config = RunConfig(
        name="transformer_t",
        storage_path='/home/scur2012/Thesis/master-thesis/experiments/tuning/ray_results',
        checkpoint_config=CheckpointConfig(num_to_keep=10, checkpoint_score_attribute='loss', checkpoint_score_order='min'),
        log_to_file=True,
        stop=stopper,
    )
    
    # Create Tuner
    trainable_with_cpu_gpu = tune.with_resources(partial(train_transformer_t, data_dir=data_dir), {"cpu": 18, "gpu": 1})
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
    
    # result = tune.run(
    #     partial(train_transformer_t, data_dir=data_dir),
    #     name='transformer_t',
    #     resources_per_trial={"cpu": 18, "gpu": 1},
    #     config=config,
    #     num_samples=num_samples,
    #     scheduler=scheduler,
    #     time_budget_s=2500,
    #     log_to_file=True,
    #     raise_on_failed_trial=False,
    #     storage_path='/home/scur2012/Thesis/master-thesis/experiments/tuning/ray_results',
    #     checkpoint_config=CheckpointConfig(num_to_keep=5, checkpoint_score_attribute='loss', checkpoint_score_order='min'),
    #     stop=early_stopping,
    # )
# 
    # best_trial = result.get_best_trial("loss", "min", "last")
    # print(f"Best trial config: {best_trial.config}")
    # print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
# 
    # best_trained_model = ViTInferenceNetwork_t(
    #     batch_size = best_trial.config['batch_size'], 
    #     learning_rate = best_trial.config['learning_rate'], 
    #     patch_size = best_trial.config['patch_size'], 
    #     num_classes = best_trial.config['num_classes'], 
    #     dim = best_trial.config['dim'], 
    #     depth = best_trial.config['depth'], 
    #     heads = best_trial.config['heads'],
    #     mlp_dim = best_trial.config['mlp_dim'],
    #     dropout = best_trial.config['dropout'],
    #     emb_dropout = best_trial.config['emb_dropout']
    # )
    # 
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # best_trained_model.to(device)
# 
    # best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
    # best_checkpoint_data = best_checkpoint.to_dict()
# 
    # best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])


# In[90]:


main(num_samples=50, max_num_epochs=20)

