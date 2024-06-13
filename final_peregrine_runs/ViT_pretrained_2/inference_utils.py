import numpy as np
import os
import torch
from torch import nn
from torch.functional import F
import pickle
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import swyft.lightning as sl

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from toolz.dicttoolz import valmap
from sklearn.metrics import roc_curve, auc

# Transformer model 
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_1d.py

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
    

class InferenceNetwork(sl.SwyftModule):
    def __init__(self, conf):
        super().__init__()
        self.one_d_only = conf["tmnre"]["1d_only"]
        self.batch_size = conf["hparams"]["training_batch_size"]
        self.noise_shuffling = conf["tmnre"]["shuffling"]
        self.num_params = len(conf["priors"]["int_priors"].keys()) + len(
            conf["priors"]["ext_priors"].keys()
        )
        self.marginals = conf["tmnre"]["marginals"]
        # self.unet_t = Unet(
        #     n_in_channels=len(conf["waveform_params"]["ifo_list"]),
        #     n_out_channels=1,
        #     sizes=(16, 32, 64, 128, 256),
        #     down_sampling=(8, 8, 8, 8),
        # )
        # self.unet_f = Unet(
        #     n_in_channels=2 * len(conf["waveform_params"]["ifo_list"]),
        #     n_out_channels=1,
        #     sizes=(16, 32, 64, 128, 256),
        #     down_sampling=(2, 2, 2, 2),
        # )
        
        self.ViT_t = ViT(
            seq_len = 8192,
            channels = len(conf["waveform_params"]["ifo_list"]),
            patch_size = 16,
            num_classes = 16,
            dim = 512,
            depth = 7,
            heads = 6,
            mlp_dim = 1024,
            dropout = 0.0,
            emb_dropout = 0.0,
        )
        
        self.ViT_f = ViT(
            seq_len = 4096,
            channels = 2 * len(conf["waveform_params"]["ifo_list"]),
            patch_size = 16,
            num_classes = 16,
            dim = 512,
            depth = 7,
            heads = 6,
            mlp_dim = 1024,
            dropout = 0,
            emb_dropout = 0,
        )

        # self.flatten = nn.Flatten(1)
        # self.linear_t = LinearCompression()
        # self.linear_f = LinearCompression()

        self.logratios_1d = sl.LogRatioEstimator_1dim(
            num_features=32, num_params=int(self.num_params), varnames="z_total"
        )
        if not self.one_d_only:
            self.linear_t_2d = LinearCompression_2d()
            self.linear_f_2d = LinearCompression_2d()
            self.logratios_2d = sl.LogRatioEstimator_Ndim(
                num_features=256, marginals=self.marginals, varnames="z_total"
            )

        self.optimizer_init = sl.AdamOptimizerInit(lr=conf["hparams"]["learning_rate"])

    def forward(self, A, B):
        if self.noise_shuffling and A["d_t"].size(0) != 1:
            noise_shuffling = torch.randperm(self.batch_size)
            d_t = A["d_t"] + A["n_t"][noise_shuffling]
            d_f_w = A["d_f_w"] + A["n_f_w"][noise_shuffling]
        else:
            d_t = A["d_t"] + A["n_t"]
            d_f_w = A["d_f_w"] + A["n_f_w"]
        z_total = B["z_total"]

        d_t = self.ViT_t(d_t)
        d_f_w = self.ViT_f(d_f_w[:,:,:-1])

        if not self.one_d_only:
            features_t = self.linear_t(self.flatten(d_t))
            features_f = self.linear_f(self.flatten(d_f_w))
            features = torch.cat([features_t, features_f], dim=1)
            logratios_1d = self.logratios_1d(features, z_total)
            features_t_2d = self.linear_t_2d(self.flatten(d_t))
            features_f_2d = self.linear_f_2d(self.flatten(d_f_w))
            features_2d = torch.cat([features_t_2d, features_f_2d], dim=1)
            logratios_2d = self.logratios_2d(features_2d, z_total)
            return logratios_1d, logratios_2d
        else:
            features_t = d_t # self.linear_t(self.flatten(d_t))
            features_f = d_f_w # self.linear_f(self.flatten(d_f_w))
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
    
    def _auc_per_feature(self, batch, randomized=True):
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
        auc_tot = torch.zeros(15).to(self.device)

        logratios = self._get_logratios(
            out
        )  # Generates concatenated flattened list of all estimated log ratios
        if logratios is not None:
            labels = torch.zeros_like(logratios)
            labels[:num_pos, ...] = 1
            
            probabilities = nn.functional.softmax(logratios, dim=0)
            
            for i in range(15):
                fpr, tpr, thresholds = roc_curve(labels.cpu().numpy()[:,i], probabilities[:,i].detach().cpu().numpy())
                auc_tot[i] = auc(fpr, tpr)
            
        return auc_tot

    def training_step(self, batch, batch_idx):
        loss = self._calc_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        
        loss_per_feature = self._calc_loss_per_feature(batch)
        auc_per_feature = self._auc_per_feature(batch)
        for i in range(15):
            self.log(f"train_loss_feature_{i}", loss_per_feature[i], on_step=True, on_epoch=False)
            self.log(f"train_auc_feature_{i}", auc_per_feature[i], on_step=True, on_epoch=False)
            
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calc_loss(batch, randomized=False)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        loss_per_feature = self._calc_loss_per_feature(batch)
        auc_per_feature = self._auc_per_feature(batch)
        for i in range(15):
            self.log(f"val_loss_feature_{i}", loss_per_feature[i], on_step=False, on_epoch=True)
            self.log(f"val_auc_feature_{i}", auc_per_feature[i], on_step=False, on_epoch=True)
        return loss


def init_network(conf: dict):
    """
    Initialise the network with the settings given in a loaded config dictionary
    Args:
      conf: dictionary of config options, output of init_config
    Returns:
      Pytorch lightning network class
    Examples
    --------
    >>> sysargs = ['/path/to/config/file.txt']
    >>> tmnre_parser = read_config(sysargs)
    >>> conf = init_config(tmnre_parser, sysargs, sim=True)
    >>> network = init_network(conf)
    """
    
    checkpoint_path = "/home/scur2012/Thesis/master-thesis/models/vanilla_vit_1d/epoch=10-step=315000_val_loss=-5.59_train_loss=-5.50.ckpt"
    checkpoint = torch.load(checkpoint_path)

    network = InferenceNetwork(conf)
    network.load_state_dict(checkpoint['state_dict'])
    
    return network


def setup_zarr_store(
    conf: dict,
    simulator,
    round_id: int = None,
    coverage: bool = False,
    n_sims: int = None,
):
    """
    Initialise or load a zarr store for saving simulations
    Args:
      conf: dictionary of config options, output of init_config
      simulator: simulator object, output of init_simulator
      round_id: specific round id for store name
      coverage: specifies if store should be used for coverage sims
      n_sims: number of simulations to initialise store with
    Returns:
      Zarr store object
    Examples
    --------
    >>> sysargs = ['/path/to/config/file.txt']
    >>> tmnre_parser = read_config(sysargs)
    >>> conf = init_config(tmnre_parser, sysargs, sim=True)
    >>> simulator = init_simulator(conf)
    >>> store = setup_zarr_store(conf, simulator)
    """
    zarr_params = conf["zarr_params"]
    if zarr_params["use_zarr"]:
        chunk_size = zarr_params["chunk_size"]
        if n_sims is None:
            if "nsims" in zarr_params.keys():
                n_sims = zarr_params["nsims"]
            else:
                n_sims = zarr_params["sim_schedule"][round_id - 1]
        shapes, dtypes = simulator.get_shapes_and_dtypes()
        shapes.pop("n")
        dtypes.pop("n")
        store_path = zarr_params["store_path"]
        if round_id is not None:
            if coverage:
                store_dir = f"{store_path}/coverage_simulations_{zarr_params['run_id']}_R{round_id}"
            else:
                store_dir = (
                    f"{store_path}/simulations_{zarr_params['run_id']}_R{round_id}"
                )
        else:
            if coverage:
                store_dir = (
                    f"{store_path}/coverage_simulations_{zarr_params['run_id']}_R1"
                )
            else:
                store_dir = f"{store_path}/simulations_{zarr_params['run_id']}_R1"

        store = sl.ZarrStore(f"{store_dir}")
        store.init(N=n_sims, chunk_size=chunk_size, shapes=shapes, dtypes=dtypes)
        return store
    else:
        return None


def setup_dataloader(store, simulator, conf: dict, round_id: int = None):
    """
    Initialise a dataloader to read in simulations from a zarr store
    Args:
      store: zarr store to load from, output of setup_zarr_store
      conf: dictionary of config options, output of init_config
      simulator: simulator object, output of init_simulator
      round_id: specific round id for store name
    Returns:
      (training dataloader, validation dataloader), trainer directory
    Examples
    --------
    >>> sysargs = ['/path/to/config/file.txt']
    >>> tmnre_parser = read_config(sysargs)
    >>> conf = init_config(tmnre_parser, sysargs, sim=True)
    >>> simulator = init_simulator(conf)
    >>> store = setup_zarr_store(conf, simulator)
    >>> train_data, val_data, trainer_dir = setup_dataloader(store, simulator, conf)
    """
    if round_id is not None:
        trainer_dir = f"{conf['zarr_params']['store_path']}/trainer_{conf['zarr_params']['run_id']}_R{round_id}"
    else:
        trainer_dir = f"{conf['zarr_params']['store_path']}/trainer_{conf['zarr_params']['run_id']}_R1"
    if not os.path.isdir(trainer_dir):
        os.mkdir(trainer_dir)
    hparams = conf["hparams"]
    if conf["tmnre"]["resampler"]:
        resampler = simulator.get_resampler(targets=conf["tmnre"]["noise_targets"])
    else:
        resampler = None
    train_data = store.get_dataloader(
        num_workers=int(hparams["num_workers"]),
        batch_size=int(hparams["training_batch_size"]),
        idx_range=[0, int(hparams["train_data"] * len(store.data.z_int))],
        on_after_load_sample=resampler,
    )
    val_data = store.get_dataloader(
        num_workers=int(hparams["num_workers"]),
        batch_size=int(hparams["validation_batch_size"]),
        idx_range=[
            int(hparams["train_data"] * len(store.data.z_int)),
            len(store.data.z_int) - 1,
        ],
        on_after_load_sample=None,
    )
    return train_data, val_data, trainer_dir


def setup_trainer(trainer_dir: str, conf: dict, round_id: int):
    """
    Initialise a pytorch lightning trainer and relevant directories
    Args:
      trainer_dir: location for the training logs
      conf: dictionary of config options, output of init_config
      round_id: specific round id for store name
    Returns:
      Swyft lightning trainer instance
    Examples
    --------
    >>> sysargs = ['/path/to/config/file.txt']
    >>> tmnre_parser = read_config(sysargs)
    >>> conf = init_config(tmnre_parser, sysargs, sim=True)
    >>> simulator = init_simulator(conf)
    >>> store = setup_zarr_store(conf, simulator, 1)
    >>> train_data, val_data, trainer_dir = setup_dataloader(store, simulator, conf, 1)
    >>> trainer = setup_trainer(trainer_dir, conf, 1)
    """
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0,
        patience=conf["hparams"]["early_stopping"],
        verbose=False,
        mode="min",
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"{trainer_dir}",
        filename="{epoch}_{val_loss:.2f}_{train_loss:.2f}" + f"_R{round_id}",
        mode="min",
    )
    logger_tbl = pl_loggers.TensorBoardLogger(
        save_dir=f"{trainer_dir}",
        name=f"{conf['zarr_params']['run_id']}_R{round_id}",
        version=None,
        default_hp_metric=False,
    )

    device_params = conf["device_params"]
    hparams = conf["hparams"]
    trainer = sl.SwyftTrainer(
        accelerator=device_params["device"],
        gpus=device_params["n_devices"],
        min_epochs=hparams["min_epochs"],
        max_epochs=hparams["max_epochs"],
        logger=logger_tbl,
        callbacks=[lr_monitor, early_stopping_callback, checkpoint_callback],
    )
    return trainer


def save_logratios(logratios, conf, round_id):
    """
    Save logratios from a particular round
    Args:
      logratios: swyft logratios instance
      conf: dictionary of config options, output of init_config
      round_id: specific round id for store name
    """
    if not os.path.isdir(
        f"{conf['zarr_params']['store_path']}/logratios_{conf['zarr_params']['run_id']}"
    ):
        os.mkdir(
            f"{conf['zarr_params']['store_path']}/logratios_{conf['zarr_params']['run_id']}"
        )
    with open(
        f"{conf['zarr_params']['store_path']}/logratios_{conf['zarr_params']['run_id']}/logratios_R{round_id}",
        "wb",
    ) as p:
        pickle.dump(logratios, p)


def save_bounds(bounds, conf: dict, round_id: int):
    """
    Save bounds from a particular round
    Args:
      bounds: unpacked swyft bounds object
      conf: dictionary of config options, output of init_config
      round_id: specific round id for store name
    """
    np.savetxt(
        f"{conf['zarr_params']['store_path']}/bounds_{conf['zarr_params']['run_id']}_R{round_id}.txt",
        bounds,
    )


def load_bounds(conf: dict, round_id: int):
    """
    Load bounds from a particular round
    Args:
      conf: dictionary of config options, output of init_config
      round_id: specific round id for store name
    Returns:
      Bounds object with ordering defined by the param idxs in the config
    """
    if round_id == 1:
        return None
    else:
        bounds = np.loadtxt(
            f"{conf['zarr_params']['store_path']}/bounds_{conf['zarr_params']['run_id']}_R{round_id - 1}.txt"
        )
        return bounds


def save_coverage(coverage, conf: dict, round_id: int):
    """
    Save coverage samples from a particular round
    Args:
      coverage: swyft coverage object instance
      conf: dictionary of config options, output of init_config
      round_id: specific round id for store name
    """
    if not os.path.isdir(
        f"{conf['zarr_params']['store_path']}/coverage_{conf['zarr_params']['run_id']}"
    ):
        os.mkdir(
            f"{conf['zarr_params']['store_path']}/coverage_{conf['zarr_params']['run_id']}"
        )
    with open(
        f"{conf['zarr_params']['store_path']}/coverage_{conf['zarr_params']['run_id']}/coverage_R{round_id}",
        "wb",
    ) as p:
        pickle.dump(coverage, p)


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
