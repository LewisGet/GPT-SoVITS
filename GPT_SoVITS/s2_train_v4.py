import logging
import os
import warnings
from random import randint

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("h5py").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.INFO)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
from module import commons
from module.data_utils import (
    DistributedBucketSampler,
    TextAudioSpeakerCollateV3,
    TextAudioSpeakerCollateV4,
    TextAudioSpeakerLoaderV3,
    TextAudioSpeakerLoaderV4,
)
from module.models import SynthesizerTrnV3
from process_ckpt import savee

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")

hps = utils.get_hparams(stage=2)
os.environ["CUDA_VISIBLE_DEVICES"] = hps.train.gpu_numbers.replace("-", ",")

LOADER_REGISTRY = {
    "v3": (TextAudioSpeakerLoaderV3, TextAudioSpeakerCollateV3),
    "v4": (TextAudioSpeakerLoaderV4, TextAudioSpeakerCollateV4),
    "v5": (TextAudioSpeakerLoaderV4, TextAudioSpeakerCollateV4),
}


def main():
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps))


def run(rank: int, n_gpus: int, hps):
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    dist.init_process_group(
        backend="gloo" if os.name == "nt" or not torch.cuda.is_available() else "nccl",
        init_method="env://?use_libuv=False",
        world_size=n_gpus,
        rank=rank,
    )
    torch.manual_seed(hps.train.seed)

    logger = utils.get_logger(hps.data.exp_dir) if rank == 0 else None
    writer = SummaryWriter(log_dir=hps.s2_ckpt_dir) if rank == 0 else None
    if logger:
        logger.info(hps)

    # 依模型版本自動匹配對應採樣率與幀率的 Loader (v4/v5 -> 32kHz Mel / 100Hz; v3 -> 24kHz Mel / 93.75Hz)
    model_ver = getattr(hps.model, "version", "v4")
    if model_ver not in LOADER_REGISTRY:
        raise ValueError(f"Unsupported CFM model version: {model_ver}. Expected one of {list(LOADER_REGISTRY)}")
    loader_cls, collate_cls = LOADER_REGISTRY[model_ver]

    train_dataset = loader_cls(hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    train_loader = DataLoader(
        train_dataset,
        num_workers=getattr(hps.data, "num_workers", 1),
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_cls(),
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=3,
    )

    net_g = SynthesizerTrnV3(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)

    optim_g = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_g.parameters()),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    if torch.cuda.is_available():
        net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)

    ckpt_dir = os.path.join(hps.data.exp_dir, f"logs_s2_{model_ver}")
    os.makedirs(ckpt_dir, exist_ok=True)

    try:
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(ckpt_dir, "G_*.pth"), net_g, optim_g
        )
        epoch_str += 1
        global_step = (epoch_str - 1) * len(train_loader)
    except Exception:
        epoch_str = 1
        global_step = 0
        pretrained_s2g = getattr(hps.train, "pretrained_s2G", None)
        if pretrained_s2g and os.path.exists(pretrained_s2g):
            state_dict = torch.load(pretrained_s2g, map_location="cpu", weights_only=False)["weight"]
            target_net = net_g.module if hasattr(net_g, "module") else net_g
            res = target_net.load_state_dict(state_dict, strict=False)
            if rank == 0:
                logger.info(f"Loaded pretrained {pretrained_s2g}: {res}")

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=-1)
    for _ in range(epoch_str):
        scheduler_g.step()

    scaler = GradScaler(enabled=hps.train.fp16_run)
    if rank == 0:
        print(f"Start CFM ({model_ver}) training from epoch {epoch_str} using {loader_cls.__name__}")

    for epoch in range(epoch_str, hps.train.epochs + 1):
        global_step = train_epoch(
            rank, epoch, global_step, hps, net_g, optim_g, scaler, train_loader, logger, writer, device
        )
        scheduler_g.step()

        if rank == 0 and epoch % hps.train.save_every_epoch == 0:
            ckpt_name = "G_latest.pth" if hps.train.if_save_latest else f"G_{global_step}.pth"
            utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(ckpt_dir, ckpt_name))

            if hps.train.if_save_every_weights:
                raw_state = net_g.module.state_dict() if hasattr(net_g, "module") else net_g.state_dict()
                saved_msg = savee(raw_state, f"{hps.name}_e{epoch}_s{global_step}", epoch, global_step, hps)
                logger.info(f"Saving ckpt {hps.name}_e{epoch}: {saved_msg}")

        if rank == 0:
            logger.info(f"====> Epoch: {epoch}")

    if rank == 0:
        print("Training done.")


def train_epoch(rank, epoch, global_step, hps, net_g, optim_g, scaler, train_loader, logger, writer, device) -> int:
    train_loader.batch_sampler.set_epoch(epoch)
    net_g.train()

    iterator = tqdm(train_loader, desc=f"Epoch {epoch}") if rank == 0 else train_loader
    for batch_idx, (ssl, spec, mel, ssl_lengths, spec_lengths, text, text_lengths, mel_lengths) in enumerate(iterator):
        spec = spec.to(device, non_blocking=True)
        spec_lengths = spec_lengths.to(device, non_blocking=True)
        mel = mel.to(device, non_blocking=True)
        mel_lengths = mel_lengths.to(device, non_blocking=True)
        ssl = ssl.to(device, non_blocking=True)
        ssl.requires_grad = False
        text = text.to(device, non_blocking=True)
        text_lengths = text_lengths.to(device, non_blocking=True)

        with autocast(enabled=hps.train.fp16_run):
            cfm_loss = net_g(
                ssl,
                spec,
                mel,
                ssl_lengths,
                spec_lengths,
                text,
                text_lengths,
                mel_lengths,
                use_grad_ckpt=hps.train.grad_ckpt,
            )

        optim_g.zero_grad()
        scaler.scale(cfm_loss).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0 and global_step % hps.train.log_interval == 0:
            lr = optim_g.param_groups[0]["lr"]
            pct = 100.0 * batch_idx / len(train_loader)
            logger.info(f"Train Epoch: {epoch} [{pct:.0f}%] | cfm_loss={cfm_loss.item():.4f} | step={global_step} | lr={lr:.6f}")
            utils.summarize(
                writer=writer,
                global_step=global_step,
                scalars={"loss/g/total": cfm_loss, "learning_rate": lr, "grad_norm_g": grad_norm_g},
            )

        global_step += 1

    return global_step


if __name__ == "__main__":
    main()
