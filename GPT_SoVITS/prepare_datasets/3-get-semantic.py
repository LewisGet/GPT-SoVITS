# -*- coding: utf-8 -*-

import os
import traceback
import logging
from GPT_SoVITS.prepare_datasets.dataset_io import DatasetPathResolver
import torch
import utils

logging.getLogger("numba").setLevel(logging.WARNING)

cfg, resolver = DatasetPathResolver.from_env()
semantic_path = resolver.semantic_shard_path()

if not os.path.exists(cfg.pretrained_s2G):
    raise FileNotFoundError(cfg.pretrained_s2G)

size = os.path.getsize(cfg.pretrained_s2G)
if size < 82978 * 1024:
    version = "v1"
elif size < 100 * 1024 * 1024:
    version = "v2"
elif size < 103520 * 1024:
    version = "v1"
elif size < 700 * 1024 * 1024:
    version = "v2"
else:
    version = "v3"

if version != "v3":
    from module.models import SynthesizerTrn
else:
    from module.models import SynthesizerTrnV3 as SynthesizerTrn

if not semantic_path.exists():
    resolver.ensure_dirs()
    hps = utils.get_hparams_from_file(cfg.s2config_path)

    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        version=version,
        **hps.model,
    )
    vq_model = (vq_model.half() if cfg.is_half else vq_model).to(cfg.device)
    vq_model.eval()
    print(
        vq_model.load_state_dict(
            torch.load(cfg.pretrained_s2G, map_location="cpu", weights_only=False)["weight"], strict=False
        )
    )

    lines_out = []
    for entry in resolver.iter_entries():
        try:
            hubert_in = resolver.hubert_path(entry.item_name)
            if not hubert_in.exists():
                continue

            ssl_content = torch.load(str(hubert_in), map_location="cpu")
            ssl_content = (ssl_content.half() if cfg.is_half else ssl_content).to(cfg.device)
            with torch.no_grad():
                codes = vq_model.extract_latent(ssl_content)
            semantic = " ".join(str(i) for i in codes[0, 0, :].tolist())
            lines_out.append(f"{entry.item_name}\t{semantic}")
        except Exception:
            print(entry.item_name, traceback.format_exc())

    semantic_path.write_text("\n".join(lines_out), encoding="utf8")
