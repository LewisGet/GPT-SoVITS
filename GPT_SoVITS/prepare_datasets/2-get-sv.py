# -*- coding: utf-8 -*-

import traceback
from GPT_SoVITS.prepare_datasets.dataset_io import DatasetPathResolver
import torch
import torchaudio
from GPT_SoVITS.eres2net.ERes2NetV2 import ERes2NetV2
import GPT_SoVITS.eres2net.kaldi as Kaldi

cfg, resolver = DatasetPathResolver.from_env()
resolver.ensure_dirs(resolver.sv_dir, resolver.wav32_dir)


class SV:
    def __init__(self, device: str, is_half: bool):
        pretrained_state = torch.load(cfg.sv_path, map_location="cpu")
        embedding_model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
        embedding_model.load_state_dict(pretrained_state)
        embedding_model.eval()
        self.res = torchaudio.transforms.Resample(32000, 16000).to(device)
        self.embedding_model = (embedding_model.half() if is_half else embedding_model).to(device)
        self.is_half = is_half

    def compute_embedding3(self, wav: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            wav = self.res(wav)
            if self.is_half:
                wav = wav.half()
            feat = torch.stack(
                [Kaldi.fbank(wav0.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0) for wav0 in wav]
            )
            return self.embedding_model.forward3(feat)


sv = SV(cfg.device, cfg.is_half)

for entry in resolver.iter_entries():
    try:
        sv_out = resolver.sv_path(entry.item_name)
        wav_in = resolver.wav32_path(entry.item_name)

        if sv_out.exists() or not wav_in.exists():
            continue

        wav32k, sr0 = torchaudio.load(str(wav_in))
        assert sr0 == 32000
        emb = sv.compute_embedding3(wav32k.to(cfg.device)).cpu()
        resolver.save_tensor(emb, sv_out)
    except Exception:
        print(entry.item_name, traceback.format_exc())
