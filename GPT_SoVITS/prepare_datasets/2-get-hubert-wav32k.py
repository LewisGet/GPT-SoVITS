# -*- coding: utf-8 -*-

import traceback
from GPT_SoVITS.prepare_datasets.dataset_io import DatasetPathResolver
import numpy as np
import torch
import librosa
from scipy.io import wavfile
from feature_extractor import cnhubert

cfg, resolver = DatasetPathResolver.from_env()
resolver.ensure_dirs(resolver.hubert_dir, resolver.wav32_dir)

cnhubert.cnhubert_base_path = cfg.cnhubert_base_dir
model = cnhubert.get_model()
model = (model.half() if cfg.is_half else model).to(cfg.device)
model.eval()

maxx = 0.95
alpha = 0.5
nan_fails = []
use_half = cfg.is_half


def process_item(item_name: str, src_path):
    hubert_out = resolver.hubert_path(item_name)
    if hubert_out.exists():
        return

    tmp_audio = resolver.load_audio_mono(src_path, 32000)
    tmp_max = np.abs(tmp_audio).max()
    if tmp_max > 2.2:
        print(f"{item_name}-filtered,{tmp_max}")
        return

    tmp_audio32 = (tmp_audio / tmp_max * (maxx * alpha * 32768)) + ((1 - alpha) * 32768) * tmp_audio
    tmp_audio32b = (tmp_audio / tmp_max * (maxx * alpha * 1145.14)) + ((1 - alpha) * 1145.14) * tmp_audio
    wav16 = librosa.resample(tmp_audio32b, orig_sr=32000, target_sr=16000)

    tensor_wav16 = torch.from_numpy(wav16)
    tensor_wav16 = (tensor_wav16.half() if use_half else tensor_wav16).to(cfg.device)

    with torch.no_grad():
        ssl = model.model(tensor_wav16.unsqueeze(0))["last_hidden_state"].transpose(1, 2).cpu()

    if torch.isnan(ssl).any():
        nan_fails.append((item_name, src_path))
        print(f"nan filtered: {item_name}")
        return

    wavfile.write(str(resolver.wav32_path(item_name)), 32000, tmp_audio32.astype("int16"))
    resolver.save_tensor(ssl, hubert_out)


for entry in resolver.iter_entries():
    try:
        process_item(entry.item_name, entry.src_path)
    except Exception:
        print(entry.item_name, traceback.format_exc())

if nan_fails and use_half:
    use_half = False
    model = model.float()
    for item_name, src_path in nan_fails:
        try:
            process_item(item_name, src_path)
        except Exception:
            print(item_name, traceback.format_exc())
