# -*- coding: utf-8 -*-

from dataclasses import dataclass
import os
from pathlib import Path
import sys
import tempfile
from typing import Iterator
import av
import numpy as np


def _bootstrap_runtime():
    """在 import torch 前完成 CUDA_VISIBLE_DEVICES 映射與工作目錄路徑注入。"""
    if "_CUDA_VISIBLE_DEVICES" in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.append(cwd)


_bootstrap_runtime()

import torch


@dataclass(frozen=True)
class EnvConfig:
    inp_text: str
    inp_wav_dir: str
    exp_name: str
    opt_dir: str
    i_part: int
    all_parts: int
    is_half: bool
    device: str
    version: str | None
    bert_pretrained_dir: str
    cnhubert_base_dir: str
    sv_path: str
    pretrained_s2G: str
    s2config_path: str

    @classmethod
    def load(cls) -> "EnvConfig":
        has_cuda = torch.cuda.is_available()
        is_half = str(os.environ.get("is_half", "True")).lower() == "true" and has_cuda
        return cls(
            inp_text=os.environ.get("inp_text", ""),
            inp_wav_dir=os.environ.get("inp_wav_dir", ""),
            exp_name=os.environ.get("exp_name", ""),
            opt_dir=os.environ.get("opt_dir", ""),
            i_part=int(os.environ.get("i_part", "0")),
            all_parts=int(os.environ.get("all_parts", "1")),
            is_half=is_half,
            device="cuda:0" if has_cuda else "cpu",
            version=os.environ.get("version"),
            bert_pretrained_dir=os.environ.get("bert_pretrained_dir", ""),
            cnhubert_base_dir=os.environ.get("cnhubert_base_dir", ""),
            sv_path=os.environ.get("sv_path", ""),
            pretrained_s2G=os.environ.get("pretrained_s2G", ""),
            s2config_path=os.environ.get("s2config_path", ""),
        )


@dataclass(frozen=True)
class ListEntry:
    item_name: str
    src_path: Path
    spk_name: str
    language: str
    text: str


class DatasetPathResolver:
    """統一管理環境設定、資料集清單元資料讀取、分片、命名空間主鍵生成與原子存檔。"""

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.opt_dir = Path(cfg.opt_dir)
        self.i_part = cfg.i_part
        self.all_parts = cfg.all_parts
        self.bert_dir = self.opt_dir / "3-bert"
        self.hubert_dir = self.opt_dir / "4-cnhubert"
        self.wav32_dir = self.opt_dir / "5-wav32k"
        self.sv_dir = self.opt_dir / "7-sv_cn"

    @classmethod
    def from_env(cls) -> tuple[EnvConfig, "DatasetPathResolver"]:
        cfg = EnvConfig.load()
        return cfg, cls(cfg)

    def ensure_dirs(self, *dirs: Path):
        self.opt_dir.mkdir(parents=True, exist_ok=True)
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def resolve_source_path(raw_path: str) -> Path:
        cleaned = raw_path.strip().strip('"').strip("'").replace("\\", "/")
        return Path(cleaned)

    @classmethod
    def to_item_name(cls, raw_path: str) -> str:
        p = cls.resolve_source_path(raw_path)
        parent = p.parent.name
        return f"{parent}__{p.name}" if parent else p.name

    def iter_entries(self) -> Iterator[ListEntry]:
        """統一讀取並切分 .list，過濾空行並封裝為結構化物件。"""
        raw_lines = Path(self.cfg.inp_text).read_text(encoding="utf8").splitlines()
        valid_lines = [ln for ln in raw_lines if ln.strip()]
        for line in valid_lines[self.i_part :: self.all_parts]:
            parts = line.split("|", maxsplit=3)
            if len(parts) != 4:
                print(f"[Skip] Malformed line: {line}")
                continue
            raw_wav, spk_name, language, text = parts
            src_path = self.resolve_source_path(raw_wav)
            yield ListEntry(
                item_name=self.to_item_name(raw_wav),
                src_path=src_path,
                spk_name=spk_name.strip(),
                language=language.strip(),
                text=text,
            )

    def text_shard_path(self) -> Path:
        return self.opt_dir / f"2-name2text-{self.i_part}.txt"

    def semantic_shard_path(self) -> Path:
        return self.opt_dir / f"6-name2semantic-{self.i_part}.tsv"

    def bert_path(self, item_name: str) -> Path:
        return self.bert_dir / f"{item_name}.pt"

    def hubert_path(self, item_name: str) -> Path:
        return self.hubert_dir / f"{item_name}.pt"

    def wav32_path(self, item_name: str) -> Path:
        return self.wav32_dir / item_name

    def sv_path(self, item_name: str) -> Path:
        return self.sv_dir / f"{item_name}.pt"

    @staticmethod
    def save_tensor(tensor: torch.Tensor, target_path: Path):
        """同分區原子寫入，避免多進程競爭與中文路徑編碼問題。"""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=target_path.parent, suffix=".tmp")
        os.close(fd)
        try:
            torch.save(tensor, tmp_name)
            os.replace(tmp_name, target_path)
        except Exception:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
            raise

    @staticmethod
    def load_audio_mono(wav_path: Path, target_sr: int = 32000) -> np.ndarray:
        """直接使用 PyAV 解碼並重採樣為單聲道 float32 波形，取代外部 ffmpeg subprocess。"""
        with av.open(str(wav_path)) as container:
            stream = next(s for s in container.streams if s.type == "audio")
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr)
            chunks = []
            for frame in container.decode(stream):
                for resampled in resampler.resample(frame):
                    chunks.append(resampled.to_ndarray()[0])
            for resampled in resampler.resample(None):
                chunks.append(resampled.to_ndarray()[0])
        if not chunks:
            raise RuntimeError(f"Empty audio stream: {wav_path}")
        return np.concatenate(chunks, axis=0).astype(np.float32)
