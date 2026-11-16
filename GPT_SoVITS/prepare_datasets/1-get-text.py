# -*- coding: utf-8 -*-

import os
import traceback
from GPT_SoVITS.prepare_datasets.dataset_io import DatasetPathResolver
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from text.cleaner import clean_text

cfg, resolver = DatasetPathResolver.from_env()
txt_path = resolver.text_shard_path()

if not txt_path.exists():
    resolver.ensure_dirs(resolver.bert_dir)
    if not os.path.exists(cfg.bert_pretrained_dir):
        raise FileNotFoundError(cfg.bert_pretrained_dir)

    tokenizer = AutoTokenizer.from_pretrained(cfg.bert_pretrained_dir)
    bert_model = AutoModelForMaskedLM.from_pretrained(cfg.bert_pretrained_dir)
    bert_model = (bert_model.half() if cfg.is_half else bert_model).to(cfg.device)
    bert_model.eval()

    def get_bert_feature(text: str, word2ph: list[int]) -> torch.Tensor:
        with torch.no_grad():
            inputs = {k: v.to(cfg.device) for k, v in tokenizer(text, return_tensors="pt").items()}
            res = bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]

        assert len(word2ph) == len(text)
        phone_level_feature = [res[i].repeat(word2ph[i], 1) for i in range(len(word2ph))]
        return torch.cat(phone_level_feature, dim=0).T

    lang_map = {
        "ZH": "zh", "zh": "zh",
        "JP": "ja", "jp": "ja", "JA": "ja", "ja": "ja",
        "EN": "en", "en": "en", "En": "en",
        "KO": "ko", "Ko": "ko", "ko": "ko",
        "yue": "yue", "YUE": "yue", "Yue": "yue",
    }

    opt = []
    skipped_empty = 0
    for entry in resolver.iter_entries():
        try:
            lan = lang_map.get(entry.language)
            if not lan:
                print(f"[Warning] Unsupported language '{entry.language}' for {entry.item_name}")
                continue

            phones, word2ph, norm_text = clean_text(entry.text.replace("%", "-").replace("￥", ","), lan, cfg.version)
            if not phones or not norm_text or not word2ph:
                skipped_empty += 1
                continue

            path_bert = resolver.bert_path(entry.item_name)
            if not path_bert.exists() and lan == "zh":
                bert_feature = get_bert_feature(norm_text, word2ph)
                assert bert_feature.shape[-1] == len(phones)
                resolver.save_tensor(bert_feature, path_bert)

            opt.append(f"{entry.item_name}\t{' '.join(phones)}\t{word2ph}\t{norm_text}")
        except Exception:
            print(entry.item_name, traceback.format_exc())

    if skipped_empty:
        print(f"[Info] Skipped {skipped_empty} empty/non-verbal entries.")
    txt_path.write_text("\n".join(opt) + "\n", encoding="utf8")
