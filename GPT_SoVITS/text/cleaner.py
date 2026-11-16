from text import cleaned_text_to_sequence
import os
import re
from text import symbols as symbols_v1
from text import symbols2 as symbols_v2

special = [
    ("￥", "zh", "SP2"),
    ("^", "zh", "SP3"),
]

# 映射為 chinese-roberta-wwm-ext-large vocab.txt 原生收錄的獨立單一符號（非 [UNK]）
EVENT_TO_BERT_SYMBOL = {
    "[LAUGH]": "√",
    "[SIGH]": "±",
    "[COUGH]": "×",
    "[BREATH]": "÷",
}
EVENT_PATTERN = re.compile(r"(\[(?:LAUGH|SIGH|COUGH|BREATH)\])")


def _clean_text_segment(text, language, version=None):
    if version is None:
        version = os.environ.get("version", "v2")
    if version == "v1":
        symbols = symbols_v1.symbols
        language_module_map = {"zh": "chinese", "ja": "japanese", "en": "english"}
    else:
        symbols = symbols_v2.symbols
        language_module_map = {"zh": "chinese2", "ja": "japanese", "en": "english", "ko": "korean", "yue": "cantonese"}

    if language not in language_module_map:
        language = "en"
        text = " "
    for special_s, special_l, target_symbol in special:
        if special_s in text and language == special_l:
            return clean_special(text, language, special_s, target_symbol, version)
    language_module = __import__("text." + language_module_map[language], fromlist=[language_module_map[language]])
    if hasattr(language_module, "text_normalize"):
        norm_text = language_module.text_normalize(text)
    else:
        norm_text = text
    if language == "zh" or language == "yue":
        phones, word2ph = language_module.g2p(norm_text)
        assert len(phones) == sum(word2ph)
        assert len(norm_text) == len(word2ph)
    elif language == "en":
        phones = language_module.g2p(norm_text)
        if len(phones) < 4:
            phones = [","] + phones
        word2ph = None
    else:
        phones = language_module.g2p(norm_text)
        word2ph = None
    phones = ["UNK" if ph not in symbols else ph for ph in phones]
    return phones, word2ph, norm_text


def clean_text(text, language, version=None):
    if not EVENT_PATTERN.search(text):
        return _clean_text_segment(text, language, version)

    parts = EVENT_PATTERN.split(text)
    all_phones = []
    all_word2ph = [] if language in ("zh", "yue") else None
    all_norm_text = []

    for part in parts:
        if not part:
            continue
        if part in EVENT_TO_BERT_SYMBOL:
            all_phones.append(part)
            if all_word2ph is not None:
                all_word2ph.append(1)
            all_norm_text.append(EVENT_TO_BERT_SYMBOL[part])
        else:
            phones, word2ph, norm_text = _clean_text_segment(part, language, version)
            all_phones.extend(phones)
            if all_word2ph is not None and word2ph is not None:
                all_word2ph.extend(word2ph)
            all_norm_text.append(norm_text)

    final_norm = "".join(all_norm_text)
    if all_word2ph is not None:
        assert len(all_phones) == sum(all_word2ph)
        assert len(final_norm) == len(all_word2ph)
    return all_phones, all_word2ph, final_norm


def clean_special(text, language, special_s, target_symbol, version=None):
    if version is None:
        version = os.environ.get("version", "v2")
    if version == "v1":
        symbols = symbols_v1.symbols
        language_module_map = {"zh": "chinese", "ja": "japanese", "en": "english"}
    else:
        symbols = symbols_v2.symbols
        language_module_map = {"zh": "chinese2", "ja": "japanese", "en": "english", "ko": "korean", "yue": "cantonese"}

    text = text.replace(special_s, ",")
    language_module = __import__("text." + language_module_map[language], fromlist=[language_module_map[language]])
    norm_text = language_module.text_normalize(text)
    phones = language_module.g2p(norm_text)
    new_ph = []
    for ph in phones[0]:
        assert ph in symbols
        if ph == ",":
            new_ph.append(target_symbol)
        else:
            new_ph.append(ph)
    return new_ph, phones[1], norm_text


def text_to_sequence(text, language, version=None):
    version = os.environ.get("version", version)
    if version is None:
        version = "v2"
    phones, _, _ = clean_text(text, language, version)
    return cleaned_text_to_sequence(phones, version)
