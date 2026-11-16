import argparse
import os
import re
import traceback

import torch
from faster_whisper import WhisperModel
from funasr import AutoModel
from tqdm import tqdm

from tools.my_utils import load_cudnn

# 僅保留人聲副語言事件，明確忽略 BGM、掌聲、撞擊與環境噪音
SENSEVOICE_EVENT_MAP = {
    "<|Laughter|>": "[LAUGH]",
    "<|Sigh|>": "[SIGH]",
    "<|Cough|>": "[COUGH]",
    "<|Breath|>": "[BREATH]",
}
SV_TAG_PATTERN = re.compile(r"(<\|[^|]+?\|>)")


def extract_vocal_events(sv_res: list) -> list[tuple[float, str]]:
    """從 SenseVoice 輸出中僅提取人聲副語言事件及其時間點，忽略 BGM 與撞擊噪音。"""
    if not sv_res:
        return []
    item = sv_res[0]
    raw_text = item.get("text", "")
    timestamps = item.get("timestamp", [])  # [[start_ms, end_ms], ...]

    events = []
    tokens = SV_TAG_PATTERN.split(raw_text)
    ts_idx = 0

    for tok in tokens:
        if not tok:
            continue
        if tok in SENSEVOICE_EVENT_MAP:
            # 取得對應時間戳（若 SenseVoice 未輸出逐 token 時間戳，則回退至 0.0）
            if ts_idx < len(timestamps):
                start_sec = timestamps[ts_idx][0] / 1000.0
            else:
                start_sec = 0.0
            events.append((start_sec, SENSEVOICE_EVENT_MAP[tok]))
        if not tok.startswith("<|"):
            # 非控制標籤的文字字元會消耗 timestamp 索引
            clean_chars = tok.strip()
            ts_idx += len(clean_chars)

    # 過濾同一秒內重複觸發的相同事件
    deduped = []
    for t_sec, tag in events:
        if not deduped or deduped[-1][1] != tag or (t_sec - deduped[-1][0]) > 0.8:
            deduped.append((t_sec, tag))
    return deduped


def merge_text_and_events(words: list[tuple[float, float, str]], events: list[tuple[float, str]]) -> str:
    """按時間軸座標將人聲副語言標籤精準插入台灣 Whisper 字級逐字稿中。"""
    if not words and not events:
        return ""
    if not words:
        return "".join(tag for _, tag in events)
    if not events:
        return "".join(w[2] for w in words)

    timeline = []
    for start_s, _, char in words:
        timeline.append((start_s, 1, char))
    for ev_s, tag in events:
        timeline.append((ev_s, 0, tag))

    timeline.sort(key=lambda x: (x[0], x[1]))
    return "".join(item[2] for item in timeline)


def execute_asr(input_folder, output_folder, model_path, sv_model_path, language, precision):
    print("loading faster whisper model:", model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel(model_path, device=device, compute_type=precision)

    print("loading SenseVoice model for vocal paralinguistic events:", sv_model_path)
    sv_model = AutoModel(
        model=sv_model_path,
        
        device=device,
        disable_update=True,
        log_level="ERROR",
    )

    input_file_names = sorted(os.listdir(input_folder))
    output = []
    skipped_non_vocal = 0
    output_file_name = os.path.basename(input_folder)

    for file_name in tqdm(input_file_names):
        try:
            file_path = os.path.join(input_folder, file_name)
            if not os.path.isfile(file_path):
                continue

            segments, info = model.transcribe(
                audio=file_path,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=700),
                language=language,
                word_timestamps=True,
            )

            words = []
            for segment in segments:
                if segment.words:
                    for w in segment.words:
                        words.append((w.start, w.end, w.word))
                elif segment.text.strip():
                    words.append((segment.start, segment.end, segment.text))

            sv_res = sv_model.generate(
                input=file_path,
                cache={},
                language="zh",
                use_itn=False,
                output_timestamp=True,
            )
            events = extract_vocal_events(sv_res)
            merged_text = merge_text_and_events(words, events).strip()

            # 純音樂、撞擊聲或無語音底噪直接丟棄，不寫入 .list
            if not merged_text:
                skipped_non_vocal += 1
                continue

            output.append(f"{file_path}|{output_file_name}|{info.language.upper()}|{merged_text}")
        except Exception as e:
            print(e)
            traceback.print_exc()

    output_folder = output_folder or "output/asr_opt"
    os.makedirs(output_folder, exist_ok=True)
    output_file_path = os.path.abspath(f"{output_folder}/{output_file_name}.list")

    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output) + ("\n" if output else ""))
        print(f"ASR output path: {output_file_path} (kept: {len(output)}, skipped non-vocal: {skipped_non_vocal})\n")

    return output_file_path


load_cudnn()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_folder", type=str, required=True, help="Path to the folder containing WAV files."
    )
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Output folder to store transcriptions.")
    parser.add_argument(
        "-s",
        "--model_size",
        type=str,
        default="default",
        choices=["default"],
        help="Model Size of Faster Whisper",
    )
    parser.add_argument(
        "--sv_model",
        type=str,
        default="./GPT_SoVITS/pretrained_models/SenceVoiceSmall",
        help="SenseVoice model path or ModelScope ID for vocal event detection.",
    )
    parser.add_argument(
        "-l", "--language", type=str, default="zh", choices=["zh"], help="Language of the audio files."
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=str,
        default="float16",
        choices=["float16", "float32", "int8"],
        help="fp16, int8 or fp32",
    )

    cmd = parser.parse_args()
    execute_asr(
        input_folder=cmd.input_folder,
        output_folder=cmd.output_folder,
        model_path="./Taiwan-Tongues-ASR-CE-v1.0",
        sv_model_path=cmd.sv_model,
        language=cmd.language,
        precision=cmd.precision,
    )
