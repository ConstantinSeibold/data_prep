#!/usr/bin/env python3
"""
translate_jsonl_befund.py

Translate the value under key "befund_text_plain" from German to English
for all .jsonl files in an input directory and save results to an output directory.

Features:
 - Splits long texts into sentence-like chunks and translates chunks, then rejoins.
 - Batches translations for speed.
 - Detects and uses CUDA -> MPS -> CPU automatically, or accept --device override.
 - Supports local Hugging Face models or the Hugging Face Inference API.

Usage examples:
  python translate_jsonl_befund.py --input-dir ./input_jsonl --output-dir ./translated_jsonl
  python translate_jsonl_befund.py -i ./in -o ./out --model Helsinki-NLP/opus-mt-de-en --batch-size 8 --device mps
  python translate_jsonl_befund.py -i ./in -o ./out --use-hf-inference-api --hf-token <TOKEN>
"""

import os
import re
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# ---------- Sentence/chunk splitting helpers ----------
SENTENCE_SPLIT_RE = re.compile(r'(?<=[\.\!\?…\;])\s+|\n+')

def split_into_chunks(text, max_chars=400):
    """
    Split text into sentence-ish chunks. max_chars is a soft target for chunk size.
    We split on punctuation boundaries (keeps punctuation) and then merge short
    sentences so chunks are reasonably sized.
    """
    if text is None:
        return [""]
    parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(text) if p.strip()]
    # fallback if regexp produced nothing
    if not parts:
        return [text.strip()]
    chunks = []
    current = parts[0]
    for part in parts[1:]:
        if len(current) + 1 + len(part) <= max_chars:
            current = current + " " + part
        else:
            chunks.append(current)
            current = part
    chunks.append(current)
    return chunks

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Translate 'befund_text_plain' German->English in JSONL files.")
    p.add_argument("--input-dir", "-i", required=True, help="Input directory containing .jsonl files.")
    p.add_argument("--output-dir", "-o", required=True, help="Output directory to write translated .jsonl files.")
    p.add_argument("--model", "-m", default="Helsinki-NLP/opus-mt-de-en", help="Hugging Face model name (default: Helsinki-NLP/opus-mt-de-en).")
    p.add_argument("--batch-size", "-b", type=int, default=4, help="Number of texts per translation batch (applies to top-level objects; internal chunking will batch chunks).")
    p.add_argument("--device", "-d", type=str, default=None, help="Device to use: cuda, mps, cpu, or an integer index. If omitted, auto-detects (cuda -> mps -> cpu).")
    p.add_argument("--use-hf-inference-api", action="store_true", help="Use Hugging Face Inference API instead of local model. Requires --hf-token.")
    p.add_argument("--hf-token", type=str, default=None, help="Hugging Face token (needed when --use-hf-inference-api).")
    p.add_argument("--key", type=str, default="befund_text_plain", help="JSON key to translate (default: befund_text_plain).")
    p.add_argument("--max-chunk-chars", type=int, default=400, help="Soft max characters per chunk when splitting long texts.")
    return p.parse_args()

def get_jsonl_files(input_dir):
    p = Path(input_dir)
    return sorted([str(p / f) for f in os.listdir(p) if f.endswith(".jsonl")])

# ---------- Model / pipeline loading ----------
def load_transformer_pipeline(model_name, device_arg):
    """
    Load tokenizer + model and return a transformers pipeline.
    device_arg can be None, 'cuda', 'mps', 'cpu', or an integer string.
    We'll move the model to the selected torch.device and create a pipeline.
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

    # determine torch device
    if device_arg is None:
        if torch.cuda.is_available():
            torch_device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch_device = torch.device("mps")
        else:
            torch_device = torch.device("cpu")
    else:
        arg = str(device_arg).lower()
        if arg in ("cuda", "gpu"):
            torch_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        elif arg == "mps":
            torch_device = torch.device("mps") if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else torch.device("cpu")
        elif arg == "cpu":
            torch_device = torch.device("cpu")
        else:
            # try to parse as integer device index -> treat as cuda index if possible
            try:
                idx = int(arg)
                # if cuda is available, map index to cuda device, else fallback to cpu
                if torch.cuda.is_available():
                    torch_device = torch.device(f"cuda:{idx}")
                else:
                    torch_device = torch.device("cpu")
            except Exception:
                torch_device = torch.device("cpu")

    # load model + tokenizer
    print(f"Loading model '{model_name}' (this may download weights) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # move model to device
    try:
        model.to(torch_device)
    except Exception as e:
        print("Warning: failed to .to(device) on model:", e)

    # Pipeline `device` argument expects an int device index for CUDA or -1 for CPU.
    # We already moved model to the desired torch device: pass device index 0 for CUDA, -1 otherwise.
    pipeline_device = -1
    if torch_device.type == "cuda":
        # Use first device index if CUDA; if specific cuda:X was requested, pick that index
        if ":" in str(torch_device):
            try:
                pipeline_device = int(str(torch_device).split(":")[1])
            except Exception:
                pipeline_device = 0
        else:
            pipeline_device = 0
    else:
        pipeline_device = -1

    # Create translation pipeline using the already loaded model to ensure generation runs on the model's device.
    translator = pipeline("translation", model=model, tokenizer=tokenizer, device=pipeline_device)
    # Inform what device we expect to be used:
    device_descr = str(torch_device)
    print(f"Device set to use {device_descr}")
    return translator

# ---------- HF Inference API ----------
def translate_batch_hf_api(model_name, texts, hf_token):
    """
    Uses Hugging Face Inference API. Makes one request per text (simple).
    For heavy use, consider batching server-side or using other strategies.
    """
    import requests
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    translations = []
    for t in texts:
        payload = {"inputs": t}
        resp = requests.post(API_URL, headers=headers, json=payload)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                translations.append(data[0].get("translation_text", ""))
            else:
                translations.append(str(data))
        else:
            raise RuntimeError(f"Hugging Face API returned {resp.status_code}: {resp.text}")
    return translations

# ---------- Local translation with chunking ----------
def translate_batch_local(translator, texts, max_chunk_chars=400, chunk_batch_size=32):
    """
    Translate a list of texts (top-level JSON items). For long texts, split into chunks,
    translate the flattened chunks in batches, then re-join per original entry.

    translator: transformers pipeline
    texts: list[str]
    max_chunk_chars: soft chunk size
    chunk_batch_size: how many chunks to send to the pipeline at once (tunable)
    """
    # Build flat_chunks list and a map back to originals
    flat_chunks = []
    mapping = []  # list of (start_idx, end_idx) for each original text
    for t in texts:
        if t is None:
            mapping.append((len(flat_chunks), len(flat_chunks)))
            continue
        if len(t) > max_chunk_chars:
            chunks = split_into_chunks(t, max_chars=max_chunk_chars)
        else:
            chunks = [t]
        start = len(flat_chunks)
        flat_chunks.extend(chunks)
        end = len(flat_chunks)
        mapping.append((start, end))

    # Translate flat_chunks in batches
    translations_flat = []
    if flat_chunks:
        # translator may have an internal batch_size attribute or not; respect chunk_batch_size param
        for i in range(0, len(flat_chunks), chunk_batch_size):
            batch = flat_chunks[i:i+chunk_batch_size]
            # call pipeline - pass truncation True to avoid tokenization warnings
            results = translator(batch, truncation=True)
            # Normalize results (list of dicts or list-of-lists)
            for r in results:
                if isinstance(r, list) and r:
                    r = r[0]
                if isinstance(r, dict):
                    translations_flat.append(r.get("translation_text", ""))
                else:
                    translations_flat.append(str(r))
    # Reconstruct original translations by joining their chunks
    reconstructed = []
    for (start, end) in mapping:
        if start == end:
            reconstructed.append(None)
        else:
            pieces = translations_flat[start:end]
            # Join with a single space and strip extra spacing
            joined = " ".join(pieces).replace("  ", " ").strip()
            reconstructed.append(joined)
    return reconstructed

# ---------- File processing ----------
def process_file(input_path, output_path, translator, args, use_api=False):
    key = args.key
    batch_size = args.batch_size
    max_chunk_chars = args.max_chunk_chars

    total_lines = 0
    # count lines for progress bar
    with open(input_path, "r", encoding="utf-8") as _f:
        for _ in _f:
            total_lines += 1

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout, \
         tqdm(total=total_lines, desc=f"Translating {os.path.basename(input_path)}", unit="lines") as pbar:

        buffer_texts = []
        buffer_objs = []
        for raw_line in fin:
            pbar.update(1)
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # write unchanged line and continue
                fout.write(raw_line)
                continue

            text = obj.get(key)
            if text is None:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            buffer_texts.append(text)
            buffer_objs.append(obj)

            if len(buffer_texts) >= batch_size:
                if use_api:
                    translations = translate_batch_hf_api(args.model, buffer_texts, args.hf_token)
                else:
                    translations = translate_batch_local(translator, buffer_texts, max_chunk_chars=max_chunk_chars)
                for obj_item, translated in zip(buffer_objs, translations):
                    obj_item[key] = translated
                    fout.write(json.dumps(obj_item, ensure_ascii=False) + "\n")
                buffer_texts = []
                buffer_objs = []

        # flush remaining
        if buffer_texts:
            if use_api:
                translations = translate_batch_hf_api(args.model, buffer_texts, args.hf_token)
            else:
                translations = translate_batch_local(translator, buffer_texts, max_chunk_chars=max_chunk_chars)
            for obj_item, translated in zip(buffer_objs, translations):
                obj_item[key] = translated
                fout.write(json.dumps(obj_item, ensure_ascii=False) + "\n")

# ---------- Main ----------
def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = get_jsonl_files(input_dir)
    if not files:
        print(f"No .jsonl files found in {input_dir}")
        return

    use_api = args.use_hf_inference_api
    translator = None
    if not use_api:
        try:
            translator = load_transformer_pipeline(args.model, args.device)
        except Exception as e:
            print("Failed to load local model:", e)
            print("You can retry with --use-hf-inference-api and provide --hf-token")
            raise

    for infile in files:
        fname = os.path.basename(infile)
        outfile = output_dir / fname
        print(f"Processing {fname} → {outfile} ...")
        try:
            process_file(infile, outfile, translator, args, use_api=use_api)
        except Exception as e:
            print(f"Error processing {infile}: {e}")
            raise

    print("Done. Translated files written to:", str(output_dir))

if __name__ == "__main__":
    main()
