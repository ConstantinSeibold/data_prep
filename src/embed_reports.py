#!/usr/bin/env python3
"""
Embed JSONL files in a folder.

For each file F.jsonl in the input directory:
 - read lines as JSON objects
 - extract 'befund_schluessel' -> id and 'befund_text_plain' -> text
 - embed texts in batches using SentenceTransformer
 - save results to F.emb.npz with arrays: ids (object/str) and embeddings (float32)

Usage:
    python embed_jsonl_folder.py --input-dir /path/to/jsonl --batch-size 64
"""

import argparse
import os
import time
import glob
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer


def choose_device():
    # Prefer MPS on macOS, then CUDA, then CPU
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync_device(device: torch.device):
    # Best-effort synchronization for accurate timings
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            # older torch builds might not expose this — ignore
            pass


def embed_file(
    model: SentenceTransformer,
    input_path: Path,
    output_path: Path,
    batch_size: int = 64,
    show_inner_tqdm: bool = True,
):
    """
    Read input_path (.jsonl), extract ids and texts, embed texts in batches, and save to output_path (.npz).
    """
    t_start_file = time.perf_counter()

    ids: List[str] = []
    embeddings_list: List[np.ndarray] = []

    # First: count lines for tqdm (so we can show progress). This reads file once quickly.
    with input_path.open("r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    # Re-open for actual processing
    with input_path.open("r", encoding="utf-8") as f:
        # We'll collect texts and ids per batch
        batch_texts: List[str] = []
        batch_ids: List[str] = []

        pbar = tqdm(f, total=total_lines, desc=f"Lines {input_path.name}", unit="lines")
        # inner progress bar for batches optional
        for line in pbar:
            line = line.strip()
            if not line:
                pbar.update(0)
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # skip/ warn
                pbar.write(f"Warning: could not parse JSON line in {input_path.name}")
                continue

            # Extract fields
            key = obj.get("befund_schluessel")
            text = obj.get("befund_text_plain")

            if key is None or text is None:
                # skip lines that don't have required keys
                pbar.write(f"Skipping line: missing key/text in {input_path.name}")
                continue

            batch_ids.append(str(key))
            batch_texts.append(str(text))

            # if batch full, encode
            if len(batch_texts) >= batch_size:
                # encode batch
                t0 = time.perf_counter()
                # ensure the outputs are tensors on model device
                # SentenceTransformer encode API: convert_to_tensor=True and device will place them
                batch_embeddings = model.encode_document(batch_texts, convert_to_tensor=True)
                sync_device(batch_embeddings.device if hasattr(batch_embeddings, "device") else choose_device())
                t1 = time.perf_counter()

                # move to CPU numpy
                if isinstance(batch_embeddings, torch.Tensor):
                    emb_np = batch_embeddings.detach().cpu().numpy().astype(np.float32)
                else:
                    emb_np = np.asarray(batch_embeddings, dtype=np.float32)

                embeddings_list.append(emb_np)
                ids.extend(batch_ids)

                # reset batch
                batch_texts = []
                batch_ids = []

                pbar.set_postfix({"last_batch_s": f"{(t1 - t0):.3f}"})

            # advance tqdm by 1 line (we used file iterator)
            # tqdm auto-advances when iterating over f; don't call update manually here.

        # after loop, flush remaining
        if batch_texts:
            t0 = time.perf_counter()
            batch_embeddings = model.encode_document(batch_texts, convert_to_tensor=True)
            sync_device(batch_embeddings.device if hasattr(batch_embeddings, "device") else choose_device())
            t1 = time.perf_counter()
            if isinstance(batch_embeddings, torch.Tensor):
                emb_np = batch_embeddings.detach().cpu().numpy().astype(np.float32)
            else:
                emb_np = np.asarray(batch_embeddings, dtype=np.float32)
            embeddings_list.append(emb_np)
            ids.extend(batch_ids)
            # small progress print
            pbar.write(f"Flushed final batch ({len(batch_texts)} items) in {(t1 - t0):.3f}s")

        pbar.close()

    # concatenate embeddings
    if embeddings_list:
        all_embeddings = np.vstack(embeddings_list)
    else:
        all_embeddings = np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)

    # Save: same base name, extension .emb.npz
    np.savez_compressed(
        output_path,
        ids=np.array(ids, dtype=object),
        embeddings=all_embeddings,
        count=len(ids),
    )

    t_end_file = time.perf_counter()
    print(f"Saved {len(ids)} embeddings to {output_path} (time: {t_end_file - t_start_file:.3f}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", "-i", required=True, help="Directory containing .jsonl files")
    parser.add_argument("--pattern", "-p", default="*.jsonl", help="Glob pattern for jsonl files (default: *.jsonl)")
    parser.add_argument("--batch-size", "-b", type=int, default=64, help="Batch size for encoding")
    parser.add_argument("--model", "-m", type=str, default="google/embeddinggemma-300m", help="SentenceTransformer model")
    parser.add_argument("--output-ext", default=".emb.npz", help="Extension for output files (default .emb.npz)")
    parser.add_argument("--output-path", default="out/", help="directory for output files (default out/)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory {input_dir} not found or not a directory")

    device = choose_device()
    print("Using device:", device)

    t0 = time.perf_counter()
    model = SentenceTransformer(args.model, device=device)
    sync_device(device)
    t1 = time.perf_counter()
    print(f"Loaded model {args.model} in {t1 - t0:.3f}s")
    print("Embedding dimension:", model.get_sentence_embedding_dimension())

    pattern = str(input_dir / args.pattern)
    files = sorted(glob.glob(pattern))
    if not files:
        print("No files found matching", pattern)
        return

    # Iterate files with tqdm
    for file_path in tqdm(files, desc="Files", unit="file"):
        p = Path(file_path)
        out_p = p.with_suffix("")  # remove suffix, e.g. 'data.jsonl' -> 'data'
        out_name = str(out_p) + args.output_ext
        embed_file(model, p, Path(out_name), batch_size=args.batch_size)

    print("All done.")


if __name__ == "__main__":
    main()
