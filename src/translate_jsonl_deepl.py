#!/usr/bin/env python3
"""
translate_jsonl_deepl.py

Translate the value under key "befund_text_plain" from German to English
for all .jsonl files in an input directory using the DeepL API.

Features:
 - Uses DeepL API for high-quality translations
 - Handles long texts by splitting into chunks (DeepL has size limits)
 - Batches requests for efficiency
 - Progress tracking with tqdm
 - Preserves original text option

Requirements:
 - DeepL API key (get one at https://www.deepl.com/pro-api)
 - pip install deepl tqdm

Usage examples:
  # Using API key from command line
  python translate_jsonl_deepl.py -i ./input_jsonl -o ./translated_jsonl --api-key YOUR_API_KEY

  # Using API key from environment variable
  export DEEPL_API_KEY="your-api-key-here"
  python translate_jsonl_deepl.py -i ./input_jsonl -o ./translated_jsonl

  # Preserve original text in a separate field
  python translate_jsonl_deepl.py -i ./input -o ./output --api-key KEY --preserve-original

  # Translate different key
  python translate_jsonl_deepl.py -i ./input -o ./output --api-key KEY --key custom_field
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import re

# ---------- Sentence/chunk splitting helpers ----------
SENTENCE_SPLIT_RE = re.compile(r'(?<=[\.\!\?…\;])\s+|\n+')

def split_into_chunks(text, max_chars=5000):
    """
    Split text into sentence-ish chunks for DeepL API.
    DeepL has a limit per request, default is 5000 chars per chunk.
    """
    if text is None:
        return [""]

    # If text is short enough, return as-is
    if len(text) <= max_chars:
        return [text]

    parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(text) if p.strip()]
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
    p = argparse.ArgumentParser(
        description="Translate 'befund_text_plain' German->English in JSONL files using DeepL API.",
        epilog="Get your DeepL API key at https://www.deepl.com/pro-api"
    )
    p.add_argument("--input-dir", "-i", required=True, help="Input directory containing .jsonl files.")
    p.add_argument("--output-dir", "-o", required=True, help="Output directory to write translated .jsonl files.")
    p.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="DeepL API key. If not provided, will use DEEPL_API_KEY environment variable."
    )
    p.add_argument(
        "--key",
        type=str,
        default="befund_text_plain",
        help="JSON key to translate (default: befund_text_plain)."
    )
    p.add_argument(
        "--output-key",
        type=str,
        default=None,
        help="JSON key for translated text (default: same as --key)."
    )
    p.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=10,
        help="Number of JSON lines to process per batch (default: 10)."
    )
    p.add_argument(
        "--max-chunk-chars",
        type=int,
        default=5000,
        help="Max characters per chunk when splitting long texts (default: 5000)."
    )
    p.add_argument(
        "--source-lang",
        type=str,
        default="DE",
        help="Source language code (default: DE for German)."
    )
    p.add_argument(
        "--target-lang",
        type=str,
        default="EN-US",
        help="Target language code (default: EN-US for American English). Options: EN-US, EN-GB."
    )
    p.add_argument(
        "--preserve-original",
        action="store_true",
        help="Keep original text in a separate field ({key}_original)."
    )
    p.add_argument(
        "--formality",
        type=str,
        default="default",
        choices=["default", "more", "less", "prefer_more", "prefer_less"],
        help="Formality level for translation (default: default)."
    )
    return p.parse_args()


def get_jsonl_files(input_dir):
    p = Path(input_dir)
    return sorted([str(p / f) for f in os.listdir(p) if f.endswith(".jsonl")])


def translate_batch_deepl(translator, texts, source_lang, target_lang, formality, max_chunk_chars):
    """
    Translate a batch of texts using DeepL API.
    Handles chunking for long texts and rejoins them.

    Args:
        translator: deepl.Translator instance
        texts: list of strings to translate
        source_lang: source language code
        target_lang: target language code
        formality: formality setting
        max_chunk_chars: max chars per chunk

    Returns:
        list of translated strings
    """
    # Build flat list of chunks with mapping back to original texts
    flat_chunks = []
    mapping = []  # list of (start_idx, end_idx) for each original text

    for t in texts:
        if t is None or t == "":
            mapping.append((len(flat_chunks), len(flat_chunks)))
            continue

        chunks = split_into_chunks(t, max_chars=max_chunk_chars)
        start = len(flat_chunks)
        flat_chunks.extend(chunks)
        end = len(flat_chunks)
        mapping.append((start, end))

    # Translate all chunks at once (DeepL can handle multiple texts per request)
    translations_flat = []
    if flat_chunks:
        try:
            # DeepL translate_text can accept a list
            results = translator.translate_text(
                flat_chunks,
                source_lang=source_lang,
                target_lang=target_lang,
                formality=formality if formality != "default" else None
            )

            # Handle both single and multiple results
            if isinstance(results, list):
                translations_flat = [r.text for r in results]
            else:
                translations_flat = [results.text]
        except Exception as e:
            raise RuntimeError(f"DeepL translation failed: {e}")

    # Reconstruct original texts by joining their chunks
    reconstructed = []
    for (start, end) in mapping:
        if start == end:
            reconstructed.append(None)
        else:
            pieces = translations_flat[start:end]
            joined = " ".join(pieces).replace("  ", " ").strip()
            reconstructed.append(joined)

    return reconstructed


def process_file(input_path, output_path, translator, args):
    """Process a single JSONL file."""
    key = args.key
    output_key = args.output_key or key
    batch_size = args.batch_size
    preserve_original = args.preserve_original

    # Count total lines for progress bar
    total_lines = 0
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
                # Write unchanged line and continue
                fout.write(raw_line)
                continue

            text = obj.get(key)
            if text is None or text == "":
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            buffer_texts.append(text)
            buffer_objs.append(obj)

            # Process batch when full
            if len(buffer_texts) >= batch_size:
                translations = translate_batch_deepl(
                    translator,
                    buffer_texts,
                    args.source_lang,
                    args.target_lang,
                    args.formality,
                    args.max_chunk_chars
                )

                for obj_item, original, translated in zip(buffer_objs, buffer_texts, translations):
                    if preserve_original:
                        obj_item[f"{key}_original"] = original
                    obj_item[output_key] = translated
                    fout.write(json.dumps(obj_item, ensure_ascii=False) + "\n")

                buffer_texts = []
                buffer_objs = []

        # Flush remaining
        if buffer_texts:
            translations = translate_batch_deepl(
                translator,
                buffer_texts,
                args.source_lang,
                args.target_lang,
                args.formality,
                args.max_chunk_chars
            )

            for obj_item, original, translated in zip(buffer_objs, buffer_texts, translations):
                if preserve_original:
                    obj_item[f"{key}_original"] = original
                obj_item[output_key] = translated
                fout.write(json.dumps(obj_item, ensure_ascii=False) + "\n")


def main():
    args = parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get("DEEPL_API_KEY")
    if not api_key:
        raise SystemExit(
            "Error: DeepL API key not provided.\n"
            "Either use --api-key argument or set DEEPL_API_KEY environment variable.\n"
            "Get your API key at https://www.deepl.com/pro-api"
        )

    # Import deepl library
    try:
        import deepl
    except ImportError:
        raise SystemExit(
            "Error: deepl library not found.\n"
            "Install it with: pip install deepl"
        )

    # Initialize DeepL translator
    print(f"Initializing DeepL translator...")
    try:
        translator = deepl.Translator(api_key)
        # Test the connection
        usage = translator.get_usage()
        if usage.character.limit_exceeded:
            print("Warning: DeepL character limit exceeded!")
        else:
            print(f"DeepL API connected successfully.")
            if usage.character.limit:
                remaining = usage.character.limit - usage.character.count
                print(f"Character usage: {usage.character.count:,} / {usage.character.limit:,} ({remaining:,} remaining)")
    except Exception as e:
        raise SystemExit(f"Failed to initialize DeepL translator: {e}")

    # Get input files
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = get_jsonl_files(input_dir)
    if not files:
        print(f"No .jsonl files found in {input_dir}")
        return

    print(f"\nFound {len(files)} JSONL file(s) to process")
    print(f"Translating from {args.source_lang} to {args.target_lang}")
    print(f"{'='*60}\n")

    # Process each file
    for infile in files:
        fname = os.path.basename(infile)
        outfile = output_dir / fname
        print(f"Processing {fname} → {outfile} ...")

        try:
            process_file(infile, outfile, translator, args)
        except Exception as e:
            print(f"Error processing {infile}: {e}")
            raise

    # Show final usage
    try:
        usage = translator.get_usage()
        print(f"\n{'='*60}")
        print(f"Translation complete!")
        print(f"{'='*60}")
        if usage.character.limit:
            print(f"Total characters used: {usage.character.count:,} / {usage.character.limit:,}")
        print(f"Translated files written to: {output_dir}")
    except Exception:
        print(f"\nTranslated files written to: {output_dir}")


if __name__ == "__main__":
    main()
