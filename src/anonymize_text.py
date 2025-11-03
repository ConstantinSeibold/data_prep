#!/usr/bin/env python3
"""
anonymize_jsonl_flair_names_dates.py

Anonymize names (PERSON) and dates in `befund_text_plain` using Flair's
`flair/ner-german-large` and date-focused regex rules.

Usage:
  pip install flair tqdm
  python anonymize_jsonl_flair_names_dates.py -i samples -o anonymized_samples --preserve-original

Only PERSON entities from Flair are replaced with [NAME_REDACTED]. Dates are
replaced with [DATE_REDACTED] using regex patterns. Other entity types
(ORG/LOC/MISC) are left untouched.
"""

import os
import re
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# ---------- chunk splitting ----------
SENTENCE_SPLIT_RE = re.compile(r'(?<=[\.\!\?…\;])\s+|\n+')

def split_into_chunks(text, max_chars=400):
    if text is None:
        return [""]
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

# ---------- date-only regex redaction rules ----------
DATE_MONTHS = r"(?:Januar|Februar|März|Mrz|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember|Jan|Feb|Mär|Apr|Jun|Jul|Aug|Sep|Okt|Nov|Dez)"
DATE_REGEX_PATTERNS = [
    # numeric dates like 12.03.1980 or 12-03-80 or 12/03/1980
    (re.compile(r"\b\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}\b"), "[DATE_REDACTED]"),
    # textual dates like '12. März 1980' or '12 März 1980'
    (re.compile(r"\b\d{1,2}\s+" + DATE_MONTHS + r"(?:\s+\d{2,4})?\b", flags=re.IGNORECASE), "[DATE_REDACTED]"),
    # German birthdate labels: "Geboren: 12.03.1980", "Geburtsdatum: ..."
    (re.compile(r"\b(Geburtsdatum|geburtsdatum|geboren|geb\.)\s*[:\-]?\s*\S.{0,40}\b", flags=re.IGNORECASE), "[DATE_REDACTED]"),
]

def apply_date_redactions(text):
    if not text:
        return text
    redacted = text
    for pattern, placeholder in DATE_REGEX_PATTERNS:
        redacted = pattern.sub(placeholder, redacted)
    return redacted

# ---------- Flair-based anonymization (PERSON only) ----------

def is_person_label(label):
    if not label:
        return False
    lab = label.upper()
    return "PER" in lab or "PERSON" in lab or "NAME" in lab


def anonymize_text_with_flair_names(tagger, text, max_chunk_chars=400, mini_batch_size=16):
    """
    - Pre-apply date regex redactions only
    - Chunk text
    - Run Flair NER and replace only PERSON spans with [NAME_REDACTED]
    - Use token offsets when available, fallback to first substring replacement
    """
    if text is None:
        return None

    # apply date redaction first
    pre = apply_date_redactions(text)

    chunks = split_into_chunks(pre, max_chars=max_chunk_chars)
    anonymized_chunks = []

    try:
        from flair.data import Sentence
    except Exception as e:
        raise RuntimeError("Flair is required (pip install flair). Import failed: " + str(e))

    for i in range(0, len(chunks), mini_batch_size):
        sub = chunks[i:i+mini_batch_size]
        sentences = [Sentence(s) for s in sub]
        tagger.predict(sentences, mini_batch_size=mini_batch_size)

        for sent_obj, chunk_text in zip(sentences, sub):
            spans = sent_obj.get_spans('ner')  # grouped entities
            if not spans:
                anonymized_chunks.append(chunk_text)
                continue

            # collect PERSON replacements only
            replacements = []
            for span in spans:
                # determine label string
                label = None
                try:
                    lbl = span.get_label('ner')
                    if lbl:
                        label = lbl.value
                except Exception:
                    # older/flair versions may differ; try span.label
                    label = getattr(span, 'label', None)

                if not is_person_label(label):
                    continue  # skip non-person entities

                # try token offsets
                try:
                    tokens = span.tokens
                    if tokens and hasattr(tokens[0], 'start_pos') and hasattr(tokens[-1], 'end_pos'):
                        s = tokens[0].start_pos
                        e = tokens[-1].end_pos
                        if s is None or e is None:
                            raise AttributeError('token positions None')
                        replacements.append((s, e, '[NAME_REDACTED]'))
                        continue
                except Exception:
                    pass

                # fallback: substring replacement (first occurrence)
                span_text = span.text
                idx = chunk_text.find(span_text)
                if idx >= 0:
                    replacements.append((idx, idx + len(span_text), '[NAME_REDACTED]'))
                else:
                    # if no match, ignore
                    continue

            # apply replacements in reverse order
            replacements_sorted = sorted(replacements, key=lambda x: x[0], reverse=True)
            chunk_chars = chunk_text
            for s, e, ph in replacements_sorted:
                if 0 <= s < e <= len(chunk_chars):
                    chunk_chars = chunk_chars[:s] + ph + chunk_chars[e:]
            anonymized_chunks.append(chunk_chars)

    result = " ".join([c.strip() for c in anonymized_chunks]).replace("  ", " ").strip()
    # final pass for dates (in case chunking changed boundaries)
    result = apply_date_redactions(result)
    return result

# ---------- JSONL processing ----------

def get_jsonl_files(input_dir):
    p = Path(input_dir)
    return sorted([str(p / f) for f in os.listdir(p) if f.endswith('.jsonl')])


def process_file(infile, outfile, tagger, args):
    key = args.key
    preserve_original = args.preserve_original
    out_key = args.output_key or key
    batch_size = args.batch_size

    total = 0
    with open(infile, 'r', encoding='utf-8') as f:
        for _ in f:
            total += 1

    with open(infile, 'r', encoding='utf-8') as fin, \
         open(outfile, 'w', encoding='utf-8') as fout, \
         tqdm(total=total, desc=f"Anonymizing {os.path.basename(infile)}", unit='lines') as pbar:

        buffer_texts = []
        buffer_objs = []
        for raw in fin:
            pbar.update(1)
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                fout.write(raw)
                continue

            text = obj.get(key)
            if text is None:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            buffer_texts.append(text)
            buffer_objs.append(obj)

            if len(buffer_texts) >= batch_size:
                anonymized = [anonymize_text_with_flair_names(tagger, t, max_chunk_chars=args.max_chunk_chars, mini_batch_size=args.mini_batch) for t in buffer_texts]
                for o, a in zip(buffer_objs, anonymized):
                    if preserve_original:
                        o[f"{key}_de"] = o.get(key)
                        o[out_key] = a
                    else:
                        o[out_key] = a
                    fout.write(json.dumps(o, ensure_ascii=False) + "\n")
                buffer_texts = []
                buffer_objs = []

        if buffer_texts:
            anonymized = [anonymize_text_with_flair_names(tagger, t, max_chunk_chars=args.max_chunk_chars, mini_batch_size=args.mini_batch) for t in buffer_texts]
            for o, a in zip(buffer_objs, anonymized):
                if preserve_original:
                    o[f"{key}_de"] = o.get(key)
                    o[out_key] = a
                else:
                    o[out_key] = a
                fout.write(json.dumps(o, ensure_ascii=False) + "\n")

# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", "-i", required=True)
    p.add_argument("--output-dir", "-o", required=True)
    p.add_argument("--model", "-m", default="flair/ner-german-large", help="Flair NER model identifier")
    p.add_argument("--batch-size", "-b", type=int, default=32, help="How many JSON lines to process per anonymization batch")
    p.add_argument("--mini-batch", type=int, default=64, help="Flair mini-batch size when predicting many chunks")
    p.add_argument("--max-chunk-chars", type=int, default=400, help="Soft chunk size for long texts")
    p.add_argument("--key", type=str, default="befund_text_plain")
    p.add_argument("--output-key", type=str, default=None)
    p.add_argument("--preserve-original", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = get_jsonl_files(input_dir)
    if not files:
        print("No .jsonl files found.")
        return

    # load flair tagger
    try:
        from flair.models import SequenceTagger
        print(f"Loading Flair tagger {args.model} ...")
        tagger = SequenceTagger.load(args.model)
    except Exception as e:
        raise RuntimeError("Failed to load Flair SequenceTagger. Ensure `flair` is installed. Error: " + str(e))

    for infile in files:
        fname = os.path.basename(infile)
        outfile = output_dir / fname
        print(f"Processing {fname} -> {outfile} ...")
        process_file(infile, outfile, tagger, args)

    print("Done. Anonymized files written to:", str(output_dir))


if __name__ == "__main__":
    main()
