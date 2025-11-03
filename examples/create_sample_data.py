#!/usr/bin/env python3
"""
Create minimal example input files for testing the parquet reader toolkit.

This script creates:
1. A sample parquet file with hex-encoded and RTF text
2. A sample JSONL file with plain text

Usage:
    python create_sample_data.py
"""

import polars as pl
import json
from pathlib import Path


def create_sample_parquet():
    """Create a minimal sample parquet file."""

    print("Creating sample parquet file...")

    # Sample data mimicking the expected structure:
    # - befund_schluessel: unique identifier for each report
    # - befund_text_sequenz: sequence number for text chunks
    # - befund_text: the actual text content (can be hex-encoded or RTF)

    sample_data = {
        "befund_schluessel": [1, 1, 1, 2, 2, 3],
        "befund_text_sequenz": [1, 2, 3, 1, 2, 1],
        "befund_text": [
            # Report 1: Hex-encoded German text
            "426566756e643a204e6f726d616c6572204865727a72687974686d75732e",  # "Befund: Normaler Herzrhythmus."
            "204b65696e6520417566666162c3a46c6c69676b656974656e2e",  # " Keine Auffälligkeiten."
            "20506174696e65743a204d757374657266726175",  # " Patient: Musterfrau"
            # Report 2: RTF formatted text
            "{\\rtf1 Befund: Röntgen Thorax unauffällig.}",
            " Keine pathologischen Veränderungen.",
            # Report 3: Plain text
            "Laborwerte im Normbereich. Patient: Max Mustermann, geb. 15.03.1980."
        ]
    }

    df = pl.DataFrame(sample_data)
    output_path = Path("examples/sample_data.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)

    print(f"✓ Created: {output_path}")
    return output_path


def create_sample_jsonl():
    """Create a minimal sample JSONL file."""

    print("Creating sample JSONL file...")

    # Sample JSONL data for testing anonymization, translation, and embedding
    sample_records = [
        {
            "befund_schluessel": "001",
            "befund_text_plain": "Befund: Normaler Herzrhythmus. Keine Auffälligkeiten. Patient: Anna Müller, geboren am 12.05.1975."
        },
        {
            "befund_schluessel": "002",
            "befund_text_plain": "Röntgen Thorax: Unauffällige Darstellung der Lunge. Herz normal groß. Patient: Hans Schmidt, geb. 23.08.1960."
        },
        {
            "befund_schluessel": "003",
            "befund_text_plain": "Laborwerte: Alle Parameter im Normbereich. Cholesterin 180 mg/dl, Glukose 95 mg/dl."
        },
        {
            "befund_schluessel": "004",
            "befund_text_plain": "MRT Schädel vom 15. Januar 2024: Keine Anzeichen für Blutungen oder Raumforderungen."
        },
        {
            "befund_schluessel": "005",
            "befund_text_plain": "Ultraschall Abdomen: Leber, Gallenblase und Nieren unauffällig. Patient: Maria Fischer."
        }
    ]

    output_path = Path("examples/sample_data.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for record in sample_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"✓ Created: {output_path}")
    return output_path


def create_readme():
    """Create a README for the examples directory."""

    readme_content = """# Example Data

This directory contains minimal example input files for testing the parquet reader toolkit.

## Files

### `sample_data.parquet`
A minimal parquet file containing 3 medical reports with:
- Hex-encoded text
- RTF-formatted text
- Plain text
- Names and dates (for testing anonymization)

**Usage:**
```bash
# Preview the data
python src/read_parquet.py examples/sample_data.parquet

# Export to JSONL
python src/read_parquet.py examples/sample_data.parquet --format jsonl --output-dir output
```

### `sample_data.jsonl`
A JSONL file with 5 sample medical reports in German, containing:
- Patient names
- Dates
- Medical terminology

**Usage:**
```bash
# Anonymize
python src/anonymize_text.py -i examples -o output/anonymized

# Translate to English
python src/translate_jsonl.py -i examples -o output/translated

# Generate embeddings
python src/embed_reports.py -i examples -b 2

# Cluster embeddings
python src/cluster_embeddings.py -i examples --k 3
```

## Creating Your Own Sample Data

Run the creation script:
```bash
python examples/create_sample_data.py
```

This will regenerate both sample files.
"""

    output_path = Path("examples/README.md")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(f"✓ Created: {output_path}")


if __name__ == "__main__":
    print("="*60)
    print("Creating sample data files...")
    print("="*60 + "\n")

    parquet_path = create_sample_parquet()
    jsonl_path = create_sample_jsonl()
    create_readme()

    print("\n" + "="*60)
    print("SUCCESS! Sample files created.")
    print("="*60)
    print("\nYou can now test the scripts with:")
    print(f"  python src/read_parquet.py {parquet_path}")
    print(f"  python src/anonymize_text.py -i examples -o output")
