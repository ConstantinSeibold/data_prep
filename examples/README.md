# Example Data

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
