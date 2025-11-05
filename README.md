# Parquet Reader & Medical Report Processing Toolkit

A comprehensive Python toolkit for processing, analyzing, and anonymizing medical reports stored in Parquet format. This toolkit provides end-to-end functionality for decoding, translating, embedding, clustering, and visualizing medical text data.

## Features

### Core Functionality

- **Parquet Processing** (`read_parquet.py`)
  - Read and aggregate parquet files containing split medical reports
  - Decode hex-encoded and RTF-formatted text fields
  - Export to multiple formats: JSON, JSONL, CSV
  - Support for batch processing of multiple files

- **Text Anonymization** (`anonymize_text.py`)
  - German NER-based anonymization using Flair
  - Redact personal names (PERSON entities)
  - Date redaction with regex patterns
  - Chunked processing for long texts
  - Batch processing with progress tracking

- **Translation** (`translate_jsonl.py`)
  - German to English translation
  - Support for both local models and Hugging Face Inference API
  - Intelligent text chunking for long documents
  - Multi-GPU support (CUDA/MPS/CPU)
  - Batch processing for efficiency

- **Text Embedding** (`embed_reports.py`)
  - Generate embeddings using SentenceTransformers
  - Support for various embedding models (default: google/embeddinggemma-300m)
  - Automatic device selection (MPS/CUDA/CPU)
  - Batch processing with progress tracking
  - NPZ output format for efficient storage

- **Clustering & Visualization** (`cluster_embeddings.py`)
  - KMeans clustering of text embeddings
  - t-SNE visualization with automatic dimensionality reduction
  - Topic extraction using TF-IDF
  - Representative sample selection per cluster
  - Support for both per-file and merged clustering modes

- **Interactive Comparison** (`streamlit_report_comparison.py`)
  - Side-by-side comparison of different report versions
  - Diff visualization
  - Filter and search capabilities
  - Export functionality

## Installation

### Requirements

Python 3.8+ is required. Install dependencies:

```bash
pip install -r requirements.txt
```

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/parquet_reader.git
cd parquet_reader

# Install dependencies
pip install -r requirements.txt

# Run the minimal example
python examples/minimal_example.py
```

## Usage

### 1. Reading Parquet Files

```bash
# Preview mode (default)
python src/read_parquet.py /path/to/file.parquet

# Export to JSONL
python src/read_parquet.py /path/to/file.parquet --format jsonl --output-dir ./output

# Process entire directory
python src/read_parquet.py /path/to/parquet_dir/ --format json_shared --output-dir ./output
```

**Supported formats:**
- `preview`: Display first N rows in console
- `json_individual`: One JSON file per record
- `json_shared`: Single JSON file with all records
- `jsonl`: JSON Lines format (one JSON per line)
- `csv`: Comma-separated values

### 2. Anonymizing Text

```bash
# Anonymize JSONL files (names and dates)
python src/anonymize_text.py \
  --input-dir ./data \
  --output-dir ./anonymized \
  --preserve-original

# Adjust batch size for performance
python src/anonymize_text.py -i ./data -o ./anonymized --batch-size 64
```

### 3. Translating Reports

#### Option A: Using DeepL API (Recommended for Quality)

```bash
# Translate German to English using DeepL
python src/translate_jsonl_deepl.py \
  --input-dir ./data \
  --output-dir ./translated \
  --api-key YOUR_DEEPL_API_KEY

# Or set environment variable
export DEEPL_API_KEY="your-api-key-here"
python src/translate_jsonl_deepl.py -i ./data -o ./translated

# Preserve original text
python src/translate_jsonl_deepl.py \
  -i ./data \
  -o ./translated \
  --api-key YOUR_KEY \
  --preserve-original

# Use different target language
python src/translate_jsonl_deepl.py \
  -i ./data \
  -o ./translated \
  --api-key YOUR_KEY \
  --target-lang EN-GB  # British English
```

**Get your DeepL API key at:** https://www.deepl.com/pro-api

#### Option B: Using Local Hugging Face Models

```bash
# Translate German to English (local model)
python src/translate_jsonl.py \
  --input-dir ./data \
  --output-dir ./translated \
  --device cuda

# Use Hugging Face Inference API
python src/translate_jsonl.py \
  --input-dir ./data \
  --output-dir ./translated \
  --use-hf-inference-api \
  --hf-token YOUR_TOKEN
```

### 4. Generating Embeddings

```bash
# Generate embeddings for JSONL files
python src/embed_reports.py \
  --input-dir ./data \
  --batch-size 64 \
  --model google/embeddinggemma-300m

# Use different model
python src/embed_reports.py \
  -i ./data \
  -m sentence-transformers/all-MiniLM-L6-v2
```

### 5. Clustering & Visualization

```bash
# Cluster embeddings (per-file mode)
python src/cluster_embeddings.py \
  --input-dir ./embeddings \
  --k 10 \
  --reps 10

# Merge all embeddings and cluster together
python src/cluster_embeddings.py \
  --input-dir ./embeddings \
  --merge \
  --k 15 \
  --reps 5
```

**Outputs:**
- `*.clusters.jsonl`: Representative samples with cluster assignments
- `*.tsne.png`: t-SNE visualization
- `*.topics.json`: Top keywords per cluster

### 6. Interactive Comparison

```bash
# Launch Streamlit app
streamlit run src/streamlit_report_comparison.py
```

Then upload JSONL files or specify a local directory to compare different versions of reports side-by-side.

## Project Structure

```
parquet_reader/
├── src/
│   ├── read_parquet.py                   # Main parquet processing
│   ├── anonymize_text.py                 # Text anonymization (Flair NER)
│   ├── translate_jsonl.py                # Translation (HuggingFace models)
│   ├── translate_jsonl_deepl.py          # Translation (DeepL API)
│   ├── embed_reports.py                  # Text embedding generation
│   ├── cluster_embeddings.py             # Clustering & visualization
│   └── streamlit_report_comparison.py    # Interactive UI
├── examples/
│   ├── create_sample_data.py             # Generate sample input files
│   ├── sample_data.parquet               # Sample parquet file
│   ├── sample_data.jsonl                 # Sample JSONL file
│   └── README.md                         # Examples documentation
├── requirements.txt                      # Python dependencies
├── pyproject.toml                        # Package configuration
├── README.md                             # This file
├── LICENSE                               # License information
└── .gitignore                            # Git ignore rules
```

## Typical Workflow

```bash
# 1. Extract and decode parquet files
python src/read_parquet.py data.parquet --format jsonl --output-dir ./step1

# 2. Anonymize sensitive information
python src/anonymize_text.py -i ./step1 -o ./step2 --preserve-original

# 3. Translate to English (optional - using DeepL for best quality)
python src/translate_jsonl_deepl.py -i ./step2 -o ./step3 --api-key YOUR_KEY
# Or use local HuggingFace model: python src/translate_jsonl.py -i ./step2 -o ./step3

# 4. Generate embeddings
python src/embed_reports.py -i ./step3 -b 32

# 5. Cluster and visualize
python src/cluster_embeddings.py -i ./step3 --k 10 --merge
```

## Configuration

### Device Selection

Most scripts support automatic device selection with override options:

```bash
# Auto-detect (CUDA → MPS → CPU)
python src/translate_jsonl.py -i ./data -o ./out

# Force specific device
python src/translate_jsonl.py -i ./data -o ./out --device cuda
python src/translate_jsonl.py -i ./data -o ./out --device mps
python src/translate_jsonl.py -i ./data -o ./out --device cpu
```

### Batch Sizes

Adjust batch sizes based on your hardware:

- Small GPU (< 8GB): `--batch-size 8` or `--batch-size 16`
- Medium GPU (8-16GB): `--batch-size 32`
- Large GPU (> 16GB): `--batch-size 64` or higher

## Dependencies

Key dependencies include:

- `polars`: Fast DataFrame operations
- `sentence-transformers`: Text embeddings
- `flair`: Named Entity Recognition
- `transformers`: Translation models
- `scikit-learn`: Clustering and dimensionality reduction
- `streamlit`: Interactive web interface
- `matplotlib`: Visualization
- `tqdm`: Progress bars

See `requirements.txt` for complete list.

## Examples

### Minimal Example

```python
from src.read_parquet import read_and_decode_parquet

# Read and preview parquet file
df = read_and_decode_parquet(
    parquet_path="data.parquet",
    output_format="preview",
    preview_rows=5
)

# Export to JSONL
read_and_decode_parquet(
    parquet_path="data.parquet",
    output_format="jsonl",
    output_dir="./output"
)
```

See `examples/minimal_example.py` for a complete working example.
