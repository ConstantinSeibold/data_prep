import polars as pl
from striprtf.striprtf import rtf_to_text
import json
import os
from pathlib import Path
from typing import Literal, List
import glob


def find_parquet_files(path: str) -> List[str]:
    """
    Find all parquet files in a directory or return single file path.
    
    Parameters:
    -----------
    path : str
        Path to a parquet file or directory containing parquet files
        
    Returns:
    --------
    List[str] : List of parquet file paths
    """
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        parquet_files = glob.glob(os.path.join(path, "*.parquet"))
        if not parquet_files:
            raise ValueError(f"No parquet files found in directory: {path}")
        return sorted(parquet_files)
    else:
        raise ValueError(f"Path does not exist: {path}")


def process_single_parquet(parquet_path: str) -> pl.DataFrame:
    """
    Process a single parquet file: aggregate, decode hex and RTF.
    
    Parameters:
    -----------
    parquet_path : str
        Path to the input parquet file
        
    Returns:
    --------
    pl.DataFrame : The processed dataframe
    """
    # Read the parquet file
    print(f"\n  Reading: {os.path.basename(parquet_path)}")
    df_raw = pl.read_parquet(parquet_path)
    print(f"    Loaded {len(df_raw)} rows")
    
    # Step 1: Aggregate by befund_schluessel
    df_aggregated = (
        df_raw
        .sort("befund_text_sequenz")
        .group_by("befund_schluessel")
        .agg(pl.col("befund_text").str.join(""))
    )
    print(f"    Aggregated to {len(df_aggregated)} unique befunde")
    
    # Step 2: Decode hex
    def decode_hex_to_string(text: str) -> str:
        """Decode hex string to UTF-8 string, or return as-is if not hex."""
        if not text:
            return text
        
        sample = text[:100].replace(" ", "").replace("\n", "").replace("\t", "")
        
        if not all(c in '0123456789abcdefABCDEF' for c in sample if c):
            return text
        
        try:
            hex_clean = text.replace(" ", "").replace("\n", "").replace("\t", "")
            
            if not all(c in '0123456789abcdefABCDEF' for c in hex_clean):
                return text
            
            bytes_data = bytes.fromhex(hex_clean)
            
            try:
                return bytes_data.decode('utf-8')
            except UnicodeDecodeError:
                return bytes_data.decode('latin-1', errors='ignore')
        except Exception:
            return text
    
    df_decoded = df_aggregated.with_columns(
        pl.col("befund_text")
        .map_elements(decode_hex_to_string, return_dtype=pl.String)
        .alias("befund_text_decoded")
    )
    
    # Step 3: Decode RTF
    def safe_rtf_to_text(text: str) -> str:
        """Safely decode RTF to plain text."""
        try:
            if text and (text.startswith("{\\rtf") or "\\rtf" in text[:50]):
                return rtf_to_text(text)
            else:
                return text
        except Exception:
            return text
    
    df_final = df_decoded.with_columns(
        pl.col("befund_text_decoded")
        .map_elements(safe_rtf_to_text, return_dtype=pl.String)
        .alias("befund_text_plain")
    )
    
    print(f"    Decoding complete!")
    
    return df_final


def read_and_decode_parquet(
    parquet_path: str,
    output_format: Literal["json_individual", "json_shared", "jsonl", "csv", "preview"] = "preview",
    output_dir: str = "./output",
    preview_rows: int = 10,
    combine_files: bool = True
):
    """
    Reads parquet file(s), aggregates by befund_schluessel, decodes hex and RTF-encoded text fields,
    and exports to various formats.
    
    Parameters:
    -----------
    parquet_path : str
        Path to a parquet file OR directory containing parquet files
    output_format : str
        One of: "json_individual", "json_shared", "jsonl", "csv", "preview"
        - json_individual: Creates one JSON file per befund_schluessel
        - json_shared: Creates a single JSON file with all befunde
        - jsonl: Creates JSON Lines file (one JSON object per line)
        - csv: Exports to CSV format
        - preview: Shows first N rows in console
    output_dir : str
        Directory where output files will be saved (for json/csv modes)
    preview_rows : int
        Number of rows to show in preview mode (default: 10)
    combine_files : bool
        If True, combine all parquet files into one output.
        If False, create separate outputs for each input file (default: True)
    
    Returns:
    --------
    pl.DataFrame or List[pl.DataFrame] : The decoded dataframe(s)
    """
    
    # Find all parquet files
    parquet_files = find_parquet_files(parquet_path)
    
    print(f"\n{'='*80}")
    print(f"Found {len(parquet_files)} parquet file(s) to process")
    print(f"{'='*80}")
    
    # Process all files
    processed_dfs = []
    for pq_file in parquet_files:
        df = process_single_parquet(pq_file)
        processed_dfs.append(df)
    
    # Combine or keep separate
    if combine_files and len(processed_dfs) > 1:
        print(f"\nCombining {len(processed_dfs)} dataframes...")
        df_final = pl.concat(processed_dfs)
        print(f"Combined dataframe has {len(df_final)} total befunde")
    else:
        if len(processed_dfs) == 1:
            df_final = processed_dfs[0]
        else:
            df_final = processed_dfs  # Keep as list for separate processing
    
    # Handle different output formats
    if output_format == "preview":
        print(f"\n{'='*80}")
        print(f"PREVIEW - First {preview_rows} befunde:")
        print(f"{'='*80}\n")
        
        # If multiple separate dataframes, preview the first one
        preview_df = df_final if isinstance(df_final, pl.DataFrame) else df_final[0]
        
        for row in preview_df.head(preview_rows).iter_rows(named=True):
            print(f"Befund Schluessel: {row['befund_schluessel']}")
            print(f"Encoded (first 100 chars): {row['befund_text'][:100]}...")
            print(f"Decoded (first 100 chars): {row['befund_text_decoded'][:100]}...")
            print(f"Plain text (first 200 chars): {row['befund_text_plain'][:200]}...")
            print("-" * 80)
        
        return df_final
        
    elif output_format == "json_individual":
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Handle single or multiple dataframes
        dfs_to_process = [df_final] if isinstance(df_final, pl.DataFrame) else df_final
        
        total_written = 0
        for df in dfs_to_process:
            records = df.to_dicts()
            
            for record in records:
                filename = f"befund_{record['befund_schluessel']}.json"
                filepath = os.path.join(output_dir, filename)
                
                output_record = {
                    'befund_schluessel': record['befund_schluessel'],
                    'befund_text_plain': record['befund_text_plain']
                }
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(output_record, f, ensure_ascii=False, indent=2)
                
                total_written += 1
        
        print(f"\n✓ Successfully wrote {total_written} individual JSON files to: {output_dir}")
        
    elif output_format == "json_shared":
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Handle single or multiple dataframes
        dfs_to_process = [df_final] if isinstance(df_final, pl.DataFrame) else df_final
        
        all_records = []
        for df in dfs_to_process:
            records = [
                {
                    'befund_schluessel': row['befund_schluessel'],
                    'befund_text_plain': row['befund_text_plain']
                }
                for row in df.iter_rows(named=True)
            ]
            all_records.extend(records)
        
        output_path = os.path.join(output_dir, parquet_path.split('/')[-1].replace('.parquet',"")+".json")
        print(f"\nWriting shared JSON file to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_records, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Successfully wrote {len(all_records)} befunde to shared JSON file")
        
    elif output_format == "jsonl":
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Handle single or multiple dataframes
        if isinstance(df_final, pl.DataFrame):
            # Single combined file
            output_path = os.path.join(output_dir, parquet_path.split('/')[-1].replace('.parquet',"")+".jsonl")
            print(f"\nWriting JSONL file to: {output_path}")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for row in df_final.iter_rows(named=True):
                    record = {
                        'befund_schluessel': row['befund_schluessel'],
                        'befund_text_plain': row['befund_text_plain']
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            print(f"✓ Successfully wrote {len(df_final)} befunde to JSONL file")
        else:
            # Separate files
            total_written = 0
            for idx, df in enumerate(df_final):
                output_path = os.path.join(output_dir, f"befunde_{idx+1}.jsonl")
                print(f"\nWriting JSONL file to: {output_path}")
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    for row in df.iter_rows(named=True):
                        record = {
                            'befund_schluessel': row['befund_schluessel'],
                            'befund_text_plain': row['befund_text_plain']
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                
                total_written += len(df)
            
            print(f"✓ Successfully wrote {total_written} befunde to {len(df_final)} JSONL files")
        
    elif output_format == "csv":
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Handle single or multiple dataframes
        if isinstance(df_final, pl.DataFrame):
            # Single combined file
            df_export = df_final.select(['befund_schluessel', 'befund_text_plain'])
            output_path = os.path.join(output_dir, "befunde_export.csv")
            print(f"\nWriting CSV file to: {output_path}")
            df_export.write_csv(output_path)
            print(f"✓ Successfully wrote {len(df_export)} befunde to CSV file")
        else:
            # Separate files
            total_written = 0
            for idx, df in enumerate(df_final):
                df_export = df.select(['befund_schluessel', 'befund_text_plain'])
                output_path = os.path.join(output_dir, f"befunde_export_{idx+1}.csv")
                print(f"\nWriting CSV file to: {output_path}")
                df_export.write_csv(output_path)
                total_written += len(df_export)
            
            print(f"✓ Successfully wrote {total_written} befunde to {len(df_final)} CSV files")
    
    return df_final


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Read parquet file(s) with split befund_text, aggregate, decode hex and RTF, and export"
    )
    parser.add_argument(
        "parquet_path",
        help="Path to a parquet file OR directory containing parquet files"
    )
    parser.add_argument(
        "--format",
        choices=["json_individual", "json_shared", "jsonl", "csv", "preview"],
        default="preview",
        help="Output format (default: preview)"
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Output directory for exported files (default: ./output)"
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=10,
        help="Number of rows to show in preview mode (default: 10)"
    )
    parser.add_argument(
        "--no-combine",
        action="store_true",
        help="Keep output files separate instead of combining (only for multiple input files)"
    )
    
    args = parser.parse_args()
    
    # Run the export
    read_and_decode_parquet(
        parquet_path=args.parquet_path,
        output_format=args.format,
        output_dir=args.output_dir,
        preview_rows=args.preview_rows,
        combine_files=not args.no_combine
    )