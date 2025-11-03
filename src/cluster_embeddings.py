#!/usr/bin/env python3
"""
cluster_embeddings.py

- Reads .emb.npz files produced by the previous embedding script. Each .npz is expected
  to contain:
    ids: object array of befund_schluessel strings
    embeddings: float32 matrix (N, D)

- Modes:
    * per-file mode (default): cluster each .emb.npz individually (same behavior as before)
    * merge mode (--merge): concatenate embeddings from all .emb.npz files in the input
      directory and run a single clustering + visualization over the merged dataset.

- For each clustering run:
    * cluster embeddings with KMeans (k default 10)
    * compute representative samples per cluster (closest to cluster center)
    * attempt to find original .jsonl with befund_text_plain for topic extraction
    * compute top-n keywords per cluster using TF-IDF (if texts available)
    * compute 2D t-SNE visualization (PCA -> TSNE) and save PNG
    * export representative samples to <base>.clusters.jsonl (one JSON object per line)

Outputs (per input file or merged):
  - <base>.clusters.jsonl     (representative samples with cluster assignment)
  - <base>.tsne.png          (tsne visualization)
  - <base>.topics.json       (top keywords per cluster)

Usage examples:
    # cluster each emb file separately
    python cluster_embeddings.py --input-dir ./my_embeddings --k 10 --reps 10

    # merge all embeddings in the directory and cluster/visualize them together
    python cluster_embeddings.py --input-dir ./my_embeddings --merge --k 10 --reps 10
"""
from pathlib import Path
import argparse
import json
import numpy as np
from tqdm.auto import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import matplotlib.patches as mpatches


def load_npz(npz_path: Path):
    arr = np.load(npz_path, allow_pickle=True)
    ids = arr.get("ids")
    embeddings = arr.get("embeddings")

    if ids is None or embeddings is None:
        raise RuntimeError(f"{npz_path} missing expected 'ids' or 'embeddings' arrays")

    # filter out rows with NaNs/infs just in case
    mask = np.isfinite(embeddings).all(axis=1)
    embeddings = embeddings[mask]
    ids = ids[mask]

    ids = ids.tolist()
    embeddings = np.asarray(embeddings, dtype=np.float32)
    return ids, embeddings


def find_source_jsonl(npz_path: Path, ids_set: set, max_scan_lines=200_000):
    """
    Try to locate the matching source JSONL with befund_text_plain for a single .npz.
    """
    folder = npz_path.parent
    fname = npz_path.name
    # candidate replacement
    if fname.endswith(".emb.npz"):
        cand = folder / (fname[:-8] + ".jsonl")
        if cand.exists():
            return cand
    alt = npz_path.with_suffix(".jsonl")
    if alt.exists():
        return alt

    # scan other jsonl files, try to find one with multiple id hits
    jsonl_files = sorted(folder.glob("*.jsonl"))
    if not jsonl_files:
        return None

    min_hits = max(1, min(50, len(ids_set) // 100))
    for jf in jsonl_files:
        hits = 0
        lines_scanned = 0
        try:
            with jf.open("r", encoding="utf-8") as f:
                for line in f:
                    lines_scanned += 1
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    key = obj.get("befund_schluessel")
                    if key is not None and str(key) in ids_set:
                        hits += 1
                        if hits >= min_hits:
                            return jf
                    if lines_scanned >= max_scan_lines:
                        break
        except Exception:
            continue
    return None


def build_id_to_text_map(jsonl_path: Path, wanted_ids: set):
    """Read jsonl and return dict id->text for ids present."""
    id2text = {}
    try:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                key = obj.get("befund_schluessel")
                if key is None:
                    continue
                key = str(key)
                if key in wanted_ids and key not in id2text:
                    id2text[key] = obj.get("befund_text_plain", "")
                    if len(id2text) >= len(wanted_ids):
                        break
    except Exception:
        pass
    return id2text


def build_global_id2text_map(folder: Path, wanted_ids: set, max_scan_lines_per_file=200_000):
    """
    Scan all jsonl files in folder and return id->text map for wanted_ids.
    Stops early when all wanted_ids are found.
    """
    id2text = {}
    jsonl_files = sorted(folder.glob("*.jsonl"))
    if not jsonl_files:
        return id2text

    for jf in jsonl_files:
        lines_scanned = 0
        try:
            with jf.open("r", encoding="utf-8") as f:
                for line in f:
                    lines_scanned += 1
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    key = obj.get("befund_schluessel")
                    if key is None:
                        continue
                    key = str(key)
                    if key in wanted_ids and key not in id2text:
                        id2text[key] = obj.get("befund_text_plain", "")
                        if len(id2text) >= len(wanted_ids):
                            return id2text
                    if lines_scanned >= max_scan_lines_per_file:
                        break
        except Exception:
            continue
    return id2text


def cluster_and_export_for_arrays(ids, embeddings, idx_to_source, out_base: Path, k=10, reps_per_cluster=10, top_n_terms=10, tsne_perplexity=30, random_state=42, id2text=None):
    """
    Core routine that receives arrays (ids, embeddings) and optionally id2text map and
    writes outputs using out_base as the base filename (without extension).
    idx_to_source: list mapping each index -> source filename (or None)
    Returns a summary dict.
    """
    n, dim = embeddings.shape
    print(f"Running clustering on {n} samples (dim={dim}) -> out base: {out_base.name}")

    k_eff = min(k, n)
    if k_eff <= 0:
        raise RuntimeError("No samples to cluster")

    km = KMeans(n_clusters=k_eff, random_state=random_state, n_init="auto")
    km.fit(embeddings)
    labels = km.labels_
    centers = km.cluster_centers_

    # distances to assigned center
    assigned_center = centers[labels]
    member_diffs = embeddings - assigned_center
    member_dists = np.linalg.norm(member_diffs, axis=1)

    cluster_to_indices = defaultdict(list)
    for idx, lab in enumerate(labels):
        cluster_to_indices[int(lab)].append(idx)

    # representatives
    reps = []
    for c in range(k_eff):
        idxs = cluster_to_indices[c]
        if not idxs:
            continue
        member_embs = embeddings[idxs]
        center = centers[c]
        dists = np.linalg.norm(member_embs - center.reshape(1, -1), axis=1)
        order = np.argsort(dists)
        for rank, oi in enumerate(order[:reps_per_cluster], start=1):
            sample_idx = idxs[oi]
            reps.append({
                "cluster": int(c),
                "rank": int(rank),
                "idx_global": int(sample_idx),
                "befund_schluessel": str(ids[sample_idx]),
                "distance": float(dists[oi]),
                "source_file": idx_to_source[sample_idx],
                "befund_text_plain": (id2text.get(str(ids[sample_idx])) if id2text else None)
            })

    out_jsonl = out_base.with_suffix('.clusters.jsonl')
    with out_jsonl.open('w', encoding='utf-8') as f:
        for r in reps:
            f.write(json.dumps(r, ensure_ascii=False) + "")
    print(f"Wrote representative samples: {out_jsonl} ({len(reps)} entries)")

    # Topic extraction
    topics = {}
    if id2text:
        cluster_texts = {}
        for c in range(k_eff):
            cluster_texts[c] = []
            for idx in cluster_to_indices[c]:
                key = str(ids[idx])
                txt = id2text.get(key, "")
                cluster_texts[c].append(txt)

        corpus = []
        index_map = []
        for c in range(k_eff):
            for pos, txt in enumerate(cluster_texts[c]):
                corpus.append(txt if txt is not None else "")
                index_map.append((c, pos))

        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        try:
            X = tfidf.fit_transform(corpus)
            feature_names = tfidf.get_feature_names_out()
            cluster_rows = defaultdict(list)
            for row_idx, (c, pos) in enumerate(index_map):
                cluster_rows[c].append(row_idx)
            for c in range(k_eff):
                rows = cluster_rows.get(c, [])
                if not rows:
                    topics[c] = []
                    continue
                sub = X[rows, :]
                mean_tfidf = np.asarray(sub.mean(axis=0)).ravel()
                if np.all(mean_tfidf == 0):
                    topics[c] = []
                    continue
                top_idx = np.argsort(mean_tfidf)[::-1][:top_n_terms]
                top_terms = [feature_names[i] for i in top_idx if mean_tfidf[i] > 0]
                topics[c] = top_terms
        except Exception as e:
            print("TF-IDF failed:", e, file=sys.stderr)
            topics = {c: [] for c in range(k_eff)}
    else:
        topics = {c: [] for c in range(k_eff)}

    out_topics = out_base.with_suffix('.topics.json')
    with out_topics.open('w', encoding='utf-8') as f:
        json.dump({"k": k_eff, "topics": topics}, f, ensure_ascii=False, indent=2)
    print(f"Wrote cluster topics to {out_topics}")

    # t-SNE
    tsne_out = out_base.with_suffix('.tsne.png')
    try:
        pca_dim = min(50, embeddings.shape[1], max(2, embeddings.shape[1] // 2))
        pca = PCA(n_components=pca_dim, random_state=random_state)
        emb_pca = pca.fit_transform(embeddings)
        print(f"PCA reduced to {pca_dim} dims (explained var: {pca.explained_variance_ratio_.sum():.3f})")

        tsne = TSNE(n_components=2, perplexity=min(tsne_perplexity, max(5, n // 3)), init="pca", learning_rate="auto", random_state=random_state, verbose=1)
        emb_tsne = tsne.fit_transform(emb_pca)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(emb_tsne[:, 0], emb_tsne[:, 1], c=labels, cmap="tab10", s=10, alpha=0.8)
        plt.title(f"t-SNE of {out_base.name} (k={k_eff})")
        plt.xlabel("tsne-1")
        plt.ylabel("tsne-2")

        cmap = plt.get_cmap("tab10")
        patches = []
        for i in range(k_eff):
            color = cmap(i % cmap.N)
            tlist = topics.get(i, [])
            top_label = tlist[0] if isinstance(tlist, (list, tuple)) and len(tlist) > 0 else None
            label_text = top_label if top_label else f"cluster {i}"
            patches.append(mpatches.Patch(color=color, label=label_text))

        plt.legend(handles=patches, title="clusters", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(tsne_out, dpi=200)
        plt.close()
        print(f"Saved t-SNE plot to {tsne_out}")
    except Exception as e:
        print("t-SNE failed:", e, file=sys.stderr)

    return {
        "n": n,
        "k": k_eff,
        "representatives": len(reps),
        "topics_file": str(out_topics),
        "clusters_jsonl": str(out_jsonl),
        "tsne_png": str(tsne_out) if tsne_out.exists() else None,
    }


def cluster_and_export(npz_path: Path, out_dir: Path, k=10, reps_per_cluster=10, top_n_terms=10, tsne_perplexity=30, random_state=42):
    ids, embeddings = load_npz(npz_path)
    idx_to_source = [npz_path.name] * len(ids)

    # try to find source jsonl for this file
    ids_set = set(map(str, ids))
    source_jsonl = find_source_jsonl(npz_path, ids_set)
    id2text = {}
    if source_jsonl:
        id2text = build_id_to_text_map(source_jsonl, ids_set)

    out_base = out_dir / npz_path.name.replace('.emb.npz', '')
    return cluster_and_export_for_arrays(ids, embeddings, idx_to_source, out_base, k=k, reps_per_cluster=reps_per_cluster, top_n_terms=top_n_terms, tsne_perplexity=tsne_perplexity, random_state=random_state, id2text=id2text)


def merge_and_cluster(input_dir: Path, out_dir: Path, files, k=10, reps_per_cluster=10, top_n_terms=10, tsne_perplexity=30, random_state=42):
    all_ids = []
    all_embeddings = []
    idx_to_source = []
    for p in files:
        ids, embeddings = load_npz(p)
        all_ids.extend(ids)
        all_embeddings.append(embeddings)
        idx_to_source.extend([p.name] * len(ids))

    if not all_embeddings:
        raise SystemExit("No embeddings found to merge")

    all_embeddings = np.vstack(all_embeddings)
    all_ids = np.array(all_ids, dtype=object)

    # build global id->text map by scanning all jsonl files in directory
    wanted_ids = set(map(str, all_ids))
    id2text = build_global_id2text_map(input_dir, wanted_ids)
    print(f"Recovered texts for {len(id2text)}/{len(all_ids)} merged ids.")

    out_base = out_dir / (input_dir.name + '.merged')
    return cluster_and_export_for_arrays(all_ids, all_embeddings, idx_to_source, out_base, k=k, reps_per_cluster=reps_per_cluster, top_n_terms=top_n_terms, tsne_perplexity=tsne_perplexity, random_state=random_state, id2text=id2text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", "-i", required=True, help="Directory containing .emb.npz files")
    parser.add_argument("--k", type=int, default=10, help="Number of clusters for KMeans")
    parser.add_argument("--reps", type=int, default=10, help="Representative samples per cluster")
    parser.add_argument("--top-n", type=int, default=10, help="Top-n TF-IDF terms to output per cluster")
    parser.add_argument("--out-dir", "-o", default=None, help="Output directory (defaults to input-dir)")
    parser.add_argument("--tsne-perplexity", type=int, default=30, help="t-SNE perplexity")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--merge", action='store_true', help="Merge all .emb.npz files in the input dir and run a single clustering/visualization")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"input-dir {input_dir} does not exist or is not a directory")

    out_dir = Path(args.out_dir) if args.out_dir else input_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.emb.npz"))
    if not files:
        print("No .emb.npz files found in", input_dir)
        return

    summaries = []
    if args.merge:
        print("Running in MERGE mode: concatenating embeddings from all files")
        try:
            summary = merge_and_cluster(input_dir, out_dir, files, k=args.k, reps_per_cluster=args.reps, top_n_terms=args.top_n, tsne_perplexity=args.tsne_perplexity, random_state=args.random_state)
            summaries.append({"mode": "merge", **summary})
        except Exception as e:
            print(f"Merge failed: {e}", file=sys.stderr)
    else:
        for f in tqdm(files, desc="Files", unit="file"):
            try:
                summary = cluster_and_export(f, out_dir, k=args.k, reps_per_cluster=args.reps, top_n_terms=args.top_n, tsne_perplexity=args.tsne_perplexity, random_state=args.random_state)
                summaries.append({"file": f.name, **summary})
            except Exception as e:
                print(f"Failed for {f}: {e}", file=sys.stderr)

    print("Done. Summaries:")
    for s in summaries:
        print(json.dumps(s, ensure_ascii=False))

if __name__ == "__main__":
    main()
