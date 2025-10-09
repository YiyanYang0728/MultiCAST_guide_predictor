#!/usr/bin/env python3
"""
python MultiCAST_guide_predictor.py \
  --genome example/GCF_008369605.1.fna \
  --gff3 example/GCF_008369605.1.gff \
  --genes example/gene.csv \
  --model model/model.joblib \
  --outprefix results/predictions

Optional:
  --threshold 0.5
  --write-guides example/guides.csv    # also save the extracted guides

Requires
--------
- biopython, numpy, pandas, scikit-learn, joblib
"""

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import load as joblib_load
from Bio import SeqIO

# --------------------------- Part 1: Extract guides (from your first script) ---------------------------

def parse_gff3(gff3_file):
    """Parse GFF3 file and return dictionary of gene features."""
    genes = {}
    with open(gff3_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            feature_type = fields[2]
            if feature_type not in ['gene', 'CDS']:
                continue

            seqid = fields[0]
            start = int(fields[3])
            end = int(fields[4])
            strand = fields[6]
            attributes = fields[8]

            attr_dict = {}
            for attr in attributes.split(';'):
                if '=' in attr:
                    key, value = attr.split('=', 1)
                    attr_dict[key] = value

            gene_info = {
                'seqid': seqid,
                'start': start,
                'end': end,
                'strand': strand,
                'length': end - start + 1
            }

            for key in ['ID', 'Name', 'locus_tag', 'gene']:
                if key in attr_dict:
                    genes[attr_dict[key]] = gene_info
    return genes


def find_cn_pam_guides(sequence, gene_length, gene_strand):
    """
    Find all CN PAM sites with upstream context and downstream 32nt guide sequences.
    Returns list of (pam_n_pos, upstream_pam(5nt), guide(32nt), strand_type, near_region).
    """
    results = []
    seq_str = str(sequence)

    # Forward strand search
    pattern = r'(?=([ATCG]{3}C[ATCG].{32}))'
    for match in re.finditer(pattern, seq_str):
        pos = match.start()
        full_seq = match.group(1)
        upstream_pam = full_seq[:5]        # 3 upstream + C + N
        guide = full_seq[5:37]             # 32 nt guide
        pam_n_pos = pos + 4                # position of the N in CN
        end_of_guide = pos + 36

        strand_type = 'coding'
        midpoint = gene_length / 2
        near_region = 'Yes' if (end_of_guide + 50) < midpoint else 'No'

        results.append((pam_n_pos, upstream_pam, guide, strand_type, near_region))

    # Reverse-complement (template) search
    rc_seq = str(sequence.reverse_complement())
    for match in re.finditer(pattern, rc_seq):
        pos = match.start()
        full_seq = match.group(1)
        upstream_pam = full_seq[:5]
        guide = full_seq[5:37]
        end_of_guide = pos + 36

        pam_n_pos = gene_length - pos - 1  # map back to forward coords
        strand_type = 'template'
        midpoint = gene_length / 2
        check_pos = end_of_guide + 50
        near_region = 'Yes' if (midpoint < check_pos < (gene_length - 75)) else 'No'

        results.append((pam_n_pos, upstream_pam, guide, strand_type, near_region))

    return results


def extract_guides(genome_fasta: str, gff3_file: str, genes_csv: str) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: gene, pam_region, guide_sequence
    (plus extra columns preserved if you want to save them).
    """
    # Load genome
    genome = {record.id: record.seq for record in SeqIO.parse(genome_fasta, 'fasta')}

    # Parse GFF3
    genes = parse_gff3(gff3_file)

    # Read gene list
    wanted_genes: List[str] = []
    with open(genes_csv, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                wanted_genes.append(row[0])

    rows_simple = []   # minimal columns for the model
    rows_full = []     # optional extra info if writing out

    for gene_name in wanted_genes:
        if gene_name not in genes:
            # silently skip missing genes (mirrors original behavior)
            continue

        info = genes[gene_name]
        seqid, start, end, strand, gene_length = (
            info['seqid'], info['start'], info['end'], info['strand'], info['length']
        )
        if seqid not in genome:
            continue

        orf_seq = genome[seqid][start-1:end]
        if strand == '-':
            orf_seq = orf_seq.reverse_complement()

        guides = find_cn_pam_guides(orf_seq, gene_length, strand)
        for i, (pos, upstream_pam, guide, strand_type, near_region) in enumerate(guides, 1):
            # minimal set used by your model
            rows_simple.append({
                "gene": gene_name,
                "pam_region": upstream_pam,
                "guide_sequence": guide
            })
            # optional extras
            rows_full.append({
                "Gene": gene_name,
                "PAM_Region": upstream_pam,
                "Guide_Sequence": guide,
                "Sequence_Number": i,
                "Strand": strand_type,
                "Center": near_region
            })

    df_simple = pd.DataFrame(rows_simple)
    df_full = pd.DataFrame(rows_full)
    # Attach full table so caller can optionally save it
    df_simple.attrs["full_table"] = df_full
    return df_simple

# --------------------------- Part 2: Features + predict (from your second script) ---------------------------

DNA = set("ACGT")

def shannon_entropy(seq: str) -> float:
    from math import log2
    if not seq:
        return 0.0
    s = seq.upper()
    L = len(s)
    counts = {b: s.count(b) for b in "ACGT"}
    probs = [c / L for c in counts.values() if c > 0]
    return float(-sum(p * log2(p) for p in probs)) if probs else 0.0

def max_run_length(seq: str, base: str | None = None) -> int:
    s = seq.upper()
    maxlen, cur = 0, 0
    last = None
    for ch in s:
        if base is None:
            if ch == last:
                cur += 1
            else:
                cur = 1
            last = ch
        else:
            if ch == base:
                cur += 1
            else:
                cur = 0
        maxlen = max(maxlen, cur)
    return maxlen

def kmer_exact_counts(seq: str, k: int, prefix: str) -> Dict[str, int]:
    s = str(seq).upper()
    feats: Dict[str, int] = {}
    if len(s) < k:
        return feats
    for i in range(len(s) - k + 1):
        kmer = s[i:i+k]
        if set(kmer) <= DNA:
            key = f"{prefix}={kmer}"
            feats[key] = feats.get(key, 0) + 1
    return feats

def row_to_feat(r, add_globals: bool = True, include_gene: bool = False) -> Dict[str, float | int]:
    feats: Dict[str, float | int] = {}
    if include_gene:
        feats[f"gene={str(r['gene'])}"] = 1

    pam = str(r["pam_region"]).strip().upper()
    for i, b in enumerate(pam):
        feats[f"pam_{i}={b}"] = 1

    guide = str(r["guide_sequence"]).strip().upper()
    for i, b in enumerate(guide):
        feats[f"guide_{i}={b}"] = 1

    if add_globals:
        gc = guide.count("G") + guide.count("C")
        feats["guide_cnt_gc"] = gc
        for b in "ACGT":
            feats[f"guide_cnt_{b}"] = guide.count(b)

        for b in "ACGT":
            feats[f"guide_max_run_{b}"] = max_run_length(guide, b)

        feats["guide_entropy"] = shannon_entropy(guide)

        for b in "ACGT":
            feats[f"pam_cnt_{b}"] = pam.count(b)

        feats.update(kmer_exact_counts(guide, 2, "k2"))
        feats.update(kmer_exact_counts(guide, 3, "k3"))

    return feats

def build_feature_dicts(df: pd.DataFrame) -> List[Dict[str, float | int]]:
    return [row_to_feat(r) for _, r in df.iterrows()]

def run_prediction(df_input: pd.DataFrame, model_path: str, outprefix: str, threshold: float = 0.5) -> str:
    """
    Takes a df with columns ['gene','pam_region','guide_sequence'], writes <outprefix>.csv
    with probabilities and thresholded predictions. Returns the written path.
    """
    os.makedirs(os.path.dirname(outprefix) or ".", exist_ok=True)

    feats = build_feature_dicts(df_input)
    pipe = joblib_load(model_path)  # Pipeline(vec, clf)

    proba = pipe.predict_proba(feats)[:, 1]
    yhat = (proba >= threshold).astype(int)

    out_df = df_input.copy()
    out_df["proba_pos"] = proba
    out_df["pred_label_thr"] = yhat
    out_df["proba_rank"] = out_df["proba_pos"].rank(method="average", ascending=True)

    pred_path = outprefix + ".csv"
    out_df.to_csv(pred_path, index=False)
    return pred_path

# --------------------------- Orchestration CLI ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Extract upstream PAM + guides from genes and predict guide activity with a trained XGBoost model.")
    ap.add_argument("--genome", "-g", required=True, help="Genome FASTA")
    ap.add_argument("--gff3", "-f", required=True, help="Annotation GFF3")
    ap.add_argument("--genes", "-l", required=True, help="Gene list file with one gene ID per line (ID/Name/locus_tag/gene matching the GFF3).")
    ap.add_argument("--model", "-m", required=True, help="Path to trained pipeline (e.g., model/final_model.joblib)")
    ap.add_argument("--outprefix", "-o", default="results/predictions", help="Output prefix for predictions CSV (default results/predictions)")
    ap.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for predicted label (default 0.5)")
    ap.add_argument("--write-guides", default=None, help="Optional path to also write extracted guides table (CSV)")
    args = ap.parse_args()

    # 1) Extract guides
    df_guides = extract_guides(args.genome, args.gff3, args.genes)

    if df_guides.empty:
        raise SystemExit("No guides found. Check your inputs (genes present in GFF3, CN-PAM presence, etc.).")

    # Optionally write the more detailed guides table, matching your original columns
    if args.write_guides:
        full = df_guides.attrs.get("full_table", pd.DataFrame())
        if full.empty:
            # fallback: write minimal columns if full not present
            df_guides.rename(columns={
                "gene": "Gene", "pam_region": "PAM_Region", "guide_sequence": "Guide_Sequence"
            }).to_csv(args.write_guides, index=False)
        else:
            os.makedirs(os.path.dirname(args.write_guides) or ".", exist_ok=True)
            full.to_csv(args.write_guides, index=False)

    # 2) Predict
    pred_path = run_prediction(
        df_guides,
        model_path=args.model,
        outprefix=args.outprefix,
        threshold=args.threshold
    )
    print(f"Wrote predictions to: {pred_path}")
    if args.write_guides:
        print(f"Wrote extracted guides to: {args.write_guides}")

if __name__ == "__main__":
    main()

