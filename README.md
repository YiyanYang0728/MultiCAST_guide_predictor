# MultiCAST Guide Predictor

Extract upstream PAM + 32-nt guides from genes and predict guide activity with a trained model pipeline.

## Requirements

- Python 3.9+
- Packages: `biopython`, `numpy`, `pandas`, `scikit-learn`, `joblib`, `scikit-learn`

```bash
pip install biopython numpy pandas scikit-learn==1.3.2 joblib
```

## Usage

```bash
python MultiCAST_guide_predictor.py \
  --genome GENOME.fna \
  --gff3 annotation.gff3 \
  --genes genes.csv \
  --model model.joblib \
  [--outprefix results/predictions] \
  [--threshold 0.5] \
  [--write-guides guides.csv]
```

### Options

- `-g, --genome` **(required)**: Genome FASTA (`.fna/.fa/.fasta`)
- `-f, --gff3` **(required)**: Annotation GFF3
- `-l, --genes` **(required)**: Gene list CSV, **one ID per line**. IDs must match one of GFF3 attributes: `ID`, `Name`, `locus_tag`, or `gene`.
- `-m, --model` **(required)**: Path to trained **scikit-learn pipeline** serialized with joblib (optionally includes XGBoost), e.g. `model/model.joblib`
- `-o, --outprefix` *(default: `results/predictions`)*: Output prefix for predictions CSV
- `--threshold` *(default: `0.5`)*: Probability cutoff used to create `pred_label_thr`
- `--write-guides` *(optional)*: Also write the extracted guides table (CSV) to this path

## Example

Using the bundled example files:

```bash
python MultiCAST_guide_predictor.py \
  -g example/GCF_008369605.1.fna \
  -f example/GCF_008369605.1.gff \
  -l example/gene.csv \
  -m model/model.joblib \
  -o results/predictions \
  --write-guides results/guides.csv
```

## Inputs

- **Genome FASTA**: One or more sequences; header IDs must match the `seqid` used in the GFF3.
- **GFF3**: Gene annotations. The script looks for `gene` and `CDS` features and reads attributes from column 9.
- **Gene list (CSV)**: A single column with one gene identifier per line; must match `ID`/`Name`/`locus_tag`/`gene` in the GFF3.

## What it does

1. Parses the GFF3 and extracts each requested gene’s ORF (reverse-complements if on `-` strand).
2. Scans for PAM sites with upstream context and downstream **32-nt guide** windows on both strands.
3. Builds features for each candidate (PAM/guide positions, counts, runs, entropy, k-mers).
4. Loads a trained pipeline (`joblib`) and predicts **probability of positive activity**.

## Outputs

- **`<outprefix>.csv`** (e.g., `results/predictions.csv`) with columns:
  - `gene`, `pam_region`, `guide_sequence`
  - `proba_pos` — predicted probability (class 1)
  - `pred_label_thr` — 0/1 using `--threshold`
  - `proba_rank` — rank of `proba_pos` (1 = lowest if ascending; useful for prioritization)
- **`--write-guides <path>`** (optional): Full guide table including extra context (strand, index, near-center flag).  
  If the detailed table isn’t available, a minimal `{Gene, PAM_Region, Guide_Sequence}` CSV is written.

## Notes & Tips

- If you see `No guides found`, double-check that your gene IDs exist in the GFF3 and that PAM sites are present.

## Online tool
- You can use the tool directly by visiting our online portal [MultiCAST_guide_predictor](https://multicastguidepredictor-v1.streamlit.app/) now!
