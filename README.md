usage: MultiCAST_guide_predictor.py [-h] --genome GENOME --gff3 GFF3 --genes GENES --model MODEL [--outprefix OUTPREFIX] [--threshold THRESHOLD]
                                    [--write-guides WRITE_GUIDES]

Extract upstream PAM + guides from genes and predict guide activity with a trained XGBoost model.

options:
  -h, --help            show this help message and exit
  --genome, -g GENOME   Genome FASTA
  --gff3, -f GFF3       Annotation GFF3
  --genes, -l GENES     Gene list file with one gene ID per line (ID/Name/locus_tag/gene matching the GFF3).
  --model, -m MODEL     Path to trained pipeline (e.g., model/final_model.joblib)
  --outprefix, -o OUTPREFIX
                        Output prefix for predictions CSV (default results/predictions)
  --threshold THRESHOLD
                        Decision threshold for predicted label (default 0.5)
  --write-guides WRITE_GUIDES
                        Optional path to also write extracted guides table (CSV)


python MultiCAST_guide_predictor.py -g example/GCF_008369605.1.fna -f example/GCF_008369605.1.gff -l example/gene.csv -m model/model.joblib -o results/predictions --write-guides

