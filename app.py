# app.py
import io
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from joblib import load as joblib_load

# Import your functions
from MultiCAST_guide_predictor import extract_guides, build_feature_dicts

st.set_page_config(page_title="MultiCAST Guide Predictor", layout="centered")
st.title("🧬 MultiCAST Guide Predictor")

st.markdown(
    "Upload a genome (FASTA), annotation (GFF3), gene list (CSV: one gene ID per line), "
    "and a trained model (.joblib). Then click **Run prediction** to get probabilities, "
    "labels, and downloads."
)

# --- Session workdir (holds tmp files for uploaded content)
if "workdir" not in st.session_state:
    st.session_state["workdir"] = tempfile.mkdtemp(prefix="multicast_")
WORKDIR = Path(st.session_state["workdir"])

# --- Sidebar controls
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
    write_guides = st.checkbox("Also export extracted guides (CSV)", value=True)
    st.divider()
    use_examples = st.toggle("Use bundled example data", value=False, help="Uses files in ./example and ./model")
    st.caption("Tip: Pin scikit-learn to the version used to train your model for joblib compatibility.")

# --- Helpers
def save_upload(uploaded_file, dest_name: str) -> Path:
    """Save an uploaded file to workdir and return its path."""
    out = WORKDIR / dest_name
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return out

@st.cache_resource(show_spinner=False)
def load_model(model_path: Path):
    return joblib_load(str(model_path))  # Pipeline(vec, clf)

def run_pipeline(genome_p: Path, gff3_p: Path, genes_p: Path, model_p: Path, threshold: float):
    # 1) Extract guides
    df_guides = extract_guides(str(genome_p), str(gff3_p), str(genes_p))
    if df_guides.empty:
        raise RuntimeError("No guides found. Check that genes exist in GFF3 and CN-PAMs are present.")

    # 2) Predict (reuse your feature builder + cached model)
    model = load_model(model_p)
    feats = build_feature_dicts(df_guides)
    proba = model.predict_proba(feats)[:, 1]
    yhat = (proba >= threshold).astype(int)

    out_df = df_guides.copy()
    out_df["proba_pos"] = proba
    out_df["pred_label_thr"] = yhat
    out_df["proba_rank"] = out_df["proba_pos"].rank(method="average", ascending=True)

    # Optional full guides table
    full_guides = df_guides.attrs.get("full_table", pd.DataFrame())
    if full_guides.empty:
        # fallback to minimal columns if detailed table wasn't attached
        full_guides = df_guides.rename(
            columns={"gene": "Gene", "pam_region": "PAM_Region", "guide_sequence": "Guide_Sequence"}
        )

    return out_df, full_guides

# --- Inputs
example_dir = Path(__file__).parent / "example"
model_dir = Path(__file__).parent / "model"

if use_examples:
    # Use repo-bundled files
    genome_path = example_dir / "GCF_008369605.1.fna"
    gff3_path   = example_dir / "GCF_008369605.1.gff"
    genes_path  = example_dir / "gene.csv"
    model_path  = model_dir / "model.joblib"

    # Quick file presence check
    missing = [p for p in [genome_path, gff3_path, genes_path, model_path] if not p.exists()]
    if missing:
        st.error(f"Example files missing: {', '.join(str(m) for m in missing)}")
else:
    c1, c2 = st.columns(2)
    with c1:
        genome_up = st.file_uploader("Genome FASTA (.fna/.fa/.fasta)", type=["fna", "fa", "fasta"])
        gff3_up   = st.file_uploader("Annotation GFF3 (.gff/.gff3)", type=["gff", "gff3"])
    with c2:
        genes_up  = st.file_uploader("Gene list CSV (one ID per line)", type=["csv"])
        model_up  = st.file_uploader("Trained model (.joblib)", type=["joblib", "pkl"])

    genome_path = save_upload(genome_up, "genome.fna") if genome_up else None
    gff3_path   = save_upload(gff3_up, "annotation.gff3") if gff3_up else None
    genes_path  = save_upload(genes_up, "genes.csv") if genes_up else None
    model_path  = save_upload(model_up, "model.joblib") if model_up else None

# --- Run
can_run = (
    use_examples and all(p and p.exists() for p in [genome_path, gff3_path, genes_path, model_path])
) or (
    (not use_examples) and all(p is not None for p in [genome_path, gff3_path, genes_path, model_path])
)

run_btn = st.button("🚀 Run prediction", disabled=not can_run)

if run_btn and can_run:
    try:
        with st.spinner("Extracting guides and running model..."):
            preds_df, guides_df = run_pipeline(genome_path, gff3_path, genes_path, model_path, threshold)

        st.success("Done! Preview below.")
        st.subheader("Predictions")
        st.dataframe(preds_df.head(50), use_container_width=True)

        # Downloads
        pred_csv = preds_df.to_csv(index=False).encode()
        st.download_button("Download predictions.csv", pred_csv, file_name="predictions.csv", mime="text/csv")

        if write_guides:
            st.subheader("Extracted guides (full)")
            st.dataframe(guides_df.head(50), use_container_width=True)
            guides_csv = guides_df.to_csv(index=False).encode()
            st.download_button("Download guides.csv", guides_csv, file_name="guides.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Error: {e}")

# --- Footer / Help
with st.expander("ℹ️ Notes & Tips"):
    st.markdown(
        """
- **Gene list CSV**: one gene identifier per line (must match one of the GFF3 attributes: `ID`, `Name`, `locus_tag`, or `gene`).
- **Model compatibility**: `joblib` pickles are version-sensitive. Pin **scikit-learn** (and **xgboost** if used) to the versions you trained with.
- **Privacy**: This app runs wherever you deploy it. For sensitive genomes, prefer running locally or on a secured private Space/server.
"""
    )

