import os
import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem

# Add the project root to path for SCScorer
sys.path.append('/home/vani/omgfine/OpenMacromolecularGenome')
from scscore.scscore import SCScorer

# Import SA score calculation
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

def compute_sascore_list(smiles_list):
    """Compute SA scores for a list of SMILES strings."""
    scores = []
    for smi in smiles_list:
        try:
            if pd.isna(smi) or smi == '':
                scores.append(np.nan)
                continue

            mol = Chem.MolFromSmiles(smi)
            score = sascorer.calculateScore(mol)
            scores.append(float(score))
        except Exception:
            scores.append(np.nan)
    return scores

def compute_scscore_list(smiles_list, model):
    """Compute SC scores for a list of SMILES strings."""
    scores = []
    for smi in smiles_list:
        try:
            if pd.isna(smi) or smi == '':
                scores.append(np.nan)
                continue
            # model.get_score_from_smi returns (smi, score) in the calculate_sc_score usage
            res = model.get_score_from_smi(smi)
            score = res[1] if (isinstance(res, (list, tuple)) and len(res) > 1) else float(res)
            scores.append(float(score))
        except Exception:
            scores.append(np.nan)
    return scores

def main():
    p = argparse.ArgumentParser(description="Create combined SA and SC score plots")
    p.add_argument("csv", help="Input CSV with SMILES for SA score (column 'smiles' or specify --smiles_col)")
    p.add_argument("--smiles_col", default="smiles", help="SMILES column name")
    p.add_argument("--out", default="combined_scores.pdf", help="Output PDF path")
    args = p.parse_args()

    # Load SA score data
    df= pd.read_csv(args.csv)
    if args.smiles_col not in df.columns:
        raise KeyError(f"SMILES column '{args.smiles_col}' not found in SA CSV")

    # Load SC score data
    df = pd.read_csv(args.csv)
    if args.smiles_col not in df.columns:
        raise KeyError(f"SMILES column '{args.smiles_col}' not found in SC CSV")

    print("Computing SA scores...")
    df["sa_score"] = compute_sascore_list(df[args.smiles_col])
    sa_valid = df["sa_score"].dropna()

    print("Loading SCScorer model...")
    model = SCScorer()
    model.restore()
    print("Model loaded.")

    print("Computing SC scores...")
    smiles_list = df[args.smiles_col].astype(str).tolist()
    sc_scores = []
    for i in range(len(smiles_list)):
        sc_scores.append(compute_scscore_list([smiles_list[i]], model)[0])
    df['SC_score'] = sc_scores
    sc_valid = df['SC_score'].dropna().astype(float)

    # Create combined figure
    plt.figure(figsize=(6.5, 3.0))  # Double width for two subplots

    # Plot (a) SA Score
    ax1 = plt.subplot(1, 2, 1)
    sns.histplot(sa_valid, bins=30, kde=False, ax=ax1, color='skyblue', edgecolor='black', alpha=0.7)

    ax1.set_xlabel("Synthetic Accessibility (SA) Score", fontsize=9)
    ax1.set_ylabel("Count", fontsize=9)
    ax1.tick_params(axis='both', labelsize=8)
    sns.despine(ax=ax1, top=True, right=True)

    # Add (a) label
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes,
             fontsize=10, fontweight='bold', va='top', ha='left')

    # Plot (b) SC Score
    ax2 = plt.subplot(1, 2, 2)
    sns.histplot(sc_valid, bins=30, kde=True, ax=ax2, color='lightcoral', edgecolor='black', alpha=0.7)

    ax2.set_xlabel("Synthetic Complexity (SC) Score", fontsize=9)
    ax2.set_ylabel("Count", fontsize=9)
    ax2.tick_params(axis='both', labelsize=8)
    sns.despine(ax=ax2, top=True, right=True)

    # Add (b) label
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes,
             fontsize=10, fontweight='bold', va='top', ha='left')

    plt.tight_layout()

    # Save combined figure
    plt.savefig(args.out, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"Saved combined scores plot to: {args.out}")

    # Print statistics
    print("\nSA Score Statistics:")
    sa_stats = {
        "count": sa_valid.count(),
        "mean": sa_valid.mean(),
        "std": sa_valid.std(),
        "min": sa_valid.min(),
        "25%": sa_valid.quantile(0.25),
        "median": sa_valid.median(),
        "75%": sa_valid.quantile(0.75),
        "max": sa_valid.max(),
    }
    for k, v in sa_stats.items():
        print(f"{k:>8s}: {v:.3f}")

    print("\nSC Score Statistics:")
    sc_stats = {
        "count": int(sc_valid.count()),
        "min": float(sc_valid.min()),
        "25%": float(sc_valid.quantile(0.25)),
        "median": float(sc_valid.median()),
        "mean": float(sc_valid.mean()),
        "75%": float(sc_valid.quantile(0.75)),
        "max": float(sc_valid.max())
    }
    for k, v in sc_stats.items():
        print(f"{k:>8s}: {v:.3f}")

if __name__ == "__main__":
    main()
