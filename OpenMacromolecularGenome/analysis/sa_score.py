from rdkit import Chem
from rdkit.Chem import RDConfig
import os
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
#  adapted from  https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score 
import sascorer
import argparse


def compute_sascore_list(smiles_list):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="Input CSV with SMILES (column 'smiles' or specify --smiles_col)")
    p.add_argument("--smiles_col", default="smiles", help="SMILES column name")
    p.add_argument("--out", default=None, help="Output CSV path (default: input_with_scscore.csv)")
    p.add_argument("--hist", default=None, help="Path to save histogram PNG")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if args.smiles_col not in df.columns:
        raise KeyError(f"SMILES column '{args.smiles_col}' not found in CSV")


    

    print("Computing SA scores...")
    df["sa_score"] = compute_sascore_list(df[args.smiles_col])
    print("SA scores computed.")

    if args.out is None:
        base, ext = os.path.splitext(args.csv)
        args.out = f"{base}_with_sa.csv"

    df.to_csv(args.out, index=False)
    print(f"Saved output CSV to: {args.out}")

    sa = df["sa_score"].dropna()

    stats = {
        "count": sa.count(),
        "mean": sa.mean(),
        "std": sa.std(),
        "min": sa.min(),
        "25%": sa.quantile(0.25),
        "median": sa.median(),
        "75%": sa.quantile(0.75),
        "max": sa.max(),
    }

    print("\nSA score statistics (valid molecules only):")
    for k, v in stats.items():
        print(f"{k:>8s}: {v:.3f}")

    if args.hist is not None:
        # ~3.25 in ACS style

        sns.set_theme(style="whitegrid", font_scale=1.0)

        plt.figure(figsize=(3, 2.4))  # ACS single-column width

        ax = sns.histplot(sa, bins=30, kde=True, color='skyblue', edgecolor='black', alpha=0.7)

        ax.set_xlabel("Synthetic Accessibility (SA) Score", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)

        ax.tick_params(axis='both', labelsize=8)
        

        plt.tight_layout()

        # --- Save as VECTOR PDF with embedded fonts ---
        plt.savefig(
            args.hist,
            format="pdf",
            bbox_inches="tight"
        )

        plt.close()

        print("Saved histogram as vector PDF")






    


if __name__ == '__main__':
    main()
