import os
import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from rdkit import Chem

sys.path.append('/home/vani/omgfine/OpenMacromolecularGenome')
from scscore.scscore import SCScorer


def compute_scscore_list(smiles_list, model):
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
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="Input CSV with SMILES (column 'smiles' or specify --smiles_col)")
    p.add_argument("--smiles_col", default="smiles", help="SMILES column name")
    p.add_argument("--out", default=None, help="Output CSV path (default: input_with_scscore.csv)")
    p.add_argument("--threshold", type=float, default=None, help="If set, save additionally a filtered CSV with scscore <= threshold")
    p.add_argument("--sample", type=int, default=None, help="If set, sample this many rows (random) before scoring")
    p.add_argument("--hist", default=None, help="Path to save histogram PNG")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if args.smiles_col not in df.columns:
        raise KeyError(f"SMILES column '{args.smiles_col}' not found in CSV")

    if args.sample is not None and len(df) > args.sample:
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)

    print(f"Rows to score: {len(df)}")

    print("Loading SCScorer model...")
    model = SCScorer()
    model.restore()
    print("Model loaded.")

    smiles_list = df[args.smiles_col].astype(str).tolist()
    scores = []
    # iterate with progress to avoid multiprocessing pickling issues with model
    for i in tqdm(range(0, len(smiles_list)), desc="SCScore"):
        scores.append(compute_scscore_list([smiles_list[i]], model)[0])

    df['SC_score'] = scores

    out_csv = args.out or str(Path(args.csv).with_name(Path(args.csv).stem + "_with_scscore.csv"))
    df.to_csv(out_csv, index=False)
    print(f"Saved scored CSV: {out_csv}")

    scs_valid = df['SC_score'].dropna().astype(float)
    if len(scs_valid) > 0:
        stats = {
            "count": int(scs_valid.count()),
            "min": float(scs_valid.min()),
            "25%": float(scs_valid.quantile(0.25)),
            "median": float(scs_valid.median()),
            "mean": float(scs_valid.mean()),
            "75%": float(scs_valid.quantile(0.75)),
            "max": float(scs_valid.max())
        }
        print("SCS summary:", stats)
    else:
        print("No valid SC scores computed.")

    if args.hist:
        plt.figure(figsize=(6,4))
        plt.hist(scs_valid, bins=60, color='C0', alpha=0.8)
        plt.xlabel("SC Score (lower = easier)")
        plt.ylabel("Count")
        plt.title("SC Score distribution")
        plt.tight_layout()
        plt.savefig(args.hist, dpi=150)
        plt.close()
        print(f"Saved histogram: {args.hist}")

    if args.threshold is not None:
        filtered = df[df['SC_score'].notna() & (df['SC_score'] <= args.threshold)].reset_index(drop=True)
        filtered_out = str(Path(out_csv).with_name(Path(out_csv).stem + f"_filtered_le_{args.threshold}.csv"))
        filtered.to_csv(filtered_out, index=False)
        print(f"Saved filtered CSV (<= {args.threshold}): {filtered_out}  (n={len(filtered)})")


if __name__ == "__main__":
    main()