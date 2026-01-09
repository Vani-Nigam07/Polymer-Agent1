
from collections import Counter
import pandas as pd
import numpy as np
import torch
import sys

sys.path.append('/home/vani/omgfine/OpenMacromolecularGenome')
import selfies as sf
from vae.preprocess import get_selfie_and_smiles_encodings_for_dataset, multiple_selfies_to_hot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def filter_and_prepare_data(
    df_new_data, 
    property_name,
    pretrained_dir,
    smiles_column='product'
):
    """
    Filters new data to ensure compatibility with VAE's alphabet and length, 
    then prepares the one-hot encoded input and scaled property values.
    """

    
    try:
        encoding_alphabet = torch.load(os.path.join(pretrained_dir, 'encoding_alphabet.pth'))
        largest_molecule_len = torch.load(os.path.join(pretrained_dir, 'largest_molecule_len.pth'))
        encoding_list = torch.load(os.path.join(pretrained_dir, 'encoding_list.pth'))
    except FileNotFoundError as e:
        print(f"Error loading VAE constraints from {pretrained_dir}. Check paths.")
        raise e

    # 1. Initial Filtering for property availability
    print(f"Filtering data flen: {df_new_data.shape[0]} for property: {property_name} ")
    df_filtered = df_new_data.dropna(subset=[smiles_column, property_name]).reset_index(drop=True)
    print(f"Loaded {len(df_filtered)} polymer-property pairs for {property_name}.")
    
    polymer_smiles = df_filtered[smiles_column].tolist()
    temp_df = pd.DataFrame({smiles_column: polymer_smiles})

    # 2. Convert to SELFIES and check compatibility
    # NOTE: get_selfie_and_smiles_encodings_for_dataset is used to get the selfies strings
    # We pass the pre-trained constraints (alphabet, length) but the function may ignore them 
    # when first calculating SELFIES for the new SMILES.

    # This call is primarily to get the SELFIES list and implicitly ensure they can be generated.
    (selfies_list, _, _, _, _, _) = get_selfie_and_smiles_encodings_for_dataset(
        temp_df)
    
    # Check 1: Token Compatibility (Crucial step)
    known_tokens = set(encoding_alphabet.keys() if isinstance(encoding_alphabet, dict) else encoding_alphabet)
    valid_indices = []
    
    for i, s in enumerate(selfies_list):
        if s is None: continue # Skip if conversion failed
        tokens = list(sf.split_selfies(s))
        # Check 2: Length Compatibility
        if all(tok in known_tokens for tok in tokens) and sf.len_selfies(s) <= largest_molecule_len:
            valid_indices.append(i)
    
    # 3. Apply Filters and Finalize Data
    if not valid_indices:
        raise ValueError("No valid polymers remain after filtering.")

    df_final = df_filtered.iloc[valid_indices].reset_index(drop=True)
    selfies_final = [selfies_list[i] for i in valid_indices]
    
    print(f"Filtered to {len(df_final)} compatible polymers (Token/Length match VAE).")

    # 4. Create One-Hot Encoding Input (X)
    data = multiple_selfies_to_hot(selfies_final, largest_molecule_len, encoding_alphabet)
    # The original script excludes the first character (likely [*])
    data = data[:, 1:, :] 

    # 5. Extract and Scale Property (Y)
    property_value = df_final[property_name].to_numpy().reshape(-1, 1)
    
    # Scale property (New StandardScaler)
    property_scaler = StandardScaler()
    property_scaled = property_scaler.fit_transform(property_value)

    # 6. Split Data
    # Match the VAE's train/test split convention (8:1:1 split is common, based on your script)
    data_train_val, data_test, prop_train_val, prop_test = train_test_split(
        data, property_scaled, test_size=0.1, random_state=42
    )
    data_train, data_valid, prop_train, prop_valid = train_test_split(
        data_train_val, prop_train_val, test_size=(1/9), random_state=42
    ) # 1/9 of train_val is 10% of total
    
    return data_train, data_valid, prop_train, prop_valid, property_scaler


# if __name__ == '__main__':
#     pretrained_dir = '/home/vani/omgfine/OpenMacromolecularGenome/train/all/vae_optuna_10000_6000_3000_500_objective_1_10_10_1e-5_weight_decay/'
#     data = '/home/vani/omgfine/OpenMacromolecularGenome/data/OMG_TP/OMG_predicted_properties_train.csv'
#     df_new = pd.read_csv(data)
#     data_train, data_valid, prop_train, prop_valid, property_scaler = filter_and_prepare_data(
#         df_new, property_name='Eea', pretrained_dir=pretrained_dir, smiles_column='product'
#     )
#     print(f"lengths of data_train: {len(data_train)}, data_valid: {len(data_valid)}")