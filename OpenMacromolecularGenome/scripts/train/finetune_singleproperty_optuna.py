import os
import pickle
import sys
import time
import torch
import random

import numpy as np
import pandas as pd

import optuna

from optuna.trial import TrialState

from multiprocessing import Manager
from joblib import parallel_backend
# "https://github.com/rapidsai/dask-cuda/issues/789"

from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdmolops import GetShortestPath

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append('/home/vani/omgfine/OpenMacromolecularGenome')
from vae.decoder.torch import Decoder
from vae.encoder.torch import CNNEncoder
from vae.preprocess import get_selfie_and_smiles_encodings_for_dataset, multiple_selfies_to_hot
from vae.property_predictor.torch import PropertyNetworkPredictionModule
from vae.training_optuna import train_model
from vae.utils.save import VAEParameters
from finetune_singleproperty_vae import filter_and_prepare_data


def finetune_single_property(
    property_name,
    pretrained_model_dir,
    csv_file_path,
    save_dir_base,
    num_epochs=50,
    batch_size=32,
    learning_rate=1e-4,  # Use a lower LR for fine-tuning the head only
    checkpoint_interval=20
):
    """
    Fine-tune a pre-trained VAE model with single-property prediction.

    Args:
        pretrained_model_dir: Path to pre-trained model directory
        csv_file_path: Path to CSV with polymer SMILES and property values
        property_columns: List of column names for the 5 properties
        save_directory: Where to save the fine-tuned model
        num_epochs: Number of fine-tuning epochs
        batch_size: Training batch size
        learning_rate: Learning rate for property predictor
    """
    # --- Paths and Device Setup ---
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Using device: {device} for {property_name} fine-tuning.")

    dtype = torch.float32
    # --- Load VAE Parameters and Constraints ---
    vae_parameters = torch.load(os.path.join(pretrained_model_dir, 'vae_parameters.pth'),
                               map_location=device)
    latent_dim = vae_parameters['latent_dimension']

     # Load encoding data from the parent directory (shared across models)
    parent_dir = os.path.dirname(pretrained_model_dir)
 

    #--- Load New Data and Preprocess ---
    df_new_data = pd.read_csv(csv_file_path)
    data_train, data_valid, prop_train, prop_valid, property_scaler = filter_and_prepare_data(
        df_new_data, 
        property_name, 
        pretrained_model_dir
    )


    # Convert to PyTorch Tensors
    data_train_tensor = torch.tensor(data_train, dtype=dtype).to(device)
    data_valid_tensor = torch.tensor(data_valid, dtype=dtype).to(device)
    y_train_tensor = torch.tensor(prop_train, dtype=dtype).to(device)
    y_valid_tensor = torch.tensor(prop_valid, dtype=dtype).to(device)
    
    # Create DataLoaders
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(data_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(data_valid_tensor, y_valid_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # --- Initialize and Freeze VAE Encoder ---
    vae_encoder = CNNEncoder(
        in_channels=vae_parameters['encoder_in_channels'],
        feature_dim=vae_parameters['encoder_feature_dim'],
        convolution_channel_dim=vae_parameters['encoder_convolution_channel_dim'],
        kernel_size=vae_parameters['encoder_kernel_size'],
        layer_1d=vae_parameters['encoder_layer_1d'],
        layer_2d=vae_parameters['encoder_layer_2d'],
        latent_dimension=latent_dim
    ).to(device)
    vae_encoder.load_state_dict(torch.load(os.path.join(pretrained_model_dir, 'encoder.pth'), map_location=device))

    vae_decoder = Decoder(
        input_size=vae_parameters['decoder_input_dimension'],
        num_layers=vae_parameters['decoder_num_gru_layers'],
        hidden_size=latent_dim,
        out_dimension=vae_parameters['decoder_output_dimension'],
        bidirectional=vae_parameters['decoder_bidirectional']
    ).to(device)

    # Load pre-trained weights
    vae_encoder.load_state_dict(torch.load(os.path.join(pretrained_model_dir, 'encoder.pth'),
                                     map_location=device))
    vae_decoder.load_state_dict(torch.load(os.path.join(pretrained_model_dir, 'decoder.pth'),
                                     map_location=device))

    # FREEZE encoder and decoder parameters
    for param in vae_encoder.parameters():
        param.requires_grad = False
    for param in vae_decoder.parameters():
        param.requires_grad = False

    # --- Initialize NEW Property Network ---
    # Single property prediction (property_dim=1)
    property_network_module = PropertyNetworkPredictionModule(
        latent_dim=latent_dim,
        property_dim=1,
        property_network_hidden_dim_list=[[64, 16]], # Simple two-layer head
        dtype=dtype,
        device=device,
        weights=(1.0,) # Single property weight is 1.0
    ).to(device)

    # --- Optimization Setup ---
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(property_network_module.parameters(), lr=learning_rate)

    # --- Training Loop ---
    save_path = os.path.join(save_dir_base, f'finetuned_{property_name}')
    os.makedirs(save_path, exist_ok=True)
    torch.save(vae_parameters, os.path.join(save_path, f'vae_parameters_{property_name}.pth'))

    
    best_val_loss = float('inf')
    print(f"Starting fine-tuning for {property_name}...")
    
    for epoch in range(num_epochs):
        # Training Phase
        property_network_module.train()
        train_loss = 0.0
        for batch_data, batch_props in train_loader:
            optimizer.zero_grad()
            
            # 1. Encode (Frozen)
            with torch.no_grad():
                batch_data_transposed = batch_data.transpose(1, 2)
                _, z_mean, _ = vae_encoder(batch_data_transposed)
            
            # 2. Predict (Trained)
            predictions = property_network_module(z_mean)
            
            # 3. Loss and Backprop
            loss = criterion(predictions, batch_props)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation Phase
        property_network_module.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data, batch_props in valid_loader:
                batch_data_transposed = batch_data.transpose(1, 2)
                _, z_mean, _ = vae_encoder(batch_data_transposed)
                predictions = property_network_module(z_mean)
                loss = criterion(predictions, batch_props)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(valid_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss (MSE): {avg_train_loss:.5f} | Val Loss (MSE): {avg_val_loss:.5f}")

        # Save the best model and scaler
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(property_network_module.state_dict(), os.path.join(save_path, f'{property_name}_predictor.pth'))
            torch.save(property_scaler, os.path.join(save_path, f'{property_name}_scaler.pth'))
            print(f"--> Saved best model and scaler at epoch {epoch+1}.")

    print(f"Fine-tuning complete for {property_name}. Best Val Loss: {best_val_loss:.5f}")




if __name__ == '__main__':

    # Path to the directory containing the VAE's encoder.pth, vae_parameters.pth, etc.
    PRETRAINED_DIR = '/home/vani/omgfine/OpenMacromolecularGenome/train/all/vae_optuna_10000_6000_3000_500_objective_1_10_10_1e-5_weight_decay/divergence_weight_4.345_latent_dim_152_learning_rate_0.002'
    NEW_DATA_CSV = '/home/vani/omgfine/OpenMacromolecularGenome/data/OMG_TP/OMG_predicted_properties_train.csv'

    # Base directory where all 5 finetuned models will be saved (e.g., finetuned_Eea, finetuned_Tg, etc.)
    SAVE_BASE_DIR = './finetuned_single_property_model'


    TARGET_PROPERTIES = ['Eea', 'Egb', 'EPS', 'PE_I', 'OPV'] 

    for prop in TARGET_PROPERTIES:
        try:
            finetune_single_property(
                property_name=prop,
                pretrained_model_dir=PRETRAINED_DIR,
                csv_file_path=NEW_DATA_CSV,
                save_dir_base=SAVE_BASE_DIR,
                num_epochs=100
            )
        except Exception as e:
            print(f"Failed to fine-tune {prop}: {e}")