import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

sys.path.append('/home/vani/omgfine/OpenMacromolecularGenome')
from vae.decoder.torch import Decoder
from vae.encoder.torch import CNNEncoder
from vae.preprocess import get_selfie_and_smiles_encodings_for_dataset, multiple_selfies_to_hot
from vae.property_predictor.torch import PropertyNetworkPredictionModule

def finetune_multiproperty_vae(
        pretrained_model_path, 
        csv_file_path,
        property_columns,
        save_directory,
        num_epochs=50, 
        batch_size=32, 
        learning_rate=1e-4, 
        checkpoint_interval=20):
    """
    Fine-tune a pre-trained VAE model with multi-property prediction.

    Args:
        pretrained_model_path: Path to pre-trained model directory
        csv_file_path: Path to CSV with polymer SMILES and property values
        property_columns: List of column names for the 5 properties
        save_directory: Where to save the fine-tuned model
        num_epochs: Number of fine-tuning epochs
        batch_size: Training batch size
        learning_rate: Learning rate for property predictor
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print(f"Using device: {device}")

    # Load pre-trained VAE parameters
    vae_parameters = torch.load(os.path.join(pretrained_model_path, 'vae_parameters.pth'),
                               map_location=device)
    latent_dim = vae_parameters['latent_dimension']

    # Load encoding data from the parent directory (shared across models)
    parent_dir = os.path.dirname(pretrained_model_path)
    encoding_alphabet = torch.load(os.path.join(parent_dir, 'encoding_alphabet.pth'))
    encoding_list = torch.load(os.path.join(parent_dir, 'encoding_list.pth'))
    largest_molecule_len = torch.load(os.path.join(parent_dir, 'largest_molecule_len.pth'))

    # Initialize and load pre-trained encoder and decoder
    encoder = CNNEncoder(
        in_channels=vae_parameters['encoder_in_channels'],
        feature_dim=vae_parameters['encoder_feature_dim'],
        convolution_channel_dim=vae_parameters['encoder_convolution_channel_dim'],
        kernel_size=vae_parameters['encoder_kernel_size'],
        layer_1d=vae_parameters['encoder_layer_1d'],
        layer_2d=vae_parameters['encoder_layer_2d'],
        latent_dimension=latent_dim
    ).to(device)

    decoder = Decoder(
        input_size=vae_parameters['decoder_input_dimension'],
        num_layers=vae_parameters['decoder_num_gru_layers'],
        hidden_size=latent_dim,
        out_dimension=vae_parameters['decoder_output_dimension'],
        bidirectional=vae_parameters['decoder_bidirectional']
    ).to(device)

    # Load pre-trained weights
    encoder.load_state_dict(torch.load(os.path.join(pretrained_model_path, 'encoder.pth'),
                                     map_location=device))
    decoder.load_state_dict(torch.load(os.path.join(pretrained_model_path, 'decoder.pth'),
                                     map_location=device))

    # FREEZE encoder and decoder parameters
    for param in encoder.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False

    print("Encoder and decoder frozen - only property predictor will be trained")

    # Load your multi-property dataset
    df_properties = pd.read_csv(csv_file_path)

    # Assume CSV has 'product' column for SMILES and property_columns for values
    # Filter to only polymers that are valid and can be encoded
    df_filtered = df_properties.dropna(subset=['product'] + property_columns)

    print(f"Loaded {len(df_filtered)} polymer-property pairs")

    # Encode polymers to SELFIES for THIS dataset
    polymer_smiles = df_filtered['product'].tolist()
    temp_df = pd.DataFrame({'product': polymer_smiles})


    from collections import Counter
    import selfies as sf  # used in OMG anyway

    def log_selfies_coverage(selfies_list, encoding_alphabet, max_examples=10):
        """
        Check how many SELFIES tokens are outside the encoding_alphabet and log them.
        """
        # Turn alphabet into a set of tokens
        if isinstance(encoding_alphabet, dict):
            known_tokens = set(encoding_alphabet.keys())
        else:
            known_tokens = set(encoding_alphabet)

        unknown_counter = Counter()
        examples_with_unknowns = []

        for i, s in enumerate(selfies_list):
            tokens = list(sf.split_selfies(s))
            unknown_tokens = [tok for tok in tokens if tok not in known_tokens]

            if unknown_tokens:
                for tok in unknown_tokens:
                    unknown_counter.update(unknown_tokens)
                if len(examples_with_unknowns) < max_examples:
                    examples_with_unknowns.append(
                        {"index": i, "selfies": s, "unknown_tokens": unknown_tokens}
                    )

        total_unknown_tokens = sum(unknown_counter.values())
        num_molecules_with_unknowns = len(examples_with_unknowns)

        print("==== SELFIES coverage check ====")
        print(f"Total molecules: {len(selfies_list)}")
        print(f"Unique unknown token types: {len(unknown_counter)}")
        print(f"Total unknown tokens: {total_unknown_tokens}")
        print(f"Molecules with unknown tokens (sampled): {num_molecules_with_unknowns}")
        if unknown_counter:
            print("Most common unknown tokens:")
            for tok, cnt in unknown_counter.most_common(20):
                print(f"  {tok}: {cnt}")

            print("\nExample molecules with unknown tokens:")
            for ex in examples_with_unknowns:
                print(f"- idx {ex['index']}: {ex['selfies']}")
                print(f"  unknown: {ex['unknown_tokens']}")
        else:
            print("No unknown tokens, all SELFIES are in-distribution.")
        print("================================")

        return unknown_counter
    


    def get_valid_selfies_indices(selfies_list, encoding_alphabet):
        """
        Return indices of molecules whose SELFIES tokens are all in encoding_alphabet.
        """
        # Turn alphabet into a set of known tokens
        if isinstance(encoding_alphabet, dict):
            known_tokens = set(encoding_alphabet.keys())
        else:
            known_tokens = set(encoding_alphabet)

        valid_indices = []
        invalid_indices = []

        for i, s in enumerate(selfies_list):
            tokens = list(sf.split_selfies(s))
            if all(tok in known_tokens for tok in tokens):
                valid_indices.append(i)
            else:
                invalid_indices.append(i)

        print(f"Total molecules: {len(selfies_list)}")
        print(f"Valid (all tokens known): {len(valid_indices)}")
        print(f"Invalid (contain unknown tokens): {len(invalid_indices)}")

        return valid_indices, invalid_indices


    selfies_list, _, _, _, _, _ = get_selfie_and_smiles_encodings_for_dataset(temp_df)
    valid_indices, invalid_indices = get_valid_selfies_indices(selfies_list, encoding_alphabet)

    # Additional filtering for molecule length compatibility with pretrained model
    length_valid_indices = [i for i in valid_indices if sf.len_selfies(selfies_list[i]) <= largest_molecule_len]
    print(f"Filtered for length: {len(length_valid_indices)} molecules remain (max length {largest_molecule_len})")

    # Filter everything consistently
    selfies_list = [selfies_list[i] for i in length_valid_indices]
    df_filtered = df_filtered.iloc[length_valid_indices].reset_index(drop=True)

    valid_indices, invalid_indices = get_valid_selfies_indices(selfies_list, encoding_alphabet)

    if invalid_indices:
        print("WARNING: Dropping molecules with SELFIES tokens not seen during VAE training.")


        # Rebuild temp_df from the filtered polymers (optional but clean)
    temp_df = pd.DataFrame({'product': df_filtered['product'].tolist()})

    # One-hot encoding

    data = multiple_selfies_to_hot(selfies_list, largest_molecule_len, encoding_alphabet)
    data = data[:, 1:, :]
    # Get property values
    property_values = df_filtered[property_columns].values  # Shape: (n_samples, 5)

    print(f"Data shape: {data.shape}, Properties shape: {property_values.shape}")



    # Create new multi-property predictor
    property_network = PropertyNetworkPredictionModule(
        latent_dim=latent_dim,
        property_dim=len(property_columns),
        property_network_hidden_dim_list=[[64, 16], [64, 16], [32, 8], [32, 8], [64, 32]],
        dtype=dtype,
        device=device,
        weights=[1.0/len(property_columns)] * len(property_columns)  # Equal weights
    ).to(device)

    # Set up training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(property_network.parameters(), lr=learning_rate)

    # Split data
    train_data, val_data, train_props_raw, val_props_raw = train_test_split(
        data, property_values, test_size=0.2, random_state=42
    )

    # Normalize properties
    property_scaler = StandardScaler()
    train_props= property_scaler.fit_transform(train_props_raw)
    val_props = property_scaler.transform(val_props_raw)

    train_dataset = TensorDataset(torch.tensor(train_data, dtype=dtype),
                                torch.tensor(train_props, dtype=dtype))
    val_dataset = TensorDataset(torch.tensor(val_data, dtype=dtype),
                              torch.tensor(val_props, dtype=dtype))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    train_losses = []
    val_losses = []

    print("Starting fine-tuning...")
    for epoch in range(num_epochs):
        property_network.train()
        epoch_train_loss = 0.0

        for batch_data, batch_props in train_loader:
            batch_data = batch_data.to(device)
            batch_props = batch_props.to(device)

            optimizer.zero_grad()

            # Encode to latent space (frozen)
            with torch.no_grad():
               batch_data_transposed = batch_data.transpose(1, 2)  # [batch, seq, channels] -> [batch, channels, seq]
               _, z_mean, _ = encoder(batch_data_transposed)
            
            predictions = property_network(z_mean)

            # Compute loss
            loss = criterion(predictions, batch_props)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation
        property_network.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for batch_data, batch_props in val_loader:
                batch_data = batch_data.to(device)
                batch_props = batch_props.to(device)

                batch_data_transposed = batch_data.transpose(1, 2)  # [batch, seq, channels] -> [batch, channels, seq]
                _, z_mean, _ = encoder(batch_data_transposed)
                predictions = property_network(z_mean)
                loss = criterion(predictions, batch_props)

                epoch_val_loss += loss.item()

        epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)

                # ---- end of epoch ----

        # Save epoch-wise checkpoints
        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == num_epochs:
            ckpt_dir = os.path.join(save_directory, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)

            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": property_network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
            }
            ckpt_path = os.path.join(ckpt_dir, f"multiprop_epoch_{epoch+1}.pth")
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint at {ckpt_path}")


    # Save the fine-tuned property predictor
    os.makedirs(save_directory, exist_ok=True)

    torch.save(property_network.state_dict(),
               os.path.join(save_directory, 'property_predictor_multiproperty.pth'))

    # Save property scaler
    torch.save(property_scaler, os.path.join(save_directory, 'property_scaler.pth'))

    # Update and save parameters
    vae_parameters['property_dim'] = len(property_columns)
    vae_parameters['property_names'] = property_columns  # Save the actual property names!
    vae_parameters['property_network_hidden_dim_list'] = [[64, 16], [64, 16], [32, 8], [32, 8], [64, 32]]
    vae_parameters['property_weights'] = [1.0/len(property_columns)] * len(property_columns)
    torch.save(vae_parameters, os.path.join(save_directory, 'vae_parameters_multiproperty.pth'))

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Multi-Property Fine-tuning Loss')
    plt.legend()
    plt.savefig(os.path.join(save_directory, 'finetuning_loss.png'))
    plt.close()

    print(f"Fine-tuned model saved to {save_directory}")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")

    return save_directory

def find_pretrained_model(base_dir='OpenMacromolecularGenome/train/all'):
    """Find available pre-trained model directories"""
    import glob
    import os

    # Look for model directories
    model_pattern = os.path.join(base_dir, '*/**/vae_parameters.pth')
    model_files = glob.glob(model_pattern, recursive=True)

    if not model_files:
        # Fallback: look directly in expected location
        fallback_path = './train/all/vae_optuna_10000_6000_3000_500_objective_1_10_10_1e-5_weight_decay/divergence_weight_4.345_latent_dim_152_learning_rate_0.002'
        if os.path.exists(os.path.join(fallback_path, 'vae_parameters.pth')):
            return fallback_path

        print("Available model directories:")
        for root, dirs, files in os.walk(base_dir):
            if 'vae_parameters.pth' in files:
                print(f"  {root}")

        raise FileNotFoundError(f"No pre-trained models found in {base_dir}")

    # Use the first found model (or last in the sorted list, assuming latest is best)
    model_path = sorted(model_files)[-1]  # Get the last (likely best) model
    return os.path.dirname(model_path)

if __name__ == '__main__':
    # Auto-discover pre-trained model
    try:
        pretrained_path = find_pretrained_model()
        print(f"Found pre-trained model at: {pretrained_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have pre-trained VAE models in the ./train/all directory")
        exit(1)

    # Check if user's data file exists, otherwise use example
    csv_path = './data/OMG_predicted_properties.csv'  # Your multiproperty data
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found, using placeholder path")
        csv_path = '/home/vani/omgfine/OpenMacromolecularGenome/data/OMG_predicted_properties.csv'  # Your multiproperty data
        

    # Use actual column names from your CSV
    property_cols = ['Eea', 'Egb', 'EPS', 'PE_I', 'OPV']

    save_dir = './finetuned_multiproperty_model'

    print(f"Loading data from: {csv_path}")
    print(f"Property columns: {property_cols}")
    print(f"Saving to: {save_dir}")

    finetune_multiproperty_vae(
        pretrained_model_path=pretrained_path,
        csv_file_path=csv_path,
        property_columns=property_cols,
        save_directory=save_dir,
        num_epochs=50
    )
