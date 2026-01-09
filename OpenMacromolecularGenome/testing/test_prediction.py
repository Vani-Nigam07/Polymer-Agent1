import os
import sys
import torch
import pandas as pd
import numpy as np

sys.path.append('/home/vani/omgfine/OpenMacromolecularGenome')
from vae.decoder.torch import Decoder
from vae.encoder.torch import CNNEncoder
from vae.preprocess import get_selfie_and_smiles_encodings_for_dataset, multiple_selfies_to_hot
from vae.property_predictor.torch import PropertyNetworkPredictionModule

# ---------- PATHS ----------
PRETRAINED_VAE_DIR = "/home/vani/omgfine/OpenMacromolecularGenome/train/all/vae_optuna_10000_6000_3000_500_objective_1_10_10_1e-5_weight_decay/divergence_weight_4.345_latent_dim_152_learning_rate_0.002"
FINETUNED_DIR =  "/home/vani/omgfine/OpenMacromolecularGenome/finetuned_multiproperty_model"
PARENT_DIR = os.path.dirname(PRETRAINED_VAE_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32


def load_models_and_scaler():
    # Load VAE parameters (multi-property version you saved)
    vae_params = torch.load(
        os.path.join(FINETUNED_DIR, "vae_parameters_multiproperty.pth"),
        map_location=device,
    )

    latent_dim = vae_params["latent_dimension"]
    property_dim = vae_params["property_dim"]
    hidden_list = vae_params["property_network_hidden_dim_list"]
    weights = vae_params.get(
        "property_weights", [1.0 / property_dim] * property_dim
    )

    # Load encoding info
    encoding_alphabet = torch.load(
        os.path.join(PARENT_DIR, "encoding_alphabet.pth"), map_location=device
    )
    largest_molecule_len = torch.load(
        os.path.join(PARENT_DIR, "largest_molecule_len.pth"), map_location=device
    )

    # Rebuild encoder and load pretrained weights
    encoder = CNNEncoder(
        in_channels=vae_params["encoder_in_channels"],
        feature_dim=vae_params["encoder_feature_dim"],
        convolution_channel_dim=vae_params["encoder_convolution_channel_dim"],
        kernel_size=vae_params["encoder_kernel_size"],
        layer_1d=vae_params["encoder_layer_1d"],
        layer_2d=vae_params["encoder_layer_2d"],
        latent_dimension=latent_dim,
    ).to(device)

    encoder.load_state_dict(
        torch.load(os.path.join(PRETRAINED_VAE_DIR, "encoder.pth"), map_location=device)
    )
    encoder.eval()

    # Build property network and load fine-tuned weights
    prop_net = PropertyNetworkPredictionModule(
        latent_dim=latent_dim,
        property_dim=property_dim,
        property_network_hidden_dim_list=hidden_list,
        dtype=dtype,
        device=device,
        weights=weights,
    ).to(device)

    prop_net.load_state_dict(
        torch.load(
            os.path.join(FINETUNED_DIR, "property_predictor_multiproperty.pth"),
            map_location=device,
        )
    )
    prop_net.eval()

    # Load scaler
    scaler = torch.load(
        os.path.join(FINETUNED_DIR, "property_scaler.pth"), map_location=device, weights_only=False
    )

    return encoder, prop_net, scaler, encoding_alphabet, largest_molecule_len


def smiles_to_onehot(smiles_list, encoding_alphabet, largest_molecule_len):
    df = pd.DataFrame({"product": smiles_list})
    selfies_list, _, _, _, _, _ = get_selfie_and_smiles_encodings_for_dataset(df)

    data = multiple_selfies_to_hot(selfies_list, largest_molecule_len, encoding_alphabet)
    data = data[:, 1:, :]  # same as training
    return data


def predict_properties(smiles_list, property_columns):
    (
        encoder,
        prop_net,
        scaler,
        encoding_alphabet,
        largest_molecule_len,
    ) = load_models_and_scaler()

    # Encode SMILES to one-hot
    data = smiles_to_onehot(smiles_list, encoding_alphabet, largest_molecule_len)
    x = torch.tensor(data, dtype=dtype, device=device)  # [B, seq, alphabet]
    x = x.transpose(1, 2)  # [B, channels, seq] as in training

    with torch.no_grad():
        _, z_mean, _ = encoder(x)
        preds_scaled = prop_net(z_mean).cpu().numpy()

    # Back to physical units
    preds = scaler.inverse_transform(preds_scaled)

    # Wrap as DataFrame for readability
    return pd.DataFrame(preds, columns=property_columns)


if __name__ == "__main__":
    property_cols = ["Eea", "Egb", "EPS", "PE_I", "OPV"]

    # Example SMILES to test on
    test_smiles = [
        "*CC(*)C",
       "*CC(*)CC"
    ]

    preds_df = predict_properties(test_smiles, property_cols)
    print(preds_df)
