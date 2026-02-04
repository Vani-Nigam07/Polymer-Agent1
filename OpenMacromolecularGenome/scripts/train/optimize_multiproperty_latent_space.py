import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as f
import matplotlib.pyplot as plt
from rdkit import Chem
from tqdm import tqdm
import sys

sys.path.append('/home/vani/omgfine/OpenMacromolecularGenome')
from vae.decoder.torch import Decoder
from vae.encoder.torch import CNNEncoder
from vae.preprocess import get_selfie_and_smiles_encodings_for_dataset, multiple_selfies_to_hot
from vae.property_predictor.torch import PropertyNetworkPredictionModule
from vae.utils.save import VAEParameters
import selfies as sf


def optimize_multiproperty_latent_space(finetuned_model_path, optimization_weights=None,
                                       num_seeds=100, num_steps=50, step_size=0.1,
                                       save_directory="./optimized_polymers"):
    """
    Perform latent space optimization for multi-property objectives.

    Args:
        finetuned_model_path: Path to fine-tuned multi-property model directory
        optimization_weights: Weights for each property (higher = more important)
        num_seeds: Number of random starting points in latent space
        num_steps: Number of gradient ascent steps per seed
        step_size: Size of each optimization step
        save_directory: Where to save results
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print(f"Using device: {device}")

    # Load fine-tuned model parameters
    vae_parameters = torch.load(os.path.join(finetuned_model_path, 'vae_parameters_multiproperty.pth'),
                               map_location=device)
    latent_dim = vae_parameters['latent_dimension']
    property_dim = vae_parameters['property_dim']
    
    
    # Load property names if available, otherwise use generic names
    property_names =  ['Eea', 'Egb', 'EPS', 'PE_I', 'OPV']
        # Resolve property names: explicit override -> saved -> generic defaults
    saved_names = vae_parameters.get('property_names')
    if property_names is not None:
        if len(property_names) != property_dim:
            raise ValueError(f"property_names length ({len(property_names)}) != property_dim ({property_dim})")
        vae_parameters['property_names'] = property_names
        # persist the override so downstream code finds the names
        torch.save(vae_parameters, os.path.join(finetuned_model_path, 'vae_parameters_multiproperty.pth'))
        resolved_names = property_names
    elif saved_names is not None:
        resolved_names = saved_names
    print(f"Property names: {resolved_names}")

    if optimization_weights is None:
        optimization_weights = vae_parameters['property_weights']
    optimization_weights = torch.tensor(optimization_weights, dtype=dtype, device=device)

    print(f"Optimizing with {property_dim} properties, weights: {optimization_weights}")

    # Load encoding data from parent directory (shared)
 

    parent_dir = '/home/vani/omgfine/OpenMacromolecularGenome/train/all/vae_optuna_10000_6000_3000_500_objective_1_10_10_1e-5_weight_decay'


    encoding_alphabet = torch.load(os.path.join(parent_dir, 'encoding_alphabet.pth'))
    largest_molecule_len = torch.load(os.path.join(parent_dir, 'largest_molecule_len.pth'))

    # Initialize and load encoder and decoder
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

    property_network = PropertyNetworkPredictionModule(
        latent_dim=latent_dim,
        property_dim=property_dim,
        property_network_hidden_dim_list=vae_parameters['property_network_hidden_dim_list'],
        dtype=dtype,
        device=device,
        weights=vae_parameters['property_weights']
    ).to(device)

    # Load weights (encoder/decoder from pre-trained, property network from fine-tuning)
    encoder.load_state_dict(torch.load(os.path.join(finetuned_model_path, 'encoder.pth'),
                                     map_location=device))
    decoder.load_state_dict(torch.load(os.path.join(finetuned_model_path, 'decoder.pth'),
                                     map_location=device))
    property_network.load_state_dict(torch.load(os.path.join(finetuned_model_path, 'property_predictor_multiproperty.pth'),
                                               map_location=device))

    # Set to evaluation mode
    encoder.eval()
    decoder.eval()
    property_network.eval()

    # Load property scaler
    property_scaler = torch.load(os.path.join(finetuned_model_path, 'property_scaler.pth'), weights_only=False)

    # Generate random starting points in latent space
    np.random.seed(42)
    latent_seeds = torch.randn(num_seeds, latent_dim, dtype=dtype, device=device)

    # Store results
    optimized_latents = []
    optimized_properties = []

    print(f"Starting optimization with {num_seeds} seeds, {num_steps} steps each...")

    pbar = tqdm(range(num_seeds), total=num_seeds)

    for seed_idx in pbar:
        z_current = latent_seeds[seed_idx].clone()
        z_current.requires_grad = True

        # Gradient ascent for property optimization
        for step in range(num_steps):
            # Get property predictions
            property_preds = property_network(z_current.unsqueeze(0))  # [1, property_dim]

            # Create weighted objective (sum of weighted predictions)
            objective = torch.sum(property_preds * optimization_weights)

            # Compute gradients
            if z_current.grad is not None:
                z_current.grad.zero_()
            objective.backward()

            # Gradient ascent step
            with torch.no_grad():
                gradient = z_current.grad
                if gradient is not None:
                    z_current = z_current + step_size * gradient / torch.norm(gradient)
                else:
                    break  # No gradient, optimization stuck

        # Store final result
        final_preds = property_network(z_current.unsqueeze(0))
        optimized_latents.append(z_current.detach().cpu().numpy())
        optimized_properties.append(final_preds.detach().cpu().numpy())

        if seed_idx % 20 == 0:
            pbar.set_postfix({'Best objective': f"{objective.item():.3f}"})

    # Convert to numpy arrays
    optimized_latents = np.array(optimized_latents)
    optimized_properties = np.array(optimized_properties).squeeze()

    # Decode optimized latent vectors to molecular structures
    print("Decoding optimized latent vector")
    generated_polymers = []
    valid_polymers = []

    with torch.no_grad():
        for latent_vec in optimized_latents:
            z_tensor = torch.tensor(latent_vec, dtype=dtype, device=device)

            # Decode using the VAE decoder
            hidden = decoder.init_hidden(z_tensor.unsqueeze(0))
            out_one_hot = torch.zeros(size=(1, largest_molecule_len, len(encoding_alphabet)),
                                      dtype=dtype, device=device)
            nop_tensor = -torch.ones(size=(1,), dtype=dtype, device=device)
            asterisk_tensor = -torch.ones(size=(1,), dtype=dtype, device=device)
            x_input = torch.zeros(size=(1, 1, len(encoding_alphabet)), dtype=dtype, device=device)

            # Generation loop (simplified version)
            gathered_indices = []
            for seq_index in range(largest_molecule_len):
                out_one_hot_line, hidden = decoder(x=x_input, hidden=hidden)
                x_hat_prob = f.softmax(out_one_hot_line, dim=-1)
                x_hat_indices = x_hat_prob.argmax(dim=-1)
                x_input = f.one_hot(x_hat_indices, num_classes=len(encoding_alphabet)).to(torch.float)
                gathered_indices.append(x_hat_indices.squeeze().item())

            # Convert to SMILES
            gathered_atoms = ''
            for idx in gathered_indices:
                gathered_atoms += encoding_alphabet[idx]

            generated_molecule = gathered_atoms.replace('[nop]', '')
            generated_molecule = '[*]' + generated_molecule
            try:
                smiles_generated = sf.decoder(generated_molecule)
                mol = Chem.MolFromSmiles(smiles_generated)
                if mol is not None:
                    canonical_smiles = Chem.MolToSmiles(mol)
                    generated_polymers.append(canonical_smiles)
                    valid_polymers.append(True)
                else:
                    generated_polymers.append(None)
                    valid_polymers.append(False)
            except:
                generated_polymers.append(None)
                valid_polymers.append(False)

    # Create results dataframe
    results_df = pd.DataFrame({
        'polymer_smiles': generated_polymers,
        'is_valid': valid_polymers,
    })

    # Add optimized property columns using actual property names
    optimized_properties_unscaled = property_scaler.inverse_transform(optimized_properties)
    for i in range(property_dim):
        results_df[property_names[i]] = optimized_properties_unscaled[:, i]

    # Add combined objective score
    weighted_objectives = np.sum(optimized_properties_unscaled * np.array(optimization_weights.cpu()),
                               axis=1)
    results_df['weighted_objective'] = weighted_objectives

    # Save results
    os.makedirs(save_directory, exist_ok=True)

    # Save only valid polymers
    valid_mask = results_df['is_valid']
    results_df_valid = results_df[valid_mask].sort_values('weighted_objective', ascending=False)

    results_df_valid.to_csv(os.path.join(save_directory, 'optimized_polymers_multiproperty.csv'), index=False)

    # Plot property distributions using actual property names
    plt.figure(figsize=(12, 8))
    for i in range(property_dim):
        plt.subplot(2, (property_dim + 1) // 2, i + 1)
        valid_props = optimized_properties_unscaled[valid_mask, i]
        plt.hist(valid_props, bins=20, alpha=0.7)
        plt.xlabel(property_names[i])
        plt.ylabel('Frequency')
        plt.title(f'{property_names[i]} Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, 'property_distributions_multiproperty.png'))
    plt.close()

    print(f"Optimization complete!")
    print(f"Generated {len(results_df_valid)} valid polymers")
    print(f"Best weighted objective: {results_df_valid['weighted_objective'].max():.3f}")
    print(f"Results saved to {save_directory}")

    return results_df_valid


if __name__ == '__main__':
    # Check for fine-tuned model
    finetuned_path ='/home/vani/omgfine/OpenMacromolecularGenome/finetuned_multiproperty_model'
    if not os.path.exists(finetuned_path):
        print(f"Error: Fine-tuned model not found at {finetuned_path}")
        print("Please run finetune_multiproperty_vae.py first to create the fine-tuned model")
        exit(1)

    save_dir = './optimized_multiproperty_results'

    # Customize weights for your specific objectives
    # Higher weight = more important during optimization
    # Properties: ['Eea', 'Egb', 'EPS', 'PE_I', 'OPV']
    custom_weights = [0.1, 0.5, 0.1, 0.2, 0.1]  # Equal weights by default

    print("Starting multi-property latent space optimization...")
    print(f"Property weights: {custom_weights}")

    results = optimize_multiproperty_latent_space(
        finetuned_model_path=finetuned_path,
        optimization_weights=custom_weights,
        num_seeds=200,
        num_steps=100,
        step_size=0.1,
        save_directory=save_dir
    )

    print("\nTop 5 optimized polymers:")
    print(results.head())
