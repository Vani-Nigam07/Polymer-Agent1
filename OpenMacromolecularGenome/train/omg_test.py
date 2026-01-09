import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as f
import matplotlib.pyplot as plt
from rdkit import Chem
from tqdm import tqdm
import sys
from typing import List, Dict, Any, Tuple, Optional

# Assuming these paths are necessary for your environment
sys.path.append('/home/vani/omgfine/OpenMacromolecularGenome')
from vae.decoder.torch import Decoder
from vae.encoder.torch import CNNEncoder
from vae.preprocess import get_selfie_and_smiles_encodings_for_dataset, multiple_selfies_to_hot
from vae.property_predictor.torch import PropertyNetworkPredictionModule
from vae.utils.save import VAEParameters
import selfies as sf

# --- Helper Functions for Modularization ---

def load_multiproperty_vae(finetuned_model_path: str, parent_dir: str, device: torch.device, dtype: torch.dtype,
                           property_names: Optional[List[str]] = None,
                           optimization_weights: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Loads VAE components, property network, parameters, and encoding data.

    Args:
        finetuned_model_path: Path to fine-tuned multi-property model directory.
        parent_dir: Path to the directory containing encoding data (alphabet, max_len).
        device: Torch device (e.g., cuda or cpu).
        dtype: Torch data type (e.g., torch.float32).
        property_names: Optional list of property names to override the saved ones.
        optimization_weights: Optional weights for the optimization objective.

    Returns:
        A dictionary containing all loaded components and parameters.
    """
    # Load VAE parameters
    vae_parameters = torch.load(os.path.join(finetuned_model_path, 'vae_parameters_multiproperty.pth'),
                                map_location=device)
    latent_dim = vae_parameters['latent_dimension']
    property_dim = vae_parameters['property_dim']

    # Resolve and set property names
    resolved_names = vae_parameters.get('property_names', ['Eea', 'Egb', 'EPS', 'PE_I', 'OPV'])
    if property_names is not None:
        if len(property_names) != property_dim:
            raise ValueError(f"property_names length ({len(property_names)}) != property_dim ({property_dim})")
        resolved_names = property_names
        vae_parameters['property_names'] = property_names
        # Persist the override
        torch.save(vae_parameters, os.path.join(finetuned_model_path, 'vae_parameters_multiproperty.pth'))
    print(f"Property names: {resolved_names}")

    # Set optimization weights
    if optimization_weights is None:
        optimization_weights = vae_parameters['property_weights']
    optimization_weights = torch.tensor(optimization_weights, dtype=dtype, device=device)
    print(f"Optimizing with {property_dim} properties, weights: {optimization_weights}")

    # Load encoding data
    encoding_alphabet = torch.load(os.path.join(parent_dir, 'encoding_alphabet.pth'))
    largest_molecule_len = torch.load(os.path.join(parent_dir, 'largest_molecule_len.pth'))

    # Initialize and load models
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

    # Load weights and set to evaluation mode
    encoder.load_state_dict(torch.load(os.path.join(finetuned_model_path, 'encoder.pth'), map_location=device))
    decoder.load_state_dict(torch.load(os.path.join(finetuned_model_path, 'decoder.pth'), map_location=device))
    property_network.load_state_dict(torch.load(os.path.join(finetuned_model_path, 'property_predictor_multiproperty.pth'), map_location=device))

    encoder.eval()
    decoder.eval()
    property_network.eval()

    # Load property scaler
    property_scaler = torch.load(os.path.join(finetuned_model_path, 'property_scaler.pth'), weights_only=False)

    return {
        'encoder': encoder,
        'decoder': decoder,
        'property_network': property_network,
        'property_scaler': property_scaler,
        'latent_dim': latent_dim,
        'property_dim': property_dim,
        'encoding_alphabet': encoding_alphabet,
        'largest_molecule_len': largest_molecule_len,
        'optimization_weights': optimization_weights,
        'property_names': resolved_names,
        'device': device,
        'dtype': dtype,
    }


def optimize_latents(property_network: torch.nn.Module, latent_dim: int, optimization_weights: torch.Tensor,
                     device: torch.device, dtype: torch.dtype, num_seeds: int, num_steps: int, step_size: float
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs gradient ascent in the latent space to optimize for the multi-property objective.

    Args:
        property_network: The loaded property prediction network.
        latent_dim: Dimension of the latent space.
        optimization_weights: Weights for each property objective.
        device: Torch device.
        dtype: Torch data type.
        num_seeds: Number of random starting points.
        num_steps: Number of gradient ascent steps per seed.
        step_size: Size of each optimization step.

    Returns:
        A tuple of (optimized_latents (numpy array), optimized_properties (numpy array)).
    """
    np.random.seed(42)
    latent_seeds = torch.randn(num_seeds, latent_dim, dtype=dtype, device=device)

    optimized_latents = []
    optimized_properties = []

    print(f"Starting optimization with {num_seeds} seeds, {num_steps} steps each...")
    pbar = tqdm(range(num_seeds), total=num_seeds)

    for seed_idx in pbar:
        z_current = latent_seeds[seed_idx].clone()
        z_current.requires_grad = True

        best_objective_val = -float('inf')

        # Gradient ascent loop
        for step in range(num_steps):
            # Get property predictions
            property_preds = property_network(z_current.unsqueeze(0))  # [1, property_dim]

            # Weighted objective (sum of weighted predictions)
            objective = torch.sum(property_preds * optimization_weights)

            # Keep track of the best objective for monitoring
            if objective.item() > best_objective_val:
                best_objective_val = objective.item()

            # Compute gradients
            if z_current.grad is not None:
                z_current.grad.zero_()
            objective.backward()

            # Gradient ascent step (normalized by gradient norm)
            with torch.no_grad():
                gradient = z_current.grad
                if gradient is not None:
                    # Normalized step (optional, but robust)
                    z_current = z_current + step_size * gradient / (torch.norm(gradient) + 1e-8)
                else:
                    break

        # Store final result
        final_preds = property_network(z_current.unsqueeze(0))
        optimized_latents.append(z_current.detach().cpu().numpy())
        optimized_properties.append(final_preds.detach().cpu().numpy())

        if seed_idx % 20 == 0:
            pbar.set_postfix({'Latest objective': f"{objective.item():.3f}"})

    # Convert to numpy arrays
    optimized_latents = np.array(optimized_latents)
    optimized_properties = np.array(optimized_properties).squeeze()

    return optimized_latents, optimized_properties


def decode_latents_to_polymers(optimized_latents: np.ndarray, optimized_properties: np.ndarray,
                               results_dict: Dict[str, Any], save_directory: str) -> pd.DataFrame:
    """
    Decodes latent vectors into polymers, calculates final properties, and saves the results.

    Args:
        optimized_latents: Array of optimized latent vectors.
        optimized_properties: Array of predicted (scaled) properties for the latents.
        results_dict: Dictionary containing decoder, scaler, alphabet, etc.
        save_directory: Directory to save the output files.

    Returns:
        DataFrame containing the valid, optimized polymers and their properties.
    """
    decoder = results_dict['decoder']
    property_scaler = results_dict['property_scaler']
    encoding_alphabet = results_dict['encoding_alphabet']
    largest_molecule_len = results_dict['largest_molecule_len']
    latent_dim = results_dict['latent_dim']
    property_dim = results_dict['property_dim']
    property_names = results_dict['property_names']
    optimization_weights = results_dict['optimization_weights'].cpu().numpy()
    device = results_dict['device']
    dtype = results_dict['dtype']

    print("Decoding optimized latent vectors to polymers...")
    generated_polymers = []
    valid_polymers = []

    with torch.no_grad():
        for latent_vec in optimized_latents:
            z_tensor = torch.tensor(latent_vec, dtype=dtype, device=device)

            # VAE decoder generation
            hidden = decoder.init_hidden(z_tensor.unsqueeze(0))
            x_input = torch.zeros(size=(1, 1, len(encoding_alphabet)), dtype=dtype, device=device)
            gathered_indices = []

            for _ in range(largest_molecule_len):
                out_one_hot_line, hidden = decoder(x=x_input, hidden=hidden)
                x_hat_prob = f.softmax(out_one_hot_line, dim=-1)
                x_hat_indices = x_hat_prob.argmax(dim=-1)
                x_input = f.one_hot(x_hat_indices, num_classes=len(encoding_alphabet)).to(torch.float)
                gathered_indices.append(x_hat_indices.squeeze().item())

            # Convert indices to SMILES via SELFIES
            gathered_atoms = ''.join([encoding_alphabet[idx] for idx in gathered_indices])
            generated_selfies = '[*]' + gathered_atoms.replace('[nop]', '')
            smiles_generated = None
            is_valid = False
            try:
                smiles_generated = sf.decoder(generated_selfies)
                mol = Chem.MolFromSmiles(smiles_generated)
                if mol is not None:
                    smiles_generated = Chem.MolToSmiles(mol)
                    is_valid = True
            except:
                pass  # Keep defaults: smiles_generated=None, is_valid=False

            generated_polymers.append(smiles_generated)
            valid_polymers.append(is_valid)

    # Create results dataframe
    results_df = pd.DataFrame({
        'polymer_smiles': generated_polymers,
        'is_valid': valid_polymers,
    })

    # Unscale and add optimized property columns
    optimized_properties_unscaled = property_scaler.inverse_transform(optimized_properties)
    for i in range(property_dim):
        results_df[property_names[i]] = optimized_properties_unscaled[:, i]

    # Add combined objective score
    weighted_objectives = np.sum(optimized_properties_unscaled * optimization_weights, axis=1)
    results_df['weighted_objective'] = weighted_objectives

    # --- Save Results and Plot ---
    os.makedirs(save_directory, exist_ok=True)
    valid_mask = results_df['is_valid']
    results_df_valid = results_df[valid_mask].sort_values('weighted_objective', ascending=False)

    # Save CSV
    results_df_valid.to_csv(os.path.join(save_directory, 'optimized_polymers_multiproperty.csv'), index=False)

    # Plot distributions 
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

    print(f"Optimization complete! Generated {len(results_df_valid)} valid polymers.")
    if not results_df_valid.empty:
        print(f"Best weighted objective: {results_df_valid['weighted_objective'].max():.3f}")
    print(f"Results saved to {save_directory}")

    return results_df_valid


# --- Main Orchestration Function ---

def optimize_multiproperty_latent_space(finetuned_model_path: str,
                                       optimization_weights: Optional[List[float]] = None,
                                       property_names: Optional[List[str]] = None,
                                       num_seeds: int = 100, num_steps: int = 50,
                                       step_size: float = 0.1,
                                       save_directory: str = "./optimized_polymers") -> pd.DataFrame:
    """
    Perform latent space optimization for multi-property objectives.

    Args:
        finetuned_model_path: Path to fine-tuned multi-property model directory.
        optimization_weights: Weights for each property (higher = more important).
        property_names: Optional list of property names to override defaults.
        num_seeds: Number of random starting points in latent space.
        num_steps: Number of gradient ascent steps per seed.
        step_size: Size of each optimization step.
        save_directory: Where to save results.
    
    Returns:
        DataFrame containing the valid, optimized polymers and their properties.
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print(f"Using device: {device}")

    # **HARDCODED PATHS**: These should ideally be passed as arguments or configuration files.
    # Keeping them here as they were in the original un-modularized code:
    parent_dir = '/home/vani/omgfine/OpenMacromolecularGenome/train/all/vae_optuna_10000_6000_3000_500_objective_1_10_10_1e-5_weight_decay'
    
    # 1. Load VAE and components
    results_dict = load_multiproperty_vae(
        finetuned_model_path,
        parent_dir,
        device,
        dtype,
        property_names=property_names,
        optimization_weights=optimization_weights
    )
    
    # 2. Optimize Latents
    optimized_latents, optimized_properties = optimize_latents(
        property_network=results_dict['property_network'],
        latent_dim=results_dict['latent_dim'],
        optimization_weights=results_dict['optimization_weights'],
        device=device,
        dtype=dtype,
        num_seeds=num_seeds,
        num_steps=num_steps,
        step_size=step_size
    )
    
    # 3. Decode Latents to Polymers and Save Results
    results_df_valid = decode_latents_to_polymers(
        optimized_latents,
        optimized_properties,
        results_dict,
        save_directory
    )

    return results_df_valid


if __name__ == '__main__':
    # Example Usage (as in the original code)
    finetuned_path ='/home/vani/omgfine/OpenMacromolecularGenome/finetuned_multiproperty_model'
    if not os.path.exists(finetuned_path):
        print(f"Error: Fine-tuned model not found at {finetuned_path}")
        print("Please run finetune_multiproperty_vae.py first to create the fine-tuned model")
        exit(1)

    save_dir = './optimized_multiproperty_results'

    # Properties: ['Eea', 'Egb', 'EPS', 'PE_I', 'OPV']
    custom_names = ['Eea', 'Egb', 'EPS', 'PE_I', 'OPV']
    custom_weights = [0.1, 0.5, 0.1, 0.2, 0.1] 

    print("Starting multi-property latent space optimization...")
    print(f"Property weights: {custom_weights}")

    results = optimize_multiproperty_latent_space(
        finetuned_model_path=finetuned_path,
        optimization_weights=custom_weights,
        property_names=custom_names,
        num_seeds=200,
        num_steps=100,
        step_size=0.1,
        save_directory=save_dir
    )

    print("\nTop 5 optimized polymers:")
    print(results.head())