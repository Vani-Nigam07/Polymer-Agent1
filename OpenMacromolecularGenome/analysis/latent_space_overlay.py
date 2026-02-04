"""
Latent space PCA + Generated polymer overlay
============================================

This script:
1. Computes PCA on VAE latent means (z_mean) for a database
2. Saves PCA + scaler
3. Encodes generated SMILES using the SAME encoder
4. Projects generated polymers into the SAME PCA
5. Overlays generated points on latent PCA plot

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib 
import torch
import sys
sys.path.append('/home/vani/omgfine/OpenMacromolecularGenome')
import selfies as sf
from vae.encoder.torch import CNNEncoder
from vae.preprocess import get_selfie_and_smiles_encodings_for_dataset, multiple_selfies_to_hot

def pca_on_vae_latents(csv_path,
                       pretrained_dir,
                       smiles_col='product',
                       sample=None,
                       out_pdf='overlay_latents.pdf',
                       save_prefix='vae_latents',
                       device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_params = torch.load(os.path.join(pretrained_dir, 'vae_parameters.pth'), map_location=device)
    latent_dim = vae_params['latent_dimension']
    encoder = CNNEncoder(
        in_channels=vae_params['encoder_in_channels'],
        feature_dim=vae_params['encoder_feature_dim'],
        convolution_channel_dim=vae_params['encoder_convolution_channel_dim'],
        kernel_size=vae_params['encoder_kernel_size'],
        layer_1d=vae_params['encoder_layer_1d'],
        layer_2d=vae_params['encoder_layer_2d'],
        latent_dimension=latent_dim
    ).to(device)
    encoder.load_state_dict(torch.load(os.path.join(pretrained_dir, 'encoder.pth'), map_location=device))
    encoder.eval()

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[smiles_col]).reset_index(drop=True)
    if sample is not None and len(df) > sample:
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)

    selfies_list, _, _, _, _, _ = get_selfie_and_smiles_encodings_for_dataset(df[[smiles_col]])
    # convert to one-hot (may be memory heavy; sample reduces it)
    parent_dir = pretrained_dir  # adjust if encoding_alphabet stored elsewhere
    encoding_alphabet = torch.load(os.path.join(parent_dir, 'encoding_alphabet.pth'))
    largest_molecule_len = torch.load(os.path.join(parent_dir, 'largest_molecule_len.pth'))
    selfies_valid = []
    valid_indices = []
    for i, s in enumerate(selfies_list):
        if s is None: continue
        # simple token check skipped here; assume compatible or preprocess externally
        selfies_valid.append(s)
        valid_indices.append(i)
    if not selfies_valid:
        raise RuntimeError("No valid selfies found for encoding.")

    X_onehot = multiple_selfies_to_hot(selfies_valid, largest_molecule_len, encoding_alphabet)
    # match encoder input expectations:@OMG repo used data[:,1:,:]
    X_onehot = X_onehot[:, 1:, :]
    X_tensor = torch.tensor(X_onehot, dtype=torch.float32, device=device).transpose(1,2)  # (N, C, L)
    latents = []
    batch = 256
    with torch.no_grad():
        for i in range(0, X_tensor.size(0), batch):
            b = X_tensor[i:i+batch]
            _, z_mean, _ = encoder(b)
            latents.append(z_mean.cpu().numpy())
    Z_latent = np.vstack(latents)
    pca = PCA(n_components=2)
    scaler = StandardScaler()
    Z_pca = pca.fit_transform(scaler.fit_transform(Z_latent))

    Z_scaled = scaler.fit_transform(Z_latent)  # X = your feature matrix
    pca = PCA(n_components=min(Z_scaled.shape[1], 100)).fit(Z_scaled)  # up to 100 or full dim
    explained = pca.explained_variance_ratio_ * 100  # percent


    joblib.dump(pca, f"{save_prefix}_pca.joblib")
    joblib.dump(scaler, f"{save_prefix}_scaler.joblib")
    np.save(f"{save_prefix}_latent_db.npy", Z_latent)

   


    return  Z_pca, pca, df, encoder, encoding_alphabet, largest_molecule_len


def encode_generated_smiles(
    smiles,
    encoder,
    encoding_alphabet,
    largest_molecule_len,
    device
):
    """
    Encodes a generated SMILES into z_mean using SAME encoder.
    """

    selfie = sf.encoder(smiles)

    X = multiple_selfies_to_hot(
        [selfie],
        largest_molecule_len,
        encoding_alphabet
    )

    X = X[:, 1:, :]
    X = torch.tensor(X, dtype=torch.float32, device=device).transpose(1, 2)

    with torch.no_grad():
        _, z_mu, _ = encoder(X)

    return z_mu.cpu().numpy()



def project_to_pca(z_mu, pca_path, scaler_path):
    """
    Projects latent vector into saved PCA space.
    """

    pca = joblib.load(pca_path)
    scaler = joblib.load(scaler_path)

    z_scaled = scaler.transform(z_mu)
    z_pca = pca.transform(z_scaled)

    return z_pca


def plot_latent_with_generated(
    Z_pca,
    z_gen_pca,
    pca,
    out_prefix="latent_overlay"
):
    """
    ACS-compliant TOC graphic:
    - Size: 3.25 x 1.75 inches
    - Font: Sans-serif (Helvetica-like), 8 pt
    - Minimal text
    - Saved as TIF (300 dpi, RGB) and EPS
    """

    import matplotlib.pyplot as plt
    from matplotlib import rcParams

  
    # ACS STYLE SETTINGS

    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    })
  
    fig, ax = plt.subplots(
        figsize=(3.25, 1.75),   # inches (ACS requirement)
        dpi=300
    )
  
    
    # Create DataFrames for seaborn plotting
    db_df = pd.DataFrame({
        'PC1': Z_pca[:, 0],
        'PC2': Z_pca[:, 1],
        'type': 'database'
    })

    gen_df = pd.DataFrame({
        'PC1': z_gen_pca[:, 0],
        'PC2': z_gen_pca[:, 1],
        'type': 'generated'
    })

    # Plot database points with seaborn
    sns.scatterplot(
        data=db_df,
        x='PC1',
        y='PC2',
        s=3,
        alpha=0.25,
        color="blue",
        rasterized=True,
        ax=ax
    )

    # Plot generated points with seaborn
    sns.scatterplot(
        data=gen_df,
        x='PC1',
        y='PC2',
        s=70,
        marker="*",
        color="red",
        edgecolor="black",
        linewidth=0.5,
        style='type',
        ax=ax
    )

    ax.set_xlabel(
        "Principal Component 1", #f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
        labelpad=2
    )
    ax.set_ylabel(
        "Principal Component 2", #f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
        labelpad=2
    )
    ax.tick_params(direction="out")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(
        0.02, 0.98,
        "generated_smiles = *OCCOC(C)C(Br)CSO* ",  
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        ha="left"
    )

    plt.tight_layout(pad=0.2)

    tif_path = f"{out_prefix}.tif"
    eps_path = f"{out_prefix}.eps"

    plt.savefig(
        tif_path,
        dpi=300,
        format="tiff",
        pil_kwargs={"compression": "tiff_lzw"}
    )

    plt.savefig(
        eps_path,
        format="eps"
    )

    plt.close()

    print(f"ACS TOC graphic saved as:")
    print(f"    {tif_path}")
    print(f"    {eps_path}")


def encode_latents_from_csv(
    csv_path,
    pretrained_dir,
    smiles_col="product",
    target_property="Eea",
    sample=None,
    device=None
):
    """
    Returns:
        Z_latent      : (N, latent_dim)
        property_vals : (N,)
        df_used       : filtered dataframe (row-aligned)
    """

    import torch
    import os
    import pandas as pd
    import numpy as np
    from vae.encoder.torch import CNNEncoder
    from vae.preprocess import get_selfie_and_smiles_encodings_for_dataset, multiple_selfies_to_hot

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load + filter ONCE ----
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[smiles_col, target_property]).reset_index(drop=True)

    if sample is not None and len(df) > sample:
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)

    property_vals = df[target_property].values

    # ---- SELFIES ----
    selfies_list, _, _, _, _, _ = get_selfie_and_smiles_encodings_for_dataset(df[[smiles_col]])

    encoding_alphabet = torch.load(os.path.join(pretrained_dir, "encoding_alphabet.pth"))
    largest_molecule_len = torch.load(os.path.join(pretrained_dir, "largest_molecule_len.pth"))

    selfies_valid = []
    valid_idx = []

    for i, s in enumerate(selfies_list):
        if s is not None:
            selfies_valid.append(s)
            valid_idx.append(i)

    property_vals = property_vals[valid_idx]
    df = df.iloc[valid_idx].reset_index(drop=True)

    # ---- One-hot ----
    X = multiple_selfies_to_hot(selfies_valid, largest_molecule_len, encoding_alphabet)
    X = X[:, 1:, :]
    X = torch.tensor(X, dtype=torch.float32, device=device).transpose(1, 2)

    # ---- Encoder ----
    vae_params = torch.load(os.path.join(pretrained_dir, "vae_parameters.pth"), map_location=device)

    encoder = CNNEncoder(
        in_channels=vae_params["encoder_in_channels"],
        feature_dim=vae_params["encoder_feature_dim"],
        convolution_channel_dim=vae_params["encoder_convolution_channel_dim"],
        kernel_size=vae_params["encoder_kernel_size"],
        layer_1d=vae_params["encoder_layer_1d"],
        layer_2d=vae_params["encoder_layer_2d"],
        latent_dimension=vae_params["latent_dimension"],
    ).to(device)

    encoder.load_state_dict(torch.load(os.path.join(pretrained_dir, "encoder.pth"), map_location=device))
    encoder.eval()

    latents = []
    batch = 256

    with torch.no_grad():
        for i in range(0, X.size(0), batch):
            _, z_mu, _ = encoder(X[i:i+batch])
            latents.append(z_mu.cpu().numpy())

    Z_latent = np.vstack(latents)

    assert Z_latent.shape[0] == property_vals.shape[0]

    return Z_latent, property_vals, df


def find_property_correlated_latent_dims(
    Z_latent,
    property_vals,
    top_k=2,
    scale_latents=True
):
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    if scale_latents:
        Z = StandardScaler().fit_transform(Z_latent)
    else:
        Z = Z_latent.copy()

    corrs = np.array([
        np.corrcoef(Z[:, i], property_vals)[0, 1]
        for i in range(Z.shape[1])
    ])

    corrs = np.nan_to_num(corrs)
    top_dims = np.argsort(np.abs(corrs))[-top_k:]

    return top_dims, corrs

def plot_latent_property_ACS_EPS(
    Z_latent,
    property_vals,
    top_dims,
    z_gen=None,
    out_eps="latent_property.eps"
):
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from matplotlib import rcParams

    rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.linewidth": 0.6
    })

    Z = StandardScaler().fit_transform(Z_latent)
    d1, d2 = top_dims

    plt.figure(figsize=(3.25, 1.75), dpi=300)

    # Create DataFrame for seaborn plotting
    plot_df = pd.DataFrame({
        f'Latent dim {d1}': Z[:, d1],
        f'Latent dim {d2}': Z[:, d2],
        'property': property_vals
    })

    # Plot property-colored scatter with seaborn
    sc = sns.scatterplot(
        data=plot_df,
        x=f'Latent dim {d1}',
        y=f'Latent dim {d2}',
        hue='property',
        palette="cividis",
        s=4,
        alpha=0.4
    )

    if z_gen is not None:
        z_gen_scaled = StandardScaler().fit_transform(
            np.vstack([Z_latent, z_gen])
        )[-1]
        # Plot generated point with seaborn
        gen_df = pd.DataFrame({
            f'Latent dim {d1}': [z_gen_scaled[d1]],
            f'Latent dim {d2}': [z_gen_scaled[d2]],
            'type': ['generated']
        })
        sns.scatterplot(
            data=gen_df,
            x=f'Latent dim {d1}',
            y=f'Latent dim {d2}',
            color="red",
            s=70,
            marker="*",
            edgecolor="black",
            zorder=5
        )

    plt.xlabel(f"Latent dim {d1}")
    plt.ylabel(f"Latent dim {d2}")
    plt.text(
        0.02, 0.98,
        "Latent dims correlated with property",
        transform=plt.gca().transAxes,
        fontsize=7,
        va="top"
    )

    plt.tight_layout(pad=0.2)
    plt.savefig(out_eps, format="eps")
    plt.close()


if __name__ == "__main__":

    CSV = '/home/vani/omgfine/OpenMacromolecularGenome/data/OMG_TP/OMG_predicted_properties_train.csv'
    #PRETRAINED_DIR = "/home/vani/omgfine/OpenMacromolecularGenome/train/all/vae_optuna_10000_6000_3000_500_objective_1_10_10_1e-5_weight_decay"
    PRETRAINED_DIR  = "/home/vani/omgfine/OpenMacromolecularGenome/train/all/vae_optuna_10000_6000_3000_500_objective_1_10_10_1e-5_weight_decay/divergence_weight_4.345_latent_dim_152_learning_rate_0.002"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # #functions for PCA on latent space
    # Z_pca, pca, df, encoder, alphabet, max_len = pca_on_vae_latents(
    #     csv_path=CSV,
    #     pretrained_dir=PRETRAINED_DIR,
    #     sample=50000,
    #     device=device,
    #     save_prefix="latent"
    # )

    # # functions for Encode generated SMILES
    # generated_smiles = "*OCCOC(C)C(Br)CSO*" #polymer mentioned in the paper
    # z_gen = encode_generated_smiles(
    #     generated_smiles,
    #     encoder,
    #     alphabet,
    #     max_len,
    #     device
    # )

    # # function for Projections into PCA
    # z_gen_pca = project_to_pca(
    #     z_gen,
    #     "latent_pca.joblib",
    #     "latent_scaler.joblib"
    # )

    # # function for Plot overlay
    # plot_latent_with_generated(
    #     Z_pca,
    #     z_gen_pca,
    #     pca,
    #     out_prefix="latent_space_with_generated"
    # )
    Z_latent, property_vals, df = encode_latents_from_csv(CSV,PRETRAINED_DIR,target_property="PE_I")
    top_dims, corr = find_property_correlated_latent_dims(Z_latent,property_vals,top_k=2)
    plot_latent_property_ACS_EPS(Z_latent,property_vals,top_dims,out_eps="latent_property_PE_I.eps")
