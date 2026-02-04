import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import torch
import sys
sys.path.append('/home/vani/omgfine/OpenMacromolecularGenome')
from vae.encoder.torch import CNNEncoder
from vae.preprocess import get_selfie_and_smiles_encodings_for_dataset, multiple_selfies_to_hot

def pca_on_properties(csv_path,
                      property_columns=None,
                      sample=None,
                      out_png='pca_properties.png'):
    df = pd.read_csv(csv_path)
    if property_columns is None:
        # choose numeric columns if not provided
        property_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df.dropna(subset=property_columns)
    if sample is not None and len(df) > sample:
        df = df.sample(n=sample, random_state=42)
    X = df[property_columns].values
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X_scaled)

    # Create DataFrame for seaborn plotting
    pca_df = pd.DataFrame({
        'PC1': Z[:, 0],
        'PC2': Z[:, 1]
    })
    if len(property_columns) > 0:
        pca_df[property_columns[0]] = df[property_columns[0]].values

    plt.figure(figsize=(8,6))
    cmap = sns.color_palette("cividis", as_cmap=True)

    # Use seaborn scatterplot
    if len(property_columns) > 0:
        sc = sns.scatterplot(
            data=pca_df,
            x='PC1',
            y='PC2',
            hue=property_columns[0],
            palette=cmap,
            s=8,
            alpha=0.8
        )
        plt.colorbar(sc.collections[0], label=property_columns[0])
    else:
        sns.scatterplot(
            data=pca_df,
            x='PC1',
            y='PC2',
            s=8,
            alpha=0.8
        )

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title('PCA on properties')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return pca, Z, df

def pca_on_vae_latents(csv_path,
                       pretrained_dir,
                       smiles_col='product',
                       sample=None,
                       out_png='pca_latents.png',
                       cumulative_variance_png='pca_cumulative_variance.png',
                       scree_plot_png='pca_scree_plot.png',
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
    Z_pca = pca.fit_transform(StandardScaler().fit_transform(Z_latent))

    Z_scaled = StandardScaler().fit_transform(Z_latent)  # X = your feature matrix
    pca = PCA(n_components=min(Z_scaled.shape[1], 100)).fit(Z_scaled)  # up to 100 or full dim
    explained = pca.explained_variance_ratio_ * 100  # percent

    # scree plot
    plt.figure(figsize=(6,4))
    scree_data = pd.DataFrame({
        'Principal component': np.arange(1, len(explained)+1),
        'Explained variance (%)': explained
    })
    sns.lineplot(data=scree_data, x='Principal component', y='Explained variance (%)', marker='o', markersize=4)
    plt.xlabel('Principal component')
    plt.ylabel('Explained variance (%)')
    plt.title('Scree plot')
    plt.grid(True)
    plt.savefig(scree_plot_png, dpi=200)
    plt.close()

    # cumulative variance
    cum = np.cumsum(explained)
    cum_data = pd.DataFrame({
        'Number of components': np.arange(1, len(cum)+1),
        'Cumulative explained variance (%)': cum
    })
    plt.figure(figsize=(6,4))
    sns.lineplot(data=cum_data, x='Number of components', y='Cumulative explained variance (%)', marker='o', markersize=4)
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance (%)')
    plt.axhline(80, color='red', linestyle='--', label='80%')
    plt.legend()
    plt.grid(True)
    plt.savefig(cumulative_variance_png, dpi=200)
    plt.close()

    # find components needed for 80% (or other threshold)
    threshold = 80.0
    n_needed = np.searchsorted(cum, threshold) + 1
    print(f"Components needed to reach {threshold}% variance: {n_needed}")

    # PCA scatter plot using seaborn
    pca_df = pd.DataFrame({
        'Principal Component 1': Z_pca[:, 0],
        'Principal Component 2': Z_pca[:, 1]
    })
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=pca_df, x='Principal Component 1', y='Principal Component 2', s=8, alpha=0.5)
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title('PCA on VAE latent space (z_mean)')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return pca, Z_pca, df


def pca_scatter_colored(csv_path, features, color_by, sample=None,
                        n_components=2, scale=True, ax=None,
                        color_log=False, quantile_bins=None,
                        cmap='cividis',
                        alpha=0.7, s=8):
    """
    Compute PCA on `features` and plot PC1 vs PC2 colored by `color_by`.
    """

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=features + [color_by]).reset_index(drop=True)

    if sample is not None and len(df) > sample:
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)

    X = df[features].values.astype(float)
    if scale:
        X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X)

    pc1_pct = pca.explained_variance_ratio_[0] * 100
    pc2_pct = pca.explained_variance_ratio_[1] * 100

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    color_vals = df[color_by].values
    if color_log:
        color_vals = np.log1p(np.maximum(color_vals, 0.0))

    # Create DataFrame for seaborn plotting
    pca_df = pd.DataFrame({
        'Principal Component 1': Z[:, 0],
        'Principal Component 2': Z[:, 1],
        color_by: color_vals
    })

    # ===== Continuous coloring =====
    if quantile_bins is None:
        sc = sns.scatterplot(
            data=pca_df,
            x='Principal Component 1',
            y='Principal Component 2',
            hue=color_by,
            palette=cmap,
            s=s,
            alpha=alpha,
            ax=ax
        )
        plt.colorbar(sc.collections[0], ax=ax, fraction=0.046, pad=0.04)

    # ===== Quantile / categorical coloring =====
    else:
        labels = pd.qcut(
            color_vals,
            q=quantile_bins,
            labels=False,
            duplicates='drop'
        )
        pca_df['quantile_labels'] = labels

        palette = sns.color_palette(
            "tab10" if quantile_bins <= 10 else "tab20",
            n_colors=len(np.unique(labels))
        )

        sns.scatterplot(
            data=pca_df,
            x='Principal Component 1',
            y='Principal Component 2',
            hue='quantile_labels',
            palette=palette,
            s=s,
            alpha=alpha,
            ax=ax,
            legend=True
        )

        ax.legend(
            title=f'{color_by} quantiles',
            fontsize=8,
            title_fontsize=9,
            loc='best'
        )

    ax.set_xlabel(f'PC1 ({pc1_pct:.1f}%)')
    ax.set_ylabel(f'PC2 ({pc2_pct:.1f}%)')
    ax.set_title(color_by)

    return pca, Z, df


# def pca_scatter_colored(csv_path, features, color_by, sample=None,
#                         n_components=2, scale=True, ax=None,
#                         color_log=False, quantile_bins=None,
#                         cmap='cividis', 
#                         alpha=0.7, s=8):
#     """
#     Compute PCA on `features` and plot 2D PCA colored by `color_by` column.
#     - path: dataframe
#     - features: list of columns to run PCA on (numeric)
#     - color_by: column name to use for color (continuous or categorical)
#     - sample: int max rows to sample (None = use all)
#     - color_log: if True apply np.log1p to color values before plotting
#     - quantile_bins: int -> bin continuous color into quantile_bins categories
#     """
#     df = pd.read_csv(csv_path)
#     df = df.dropna(subset=features + [color_by]).reset_index(drop=True)
#     if sample is not None and len(df) > sample:
#         df = df.sample(n=sample, random_state=42).reset_index(drop=True)

#     X = df[features].values.astype(float)
#     if scale:
#         X = StandardScaler().fit_transform(X)

#     pca = PCA(n_components=n_components)
#     Z = pca.fit_transform(X)
#     pc1_pct = pca.explained_variance_ratio_[0] * 100
#     pc2_pct = pca.explained_variance_ratio_[1] * 100

#     color_vals = df[color_by].values
#     if color_log:
#         # handle negatives / zeros safely
#         color_vals = np.log1p(np.maximum(color_vals, 0.0))

#     if quantile_bins is not None and np.issubdtype(color_vals.dtype, np.number):
#         # create categorical labels by quantiles
#         labels, bins = pd.qcut(color_vals, q=quantile_bins, labels=False, retbins=True, duplicates='drop')
#         cmap_used = sns.color_palette("tab10", as_cmap=False) if quantile_bins <= 10 else cmap
#         if ax is None:
#             fig, ax = plt.subplots(figsize=(8,6))
#             sc = ax.scatter(Z[:,0], Z[:,1], c=color_vals, cmap=cmap, s=s, alpha=alpha)

#             cbar = plt.colorbar(sc, ax=ax, ticks=bins, label=color_by)
#     else:
#         if ax is None:
#             fig, ax = plt.subplots(figsize=(8,6))
#             sc = ax.scatter(Z[:,0], Z[:,1], c=color_vals, cmap=cmap, s=s, alpha=alpha)

            

#     ax.set_xlabel(f'Principal Component 1 ({pc1_pct:.1f}%)')
#     ax.set_ylabel(f'Principal Component 2 ({pc2_pct:.1f}%)')
#     ax.set_title(f'PCA colored by {color_by}')
#     plt.tight_layout()
#     return pca, Z, df

def make_pca_collage(csv_path, features, properties,
                     sample=50000,
                     out_pdf='Figure_3_PCA_properties.png'):

    fig, axes = plt.subplots(
        2, 3,
        figsize=(10, 7),
        constrained_layout=True
    )

    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    for ax, prop, label in zip(axes.flatten(), properties, panel_labels):
        pca_scatter_colored(
            csv_path=csv_path,
            features=features,
            color_by=prop,
            sample=sample,
            ax=ax
        )

        ax.text(
            0.02, 0.98, label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight='bold',
            va='top'
        )

    for ax in axes.flatten()[len(properties):]:
        ax.axis('off')

    fig.savefig(out_pdf, dpi=600, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved PCA collage to: {out_pdf}")


if __name__ == '__main__':
    CSV = '/home/vani/omgfine/OpenMacromolecularGenome/data/OMG_TP/OMG_predicted_properties_train.csv'
    # Quick PCA on a few property columns (fast)
    props = ['Eea','Egb','EPS','PE_I','OPV']  # use whichever columns exist in your CSV
    # pca_on_properties(CSV, property_columns=props, sample=50000, out_png='pca_properties.png')

    # Optional: PCA on VAE latents (slower, needs pretrained encoder files)
    PRETRAINED_DIR = '/home/vani/omgfine/OpenMacromolecularGenome/train/all/vae_optuna_10000_6000_3000_500_objective_1_10_10_1e-5_weight_decay/divergence_weight_4.345_latent_dim_152_learning_rate_0.002'
    # Uncomment to run (ensure encoder.pth and vae_parameters.pth exist)
    #pca_on_vae_latents(CSV, PRETRAINED_DIR, sample=200000, out_png='pca_latents_200000.png', 
    #                  cumulative_variance_png='pca_cumulative_variance_200000.png', scree_plot_png='pca_scree_plot_200000.png')
    # pca_scatter_colored(CSV, features=['Eea','Egb','EPS','PE_I','OPV'],
    #                  color_by='Eea', sample=50000)
    make_pca_collage(
        csv_path=CSV,
        features=['Eea','Egb','EPS','PE_I','OPV'],
        properties=['Eea','Egb','EPS','PE_I','OPV'],
        sample=50000,
        out_pdf='PCA_properties.png'
    )
