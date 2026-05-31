
#  [BaratiLab/Polymer-Agent](https://github.com/BaratiLab/Polymer-Agent).
## Finetuning a model - Graph-Encoder, RNN-Decoder

This repository contains the fine-tuning module and core model architecture for the generative inverse-design component of **Polymer-Agent**. 
---

## Architecture


* **Framework:** Adapted from the bag-of-reactants paradigm of *Molecule Chef* ([Bradshaw et al.](https://github.com/john-bradshaw/molecule-chef)).
* **Sequence & Macromolecular Tokenization:** Inspired by *OpenMacromolecularGenome* ([The Jackson Laboratory](https://github.com/TheJacksonLab/OpenMacromolecularGenome)).

### Model Pipeline

```
[Polymer Input Strings/Graphs] ──> [Graph Neural Network (GNN) Encoder] ──> [Latent Representation] ──> [Recurrent Neural Network (RNN) Decoder] ──> [Generated Product Stream]

```

1. **Encoder:** A **Graph Neural Network (GNN)** that processes explicit structural topologies and connectivity mappings directly from polymer string configurations into dense latent embeddings.
2. **Decoder:** A **Recurrent Neural Network (RNN)** conditional decoder trained to generate viable polymer product structures from an optimized latent space.

---

## Functional Workflow & Selection Criteria

The fine-tuning loop optimizes generative pipelines based on empirical target parameters:

* **Reactant Optimization:** The generative trajectory initializes from a discrete "bag of reactants" explicitly filtered for structural viability.
* **Target Fitness:** Candidates are selected and prioritized using **Synthetic Accessibility (SA) Scores**, ensuring the generated macromolecular topologies maintain realistic downstream synthetic pathways.

---

## Core References & Citations

If you utilize this architecture or the associated fine-tuning checkpoints in your research, please cite the following foundational works:

* *Foundational Methodology:* [DOI Link](https://pubs.acs.org/doi/10.1021/acs.jcim.6c00343)
* *Component Module:* Part of the AI agent hosted at [BaratiLab/Polymer-Agent](https://github.com/BaratiLab/Polymer-Agent).
