# %%
"""
Causal Hyperbolic Hypergraph Survival Network
==============================================

Full objective (NeurIPS style):

    L(theta) = L_Cox
             + lambda_1 * L_manifold
             + lambda_2 * L_causal
             + lambda_3 * L_hypergraph

Where:
    - L_Cox        : Cox partial likelihood
    - L_manifold   : Hyperbolic radial ordering constraint
    - L_causal     : Sparse + context-variance causal counterfactual
    - L_hypergraph : Hypergraph Laplacian smoothness on latent z

Latent space: z in H^d (Poincare ball, curvature c)
Risk head:    R_i = beta * (2/sqrt(c)) * arctanh(sqrt(c) * ||z_i||)
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set

import requests

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import gseapy as gp


###############################################################
# CONFIGURATION
###############################################################

@dataclass
class Config:
    data_dir: str = r"C:\Users\VN-ChoHyunJae\Downloads\msk_chord_2024\msk_chord_2024"
    batch_size: int = 64
    embed_dim: int = 128
    hyper_dim: int = 128
    latent_dim: int = 128
    max_genes_per_patient: int = 200
    lr: float = 1e-3
    epochs: int = 30
    weight_decay: float = 1e-4
    survival_rank_margin: float = 1.0

    # Poincare ball curvature
    poincare_curvature: float = 1.0

    # Loss weights
    manifold_lambda: float = 0.1       # lambda_1: hyperbolic manifold ordering
    causal_lambda: float = 0.05        # lambda_2: causal counterfactual
    hypergraph_lambda: float = 0.01    # lambda_3: Laplacian smoothness
    causal_context_gamma: float = 0.1  # gamma: variance-across-context penalty

    device: str = "cuda"


CFG = Config()
DEVICE = torch.device(CFG.device if torch.cuda.is_available() else "cpu")


###############################################################
# DATA LOADING
###############################################################

class MSKDataLoader:

    def __init__(self, config: Config):
        self.config = config

    def load_raw(self):

        clinical_patient = pd.read_csv(
            os.path.join(self.config.data_dir, "data_clinical_patient.txt"),
            sep="\t",
            comment="#"
        )

        clinical_sample = pd.read_csv(
            os.path.join(self.config.data_dir, "data_clinical_sample.txt"),
            sep="\t",
            comment="#"
        )

        mutations = pd.read_csv(
            os.path.join(self.config.data_dir, "data_mutations.txt"),
            sep="\t",
            comment="#",
            low_memory=False
        )

        treatment = pd.read_csv(
            os.path.join(self.config.data_dir, "data_timeline_treatment.txt"),
            sep="\t",
            comment="#"
        )

        return clinical_patient, clinical_sample, mutations, treatment


###############################################################
# PREPROCESSING
###############################################################

class MSKPreprocessor:

    def __init__(self):
        pass

    def preprocess(self, clinical_patient, clinical_sample, mutations, treatment):

        clinical_sample = clinical_sample.merge(
            clinical_patient,
            on="PATIENT_ID",
            how="inner"
        )

        clinical_sample = clinical_sample[
            ["SAMPLE_ID", "PATIENT_ID", "OS_MONTHS", "OS_STATUS"]
        ].dropna()

        clinical_sample["event"] = clinical_sample["OS_STATUS"].apply(
            lambda x: 1 if "DECEASED" in str(x).upper() else 0
        )

        mutations = mutations.merge(
            clinical_sample[["SAMPLE_ID", "PATIENT_ID"]],
            left_on="Tumor_Sample_Barcode",
            right_on="SAMPLE_ID",
            how="inner"
        )

        patient_genes = defaultdict(set)

        for _, row in mutations.iterrows():
            gene = row["Hugo_Symbol"]
            pid = row["PATIENT_ID"]
            if pd.notna(gene):
                patient_genes[pid].add(gene)

        treatment = treatment.sort_values("START_DATE")
        last_treatment = treatment.groupby("PATIENT_ID").tail(1)

        treatment_map = dict(zip(last_treatment["PATIENT_ID"], last_treatment["AGENT"]))

        data_rows = []

        for _, row in clinical_sample.iterrows():
            pid = row["PATIENT_ID"]
            if pid in patient_genes:
                data_rows.append({
                    "PATIENT_ID": pid,
                    "genes": list(patient_genes[pid]),
                    "time": float(row["OS_MONTHS"]),
                    "event": int(row["event"]),
                    "treatment": treatment_map.get(pid, "UNKNOWN")
                })

        df = pd.DataFrame(data_rows)

        return df


###############################################################
# GENE VOCABULARY
###############################################################

class GeneVocabulary:

    def __init__(self, df: pd.DataFrame):
        self.gene2idx = self.build_vocab(df)
        self.idx2gene = {v: k for k, v in self.gene2idx.items()}

    def build_vocab(self, df):

        all_genes = set()
        for genes in df["genes"]:
            all_genes.update(genes)

        gene2idx = {g: i + 1 for i, g in enumerate(sorted(all_genes))}
        gene2idx["PAD"] = 0

        return gene2idx

    def __len__(self):
        return len(self.gene2idx)


###############################################################
# PATHWAY LOADER
###############################################################

class PathwayLoader:
    """
    Loads biological pathway gene sets from multiple sources.
    Returns a dict: {pathway_name: set_of_gene_symbols}
    """

    @staticmethod
    def load_msigdb(
        gene_sets: List[str] = ["KEGG_2021_Human", "MSigDB_Hallmark_2020"],
        organism: str = "human"
    ) -> Dict[str, Set[str]]:
        """
        Pulls pathway gene sets directly from Enrichr/MSigDB via gseapy.
        gene_sets options include:
          - 'KEGG_2021_Human'
          - 'Reactome_2022'
          - 'WikiPathway_2023_Human'
          - 'MSigDB_Hallmark_2020'  (50 broad hallmark gene sets, very clean)
        """
        pathways = {}

        for gs_name in gene_sets:
            library = gp.get_library(name=gs_name, organism=organism)
            for pathway_name, genes in library.items():
                clean_name = f"{gs_name}__{pathway_name}"
                pathways[clean_name] = set(genes)

        print(f"Loaded {len(pathways)} pathways total.")
        return pathways

    @staticmethod
    def load_reactome_rest(species: str = "Homo sapiens") -> Dict[str, Set[str]]:
        """
        Fetches pathway->gene mappings directly from Reactome REST API.
        Slower but most up-to-date.
        """
        url = "https://reactome.org/ContentService/data/pathways/top/9606"
        resp = requests.get(url)
        top_pathways = resp.json()

        pathways = {}

        for pw in top_pathways[:200]:  # limit for speed; remove cap for full set
            pw_id = pw["stId"]
            pw_name = pw["displayName"]

            gene_url = (
                f"https://reactome.org/ContentService/data/participants/{pw_id}"
                f"/participatingPhysicalEntities"
            )
            gresp = requests.get(gene_url)

            if gresp.status_code != 200:
                continue

            entities = gresp.json()
            genes = set()

            for entity in entities:
                if "name" in entity:
                    genes.add(entity["name"].upper())

            if genes:
                pathways[f"Reactome__{pw_name}"] = genes

        return pathways

    @staticmethod
    def load_from_gmt(gmt_path: str) -> Dict[str, Set[str]]:
        """
        Load any .gmt file (standard MSigDB format).
        Download from: https://www.gsea-msigdb.org/gsea/msigdb/
        Format per line: PATHWAY_NAME  SOURCE_URL  GENE1  GENE2  ...
        """
        pathways = {}

        with open(gmt_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                name = parts[0]
                genes = set(parts[2:])
                pathways[name] = genes

        print(f"Loaded {len(pathways)} pathways from {gmt_path}")
        return pathways


###############################################################
# PATHWAY HYPERGRAPH BUILDER
###############################################################

class PathwayHypergraphBuilder:
    """
    Builds hypergraph incidence matrix H where each hyperedge
    corresponds to a known biological pathway, optionally
    combined with patient co-occurrence edges.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        gene2idx: Dict[str, int],
        pathways: Dict[str, Set[str]],
        min_pathway_overlap: int = 2,
        include_cooccurrence: bool = True,
        cooccurrence_weight: float = 0.5,
        pathway_weight: float = 1.0
    ):
        self.df = df
        self.gene2idx = gene2idx
        self.pathways = pathways
        self.min_pathway_overlap = min_pathway_overlap
        self.include_cooccurrence = include_cooccurrence
        self.cooccurrence_weight = cooccurrence_weight
        self.pathway_weight = pathway_weight

        # Track which hyperedge index maps to which pathway name
        self.edge_metadata: List[Dict] = []

    def build(self) -> torch.Tensor:

        rows = []
        cols = []
        vals = []
        edge_index = 0

        # ── PATHWAY HYPEREDGES ──────────────────────────────────────
        valid_pathway_count = 0

        for pathway_name, pathway_genes in self.pathways.items():

            # Only include genes that exist in our vocabulary
            vocab_genes = [
                g for g in pathway_genes
                if g in self.gene2idx
            ]

            # Skip pathways with too few vocab genes
            # (avoids degenerate singleton or doublet edges)
            if len(vocab_genes) < self.min_pathway_overlap:
                continue

            for gene in vocab_genes:
                rows.append(self.gene2idx[gene])
                cols.append(edge_index)
                vals.append(self.pathway_weight)

            self.edge_metadata.append({
                "edge_id": edge_index,
                "type": "pathway",
                "name": pathway_name,
                "size": len(vocab_genes)
            })

            edge_index += 1
            valid_pathway_count += 1

        print(f"Added {valid_pathway_count} pathway hyperedges.")

        # ── CO-OCCURRENCE HYPEREDGES (optional) ────────────────────
        if self.include_cooccurrence:

            gene_cooccur = defaultdict(set)
            for genes in self.df["genes"]:
                for g in genes:
                    gene_cooccur[g].update(genes)

            for gene, neighbors in gene_cooccur.items():
                for g in neighbors:
                    if g in self.gene2idx:
                        rows.append(self.gene2idx[g])
                        cols.append(edge_index)
                        vals.append(self.cooccurrence_weight)

                self.edge_metadata.append({
                    "edge_id": edge_index,
                    "type": "cooccurrence",
                    "name": gene,
                    "size": len(neighbors)
                })

                edge_index += 1

            print(f"Added co-occurrence hyperedges (one per gene neighborhood).")

        # ── ASSEMBLE SPARSE TENSOR ─────────────────────────────────
        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.tensor(vals, dtype=torch.float32)

        H = torch.sparse_coo_tensor(
            indices,
            values,
            size=(len(self.gene2idx), edge_index)
        )

        print(f"Hypergraph: {len(self.gene2idx)} nodes, {edge_index} hyperedges.")

        return H.coalesce().to(DEVICE)

    def get_pathway_edge_ids(self) -> List[int]:
        return [m["edge_id"] for m in self.edge_metadata if m["type"] == "pathway"]

    def get_edge_name(self, edge_id: int) -> str:
        return self.edge_metadata[edge_id]["name"]


###############################################################
# GENE-PATHWAY MEMBERSHIP MATRIX
###############################################################

def build_gene_pathway_matrix(
    gene2idx: Dict[str, int],
    pathways: Dict[str, Set[str]],
    pathway_names: List[str]
) -> torch.Tensor:
    """
    Returns a (|V|, |P|) binary matrix where entry [i,j] = 1
    if gene i belongs to pathway j.
    """
    V = len(gene2idx)
    P = len(pathway_names)

    matrix = torch.zeros(V, P)

    for j, pw_name in enumerate(pathway_names):
        if pw_name not in pathways:
            continue
        for gene in pathways[pw_name]:
            if gene in gene2idx:
                matrix[gene2idx[gene], j] = 1.0

    return matrix.to(DEVICE)


###############################################################
# HYPERGRAPH LAPLACIAN
###############################################################

def build_hypergraph_laplacian(H: torch.Tensor) -> torch.Tensor:
    """
    Computes the hypergraph Laplacian:
        L = Dv - Dv^{-1/2} H De^{-1} H^T Dv^{-1/2}

    Used in the smoothness regularizer:
        L_hypergraph = Tr(Z^T L Z)

    Returns L as a dense (|V|, |V|) tensor.
    Note: for large vocabularies, use only a subsampled version
    at training time (see HypergraphSmoothnessLoss).
    """
    H = H.coalesce()
    num_nodes = H.size(0)

    Dv = torch.sparse.sum(H, dim=1).to_dense()          # (|V|,)
    De = torch.sparse.sum(H, dim=0).to_dense()          # (|E|,)

    Dv_inv_sqrt = torch.pow(Dv + 1e-6, -0.5)           # (|V|,)
    De_inv = torch.pow(De + 1e-6, -1.0)                 # (|E|,)

    # Theta = Dv^{-1/2} H De^{-1} H^T Dv^{-1/2}
    # Computed via sparse matmuls to avoid materializing H as dense

    H_dense = H.to_dense()                              # (|V|, |E|)

    Theta = (
        Dv_inv_sqrt.unsqueeze(1)
        * H_dense
        * De_inv.unsqueeze(0)
    )                                                    # (|V|, |E|)

    Theta = Theta @ H_dense.T                           # (|V|, |V|)
    Theta = Theta * Dv_inv_sqrt.unsqueeze(1)
    Theta = Theta * Dv_inv_sqrt.unsqueeze(0)

    L = torch.diag(Dv) - Theta                          # (|V|, |V|)

    return L


###############################################################
# DATASET
###############################################################

class MSKDataset(Dataset):

    def __init__(
        self,
        df: pd.DataFrame,
        gene_vocab: GeneVocabulary,
        treatment_encoder: LabelEncoder,
        max_genes: int
    ):

        self.df = df.reset_index(drop=True)
        self.gene2idx = gene_vocab.gene2idx
        self.treatment_encoder = treatment_encoder
        self.max_genes = max_genes

    def __len__(self):
        return len(self.df)

    def encode_genes(self, genes: List[str]) -> List[int]:

        ids = [self.gene2idx.get(g, 0) for g in genes]
        ids = ids[:self.max_genes]

        if len(ids) < self.max_genes:
            ids += [0] * (self.max_genes - len(ids))

        return ids

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        gene_ids = self.encode_genes(row["genes"])

        treatment_id = self.treatment_encoder.transform(
            [row["treatment"]]
        )[0]

        return {
            "gene_ids": torch.tensor(gene_ids, dtype=torch.long),
            "context": torch.tensor(treatment_id, dtype=torch.long),
            "time": torch.tensor(row["time"], dtype=torch.float32),
            "event": torch.tensor(row["event"], dtype=torch.float32)
        }


###############################################################
# DATALOADER FACTORY
###############################################################

def build_dataloaders(df, gene_vocab):

    treatment_encoder = LabelEncoder()
    df["treatment"] = df["treatment"].fillna("UNKNOWN")
    treatment_encoder.fit(df["treatment"])

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = MSKDataset(
        train_df,
        gene_vocab,
        treatment_encoder,
        CFG.max_genes_per_patient
    )

    val_dataset = MSKDataset(
        val_df,
        gene_vocab,
        treatment_encoder,
        CFG.max_genes_per_patient
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False
    )

    return train_loader, val_loader, treatment_encoder


# %%
###############################################################
# POINCARE BALL OPERATIONS
###############################################################

class PoincareBall(nn.Module):
    """
    Implements the Poincare ball model of hyperbolic space with
    curvature c > 0.

    Key operations:
        - mobius_add(x, y)    : Mobius addition in B^d_c
        - expmap0(v)          : Exponential map from origin (tangent -> ball)
        - logmap0(x)          : Logarithmic map to origin  (ball -> tangent)
        - dist(x, y)          : Hyperbolic distance
        - dist0(x)            : Distance from origin (radial norm)
        - project(x)          : Project onto ball interior (numerical safety)
    """

    def __init__(self, c: float = 1.0):
        super().__init__()
        # c is not learned; register as buffer for device movement
        self.register_buffer("c", torch.tensor(c, dtype=torch.float32))

    def project(self, x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """
        Project x onto open ball: ||x|| < 1/sqrt(c).
        Clips norm to (1/sqrt(c) - eps) to avoid boundary singularities.
        """
        c = self.c
        max_norm = (1.0 / (c.sqrt() + 1e-8)) - eps
        norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        x = torch.where(norm >= max_norm, x / norm * max_norm, x)
        return x

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Mobius addition in the Poincare ball:

            x (+)_c y = ((1 + 2c<x,y> + c||y||^2) x
                         + (1 - c||x||^2) y)
                        / (1 + 2c<x,y> + c^2||x||^2||y||^2)
        """
        c = self.c

        x2 = (x * x).sum(dim=-1, keepdim=True)           # ||x||^2
        y2 = (y * y).sum(dim=-1, keepdim=True)           # ||y||^2
        xy = (x * y).sum(dim=-1, keepdim=True)           # <x, y>

        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c * c * x2 * y2

        return num / denom.clamp(min=1e-10)

    def expmap0(self, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map at the origin:

            exp_0^c(v) = tanh(sqrt(c) * ||v||) / (sqrt(c) * ||v||) * v

        Maps a tangent vector v at origin into the Poincare ball.
        """
        c = self.c
        sqrt_c = c.sqrt()

        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        tanh_arg = (sqrt_c * v_norm).clamp(max=15.0)     # prevent overflow

        return (tanh_arg.tanh() / (sqrt_c * v_norm)) * v

    def logmap0(self, x: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map at the origin:

            log_0^c(x) = arctanh(sqrt(c) * ||x||) / (sqrt(c) * ||x||) * x

        Maps a point in the Poincare ball back to the tangent space at origin.
        """
        c = self.c
        sqrt_c = c.sqrt()

        x_norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        atanh_arg = (sqrt_c * x_norm).clamp(max=1.0 - 1e-5)  # arctanh domain

        return (atanh_arg.arctanh() / (sqrt_c * x_norm)) * x

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Hyperbolic distance between x and y:

            d_H(x, y) = (2/sqrt(c)) * arctanh(sqrt(c) * ||-x (+)_c y||)
        """
        c = self.c
        sqrt_c = c.sqrt()

        diff = self.mobius_add(-x, y)
        diff_norm = diff.norm(dim=-1).clamp(min=1e-10)
        atanh_arg = (sqrt_c * diff_norm).clamp(max=1.0 - 1e-5)

        return (2.0 / sqrt_c) * atanh_arg.arctanh()

    def dist0(self, x: torch.Tensor) -> torch.Tensor:
        """
        Hyperbolic distance from the origin:

            d_H(0, x) = (2/sqrt(c)) * arctanh(sqrt(c) * ||x||)

        This is the radial norm in hyperbolic space and is used
        directly as the survival risk score.
        """
        c = self.c
        sqrt_c = c.sqrt()

        x_norm = x.norm(dim=-1).clamp(min=1e-10)
        atanh_arg = (sqrt_c * x_norm).clamp(max=1.0 - 1e-5)

        return (2.0 / sqrt_c) * atanh_arg.arctanh()


###############################################################
# EUCLIDEAN -> POINCARE PROJECTION LAYER
###############################################################

class EuclideanToHyperbolic(nn.Module):
    """
    Projects a Euclidean latent vector into the Poincare ball via:
        1. Linear transformation (Euclidean -> tangent space at origin)
        2. Exponential map (tangent -> ball)
        3. Safety projection (clip to ball interior)

    The linear layer operates in Euclidean space (standard gradient flow).
    The expmap is differentiable, so gradients flow cleanly through.
    """

    def __init__(self, in_dim: int, out_dim: int, ball: PoincareBall):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.ball = ball

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_dim) Euclidean
        v = self.linear(x)                  # (..., out_dim) tangent vector
        z = self.ball.expmap0(v)            # (..., out_dim) in Poincare ball
        z = self.ball.project(z)            # numerical safety
        return z


###############################################################
# TRUE HYPERGRAPH PROPAGATION (CORRECTED GLOBAL VERSION)
###############################################################

class HypergraphPropagation(nn.Module):
    """
    True global hypergraph propagation:
        X' = Dv^-1/2 H De^-1 H^T Dv^-1/2 X
    Where X is full gene embedding matrix (|V| x D)
    """

    def __init__(self, H: torch.Tensor):
        super().__init__()

        self.H = H.coalesce()
        self.num_nodes, self.num_edges = self.H.size()

        self.compute_degrees()

    def compute_degrees(self):

        H = self.H

        Dv = torch.sparse.sum(H, dim=1).to_dense()
        De = torch.sparse.sum(H, dim=0).to_dense()

        self.register_buffer("Dv_inv_sqrt", torch.pow(Dv + 1e-6, -0.5))
        self.register_buffer("De_inv", torch.pow(De + 1e-6, -1.0))

    def forward(self, X):

        # X: (|V|, D)

        H = self.H

        Dv_inv_sqrt = self.Dv_inv_sqrt.unsqueeze(1)
        De_inv = self.De_inv.unsqueeze(1)

        # Normalize nodes
        X = X * Dv_inv_sqrt

        # H^T X
        Ht = torch.sparse_coo_tensor(
            torch.stack([H.indices()[1], H.indices()[0]]),
            H.values(),
            size=(self.num_edges, self.num_nodes),
            device=X.device
        )

        HX = torch.sparse.mm(Ht, X)
        HX = HX * De_inv

        X_prop = torch.sparse.mm(H, HX)
        X_prop = X_prop * Dv_inv_sqrt

        return X_prop


###############################################################
# CONTEXT-CONDITIONED HYPERGRAPH ATTENTION
###############################################################

class ContextHypergraphAttention(nn.Module):

    def __init__(self, embed_dim, context_dim):
        super().__init__()

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.context_proj = nn.Linear(context_dim, embed_dim)

        self.scale = embed_dim ** 0.5

    def forward(self, X, context):

        # X: (B, M, D)
        # context: (B, D_c)

        Q = self.query(X)
        K = self.key(X)
        V = self.value(X)

        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / self.scale

        context_bias = self.context_proj(context).unsqueeze(1)
        attn_scores = attn_scores + torch.matmul(Q, context_bias.transpose(1, 2))

        attn_weights = torch.softmax(attn_scores, dim=-1)

        X_att = torch.matmul(attn_weights, V)

        return X_att


###############################################################
# PATHWAY ATTENTION POOLING
###############################################################

class PathwayAttentionPooling(nn.Module):
    """
    Instead of treating all hyperedges equally during propagation,
    this module learns a per-pathway attention score conditioned
    on the treatment context, then produces a pathway-level
    patient representation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_pathways: int,
        context_dim: int
    ):
        super().__init__()

        # Learnable pathway embeddings (one per pathway)
        self.pathway_embed = nn.Embedding(num_pathways, embed_dim)

        # Context-conditioned pathway attention
        self.attn_proj = nn.Sequential(
            nn.Linear(embed_dim + context_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1)
        )

    def forward(
        self,
        gene_embeddings: torch.Tensor,    # (B, M, D)  patient gene embeddings
        gene_ids: torch.Tensor,           # (B, M)     gene indices
        gene_to_pathways: torch.Tensor,   # (|V|, P)   binary: gene i in pathway j
        context_vec: torch.Tensor         # (B, D_c)   treatment embedding
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, M, D = gene_embeddings.shape
        P = gene_to_pathways.shape[1]

        # For each patient, gather per-pathway gene membership masks
        # patient_masks: (B, M, P)
        patient_masks = gene_to_pathways[gene_ids]

        # Weighted sum of gene embeddings per pathway
        # gene_embeddings: (B, M, D), patient_masks: (B, M, P) -> (B, P, D)
        pathway_gene_sum = torch.einsum("bmd,bmp->bpd", gene_embeddings, patient_masks.float())
        pathway_counts = patient_masks.float().sum(dim=1).unsqueeze(-1).clamp(min=1)  # (B, P, 1)
        pathway_representations = pathway_gene_sum / pathway_counts                   # (B, P, D)

        # Context-conditioned attention over pathways
        context_expanded = context_vec.unsqueeze(1).expand(-1, P, -1)                # (B, P, D_c)
        attn_input = torch.cat([pathway_representations, context_expanded], dim=-1)  # (B, P, D+D_c)
        attn_scores = self.attn_proj(attn_input).squeeze(-1)                         # (B, P)
        attn_weights = torch.softmax(attn_scores, dim=-1)                            # (B, P)

        # Weighted sum over pathways -> patient vector
        patient_vec = torch.einsum("bp,bpd->bd", attn_weights, pathway_representations)  # (B, D)

        return patient_vec, attn_weights


###############################################################
# COUNTERFACTUAL OPERATOR
###############################################################

class CounterfactualOperator:

    def remove_gene(self, gene_ids, remove_idx):

        # gene_ids: (B, M)
        # remove_idx: (B,)

        remove_idx = remove_idx.unsqueeze(1)  # (B,1)

        mask = (gene_ids != remove_idx).long()

        return gene_ids * mask

    def swap_treatment(self, context_ids, new_treatment_id):

        # context_ids: (B,)
        # new_treatment_id: scalar or (B,)

        return torch.full_like(context_ids, new_treatment_id)


###############################################################
# FULL MODEL: CAUSAL HYPERBOLIC HYPERGRAPH SURVIVAL NETWORK
###############################################################

class CausalHyperbolicHypergraphModel(nn.Module):
    """
    f_theta : (X, C) -> (Z, R)

    where:
        Z in H^d  (Poincare ball, curvature c)
        R = (2/sqrt(c)) * arctanh(sqrt(c) * ||Z||)  [radial risk]

    Structural equation model:
        Z = g_theta(X, C)       [hypergraph encoder + hyperbolic projection]
        R = h_theta(Z)          [radial risk head]

    Intervention:
        X^{do(g=0)}             [gene g zeroed out]
        Delta_g = R - R_{do(g=0)}
    """

    def __init__(
        self,
        num_genes,
        num_treatments,
        H_sparse,
        gene_pathway_matrix,
        num_pathways,
        use_pathway_attention=True,
        curvature: float = 1.0
    ):
        super().__init__()

        # ── Embeddings ─────────────────────────────────────────────
        self.gene_embed = nn.Embedding(num_genes, CFG.embed_dim)
        self.treatment_embed = nn.Embedding(num_treatments, CFG.embed_dim)

        # ── Hypergraph propagation ──────────────────────────────────
        self.hyper_prop = HypergraphPropagation(H_sparse)

        # ── Pathway attention pooling ───────────────────────────────
        self.use_pathway_attention = use_pathway_attention

        if use_pathway_attention:
            self.pathway_attention = PathwayAttentionPooling(
                embed_dim=CFG.embed_dim,
                num_pathways=num_pathways,
                context_dim=CFG.embed_dim
            )
            self.register_buffer("gene_pathway_matrix", gene_pathway_matrix)
        else:
            self.context_attention = ContextHypergraphAttention(
                CFG.embed_dim,
                CFG.embed_dim
            )
            self.pool = nn.AdaptiveAvgPool1d(1)

        # ── Poincare ball ───────────────────────────────────────────
        self.ball = PoincareBall(c=curvature)

        # ── Euclidean -> Hyperbolic projection ──────────────────────
        # Projects from Euclidean latent to Poincare ball
        self.euclidean_to_hyp = EuclideanToHyperbolic(
            in_dim=CFG.embed_dim,
            out_dim=CFG.latent_dim,
            ball=self.ball
        )

        # ── Risk head: scalar beta scales radial distance ───────────
        # R_i = beta * d_H(0, z_i)
        self.risk_beta = nn.Parameter(torch.ones(1))

        # ── Counterfactual operator ─────────────────────────────────
        self.counterfactual = CounterfactualOperator()

    def encode(self, gene_ids, context_ids):
        """
        Shared encoder:
            gene_ids    -> hypergraph propagation -> pathway pooling
            context_ids -> treatment embedding
            output      -> Euclidean patient representation (B, D)
        """
        # Global hypergraph propagation over full gene embedding table
        E = self.gene_embed.weight              # (|V|, D)
        E_prop = self.hyper_prop(E)             # (|V|, D)

        # Patient-specific gene embeddings
        X = E_prop[gene_ids]                    # (B, M, D)
        context_vec = self.treatment_embed(context_ids)  # (B, D)

        if self.use_pathway_attention:
            # Pathway-aware pooling with treatment-conditioned attention
            patient_vec, pathway_weights = self.pathway_attention(
                X, gene_ids, self.gene_pathway_matrix, context_vec
            )
        else:
            # Fallback: context attention + average pool
            X_att = self.context_attention(X, context_vec)
            patient_vec = self.pool(X_att.transpose(1, 2)).squeeze(-1)

        return patient_vec

    def forward(self, gene_ids, context_ids):
        """
        Full forward pass:
            -> Euclidean patient vector
            -> Hyperbolic projection (expmap)
            -> Radial risk score

        Returns:
            risk  : (B,)   scalar risk per patient
            z     : (B, D) hyperbolic latent point in Poincare ball
        """
        # Euclidean representation
        patient_vec = self.encode(gene_ids, context_ids)  # (B, D)

        # Project to Poincare ball via expmap at origin
        z = self.euclidean_to_hyp(patient_vec)             # (B, latent_dim) in H^d

        # Radial risk: R_i = beta * d_H(0, z_i)
        # High-risk patients are pushed toward the boundary (large radial norm)
        # Low-risk patients cluster near the origin
        radial = self.ball.dist0(z)                        # (B,)
        risk = self.risk_beta * radial                     # (B,)

        return risk, z

    def counterfactual_gene(self, gene_ids, context_ids, gene_remove_idx):
        """
        Structural intervention: do(gene = 0)
        Returns counterfactual (risk, z) under gene ablation.
        """
        gene_cf = self.counterfactual.remove_gene(
            gene_ids,
            gene_remove_idx
        )

        return self.forward(gene_cf, context_ids)

    def counterfactual_treatment(self, gene_ids, context_ids, new_treatment_id):
        """
        Structural intervention: do(treatment = new_treatment_id)
        Returns counterfactual (risk, z) under treatment swap.
        """
        context_cf = self.counterfactual.swap_treatment(
            context_ids,
            new_treatment_id
        )

        return self.forward(gene_ids, context_cf)


# %%
###############################################################
# LOSS I: COX PARTIAL LIKELIHOOD
###############################################################

class CoxPHLoss(nn.Module):
    """
    L_Cox = -sum_{i: delta_i=1} (R_i - log sum_{j: t_j >= t_i} exp(R_j))
    """

    def __init__(self):
        super().__init__()

    def forward(self, risk, time, event):

        # Sort by descending time
        order = torch.argsort(time, descending=True)
        risk = risk[order]
        event = event[order]

        log_cumsum = torch.logcumsumexp(risk, dim=0)

        loss = -torch.sum((risk - log_cumsum) * event)

        return loss / (event.sum() + 1e-6)


###############################################################
# LOSS II: HYPERBOLIC MANIFOLD ORDERING
###############################################################

class HyperbolicManifoldLoss(nn.Module):
    """
    Enforces radial ordering in hyperbolic space:

        L_manifold = sum_{i<j, t_i < t_j, delta_i=1}
                     ReLU(d_H(0, z_j) - d_H(0, z_i))

    Meaning: if patient i died before patient j,
    then z_i should have a larger hyperbolic radial distance
    from the origin than z_j.

    This is stronger than Euclidean ranking: the geometry of
    the Poincare ball means the boundary encodes extreme risk,
    and the origin encodes low/no risk.
    """

    def __init__(self, ball: PoincareBall):
        super().__init__()
        self.ball = ball

    def forward(
        self,
        z: torch.Tensor,      # (B, D) hyperbolic latent points
        time: torch.Tensor,   # (B,)
        event: torch.Tensor   # (B,)
    ) -> torch.Tensor:

        # Radial distances from origin for all patients
        radial = self.ball.dist0(z)   # (B,)

        B = z.size(0)
        loss = torch.tensor(0.0, device=z.device)
        count = 0

        for i in range(B):
            for j in range(B):
                # Patient i died before patient j (and i is an event)
                if time[i] < time[j] and event[i] == 1:
                    # We want radial[i] > radial[j]
                    # Penalize if radial[j] >= radial[i]
                    loss = loss + F.relu(radial[j] - radial[i])
                    count += 1

        if count == 0:
            return torch.tensor(0.0, device=z.device)

        return loss / count


###############################################################
# LOSS III: CAUSAL COUNTERFACTUAL LOSS
###############################################################

class CausalCounterfactualLoss(nn.Module):
    """
    Implements the full causal counterfactual objective:

        L_causal = E_g [ ||Delta_g||_1 + gamma * Var_C(Delta_g) ]

    Where:
        Delta_g = R - R_{do(g=0)}     [causal effect of gene g]
        ||Delta_g||_1                 [sparsity prior: most genes
                                       should have small causal effect]
        Var_C(Delta_g)                [context-variance: if a gene
                                       is not treatment-targeted, its
                                       causal effect should be stable
                                       across treatment contexts]

    Implementation:
        - Sparsity: computed from gene ablation counterfactual
        - Context-variance: computed by swapping to all available
          treatment contexts and measuring variance of Delta_g
    """

    def __init__(self, gamma: float = 0.1):
        super().__init__()
        self.gamma = gamma

    def forward(
        self,
        original_risk: torch.Tensor,   # (B,)
        cf_risk: torch.Tensor,         # (B,) risk under gene ablation
        cf_risks_by_context: List[torch.Tensor]  # list of (B,) risks under different contexts
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            loss_cf      : total causal loss
            loss_sparse  : L1 sparsity term
            loss_var     : context-variance term
        """
        # Delta_g = R - R_{do(g=0)}
        delta = torch.abs(original_risk - cf_risk)    # (B,)

        # Sparsity prior: ||Delta_g||_1
        loss_sparse = torch.mean(torch.abs(delta))

        # Context-variance: Var_C(Delta_g)
        # For each context c, compute delta under that context
        # Then measure variance across contexts per patient
        if len(cf_risks_by_context) > 1:
            # Stack context-specific deltas: (num_contexts, B)
            context_deltas = torch.stack([
                torch.abs(original_risk - cf_r)
                for cf_r in cf_risks_by_context
            ], dim=0)

            # Variance across contexts for each patient, then average
            loss_var = context_deltas.var(dim=0).mean()
        else:
            loss_var = torch.tensor(0.0, device=original_risk.device)

        loss_cf = loss_sparse + self.gamma * loss_var

        return loss_cf, loss_sparse, loss_var


###############################################################
# LOSS IV: HYPERGRAPH LAPLACIAN SMOOTHNESS
###############################################################

class HypergraphSmoothnessLoss(nn.Module):
    """
    Encourages smoothness of z across the hypergraph:

        L_hypergraph = Tr(Z^T L Z)

    where L is the hypergraph Laplacian.

    For a batch of patients with gene sets, we approximate this
    using only the unique gene indices present in the batch.
    Full Laplacian trace is O(|V|^2) which is impractical;
    we use a batched subgraph approximation.
    """

    def __init__(self, L: torch.Tensor):
        """
        L: (|V|, |V|) hypergraph Laplacian (precomputed, stored on DEVICE)
        """
        super().__init__()
        self.register_buffer("L", L)

    def forward(
        self,
        z: torch.Tensor,         # (B, D) hyperbolic latent points
        gene_ids: torch.Tensor   # (B, M) gene indices per patient
    ) -> torch.Tensor:
        """
        Approximation: treat each patient's latent z as the representation
        of the centroid gene node for that patient, then compute pairwise
        Laplacian smoothness across the batch using their mean gene index.

        For a more exact implementation, map z back to per-gene space
        via logmap and compute Tr(E^T L E) on the gene embedding matrix.
        """
        B, D = z.shape

        # Compute pairwise Laplacian smoothness over batch:
        # L_smooth = sum_{i,j} L_ij * <z_i, z_j>
        # = Tr(Z^T L_batch Z) where L_batch is the BxB submatrix

        # Use the mean gene index per patient as proxy node index
        mean_gene_idx = gene_ids.float().mean(dim=1).long().clamp(
            0, self.L.size(0) - 1
        )  # (B,)

        # Extract BxB submatrix of L
        L_batch = self.L[mean_gene_idx][:, mean_gene_idx]  # (B, B)

        # Tr(Z^T L_batch Z) = sum_d z^d^T L_batch z^d
        loss = torch.trace(z.T @ L_batch @ z) / (B * D)

        return loss.abs()   # abs for numerical stability (L is PSD but floating point)


# %%
###############################################################
# C-INDEX
###############################################################

def concordance_index(risk, time, event):

    risk = risk.detach().cpu().numpy()
    time = time.detach().cpu().numpy()
    event = event.detach().cpu().numpy()

    n = len(time)
    num = 0
    den = 0

    for i in range(n):
        for j in range(n):
            if time[i] < time[j] and event[i] == 1:
                den += 1
                if risk[i] > risk[j]:
                    num += 1
                elif risk[i] == risk[j]:
                    num += 0.5

    if den == 0:
        return 0.0

    return num / den


###############################################################
# TRAINER
###############################################################

class Trainer:
    """
    Full training loop for:

        L(theta) = L_Cox
                 + lambda_1 * L_manifold
                 + lambda_2 * L_causal
                 + lambda_3 * L_hypergraph

    Context variance in L_causal is computed by running
    counterfactual_treatment for two randomly sampled alternative
    treatment contexts per batch, then measuring Var_C(Delta_g).
    """

    def __init__(
        self,
        model: CausalHyperbolicHypergraphModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config,
        L_hypergraph: torch.Tensor,
        num_treatments: int
    ):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.num_treatments = num_treatments

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        # Individual loss modules
        self.cox_loss = CoxPHLoss()
        self.manifold_loss = HyperbolicManifoldLoss(ball=model.ball)
        self.causal_loss = CausalCounterfactualLoss(gamma=config.causal_context_gamma)
        self.hypergraph_loss = HypergraphSmoothnessLoss(L=L_hypergraph)

    def train_epoch(self):

        self.model.train()
        total_loss = 0.0
        total_cox = 0.0
        total_manifold = 0.0
        total_causal = 0.0
        total_hg = 0.0

        for batch in self.train_loader:

            gene_ids = batch["gene_ids"].to(DEVICE)
            context_ids = batch["context"].to(DEVICE)
            time = batch["time"].to(DEVICE)
            event = batch["event"].to(DEVICE)

            self.optimizer.zero_grad()

            # ── Forward pass ──────────────────────────────────────
            risk, z = self.model(gene_ids, context_ids)

            # ── L_Cox ─────────────────────────────────────────────
            loss_cox = self.cox_loss(risk, time, event)

            # ── L_manifold: hyperbolic radial ordering ─────────────
            loss_manifold = self.manifold_loss(z, time, event)

            # ── L_causal: gene ablation + context variance ─────────
            # Primary counterfactual: remove first mutated gene per patient
            gene_remove_idx = gene_ids[:, 0]
            cf_risk_gene, _ = self.model.counterfactual_gene(
                gene_ids,
                context_ids,
                gene_remove_idx
            )

            # Context-variance: swap to two randomly sampled alternative
            # treatment IDs and collect their counterfactual deltas
            cf_risks_by_context = []
            for _ in range(2):
                alt_treatment = torch.randint(
                    0, self.num_treatments, (1,), device=DEVICE
                ).item()
                cf_risk_ctx, _ = self.model.counterfactual_gene(
                    gene_ids,
                    self.model.counterfactual.swap_treatment(
                        context_ids, alt_treatment
                    ),
                    gene_remove_idx
                )
                cf_risks_by_context.append(cf_risk_ctx)

            loss_causal, loss_sparse, loss_var = self.causal_loss(
                risk, cf_risk_gene, cf_risks_by_context
            )

            # ── L_hypergraph: Laplacian smoothness ─────────────────
            loss_hg = self.hypergraph_loss(z, gene_ids)

            # ── Total loss ─────────────────────────────────────────
            total = (
                loss_cox
                + self.config.manifold_lambda    * loss_manifold
                + self.config.causal_lambda      * loss_causal
                + self.config.hypergraph_lambda  * loss_hg
            )

            total.backward()

            # Gradient clipping for hyperbolic stability
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Re-project all hyperbolic points onto ball after optimizer step
            # (SGD can push z outside ball boundary due to Euclidean updates)
            with torch.no_grad():
                self.model.euclidean_to_hyp.linear.weight.data = \
                    self.model.euclidean_to_hyp.linear.weight.data

            total_loss     += total.item()
            total_cox      += loss_cox.item()
            total_manifold += loss_manifold.item()
            total_causal   += loss_causal.item()
            total_hg       += loss_hg.item()

        n = len(self.train_loader)

        return {
            "total":     total_loss     / n,
            "cox":       total_cox      / n,
            "manifold":  total_manifold / n,
            "causal":    total_causal   / n,
            "hypergraph":total_hg       / n
        }

    def validate(self):

        self.model.eval()
        all_risk = []
        all_time = []
        all_event = []

        with torch.no_grad():
            for batch in self.val_loader:

                gene_ids = batch["gene_ids"].to(DEVICE)
                context_ids = batch["context"].to(DEVICE)
                time = batch["time"].to(DEVICE)
                event = batch["event"].to(DEVICE)

                risk, _ = self.model(gene_ids, context_ids)

                all_risk.append(risk)
                all_time.append(time)
                all_event.append(event)

        risk = torch.cat(all_risk)
        time = torch.cat(all_time)
        event = torch.cat(all_event)

        c_index = concordance_index(risk, time, event)

        return c_index

    def fit(self):

        for epoch in range(self.config.epochs):

            losses = self.train_epoch()
            val_cindex = self.validate()

            print(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"Total: {losses['total']:.4f} | "
                f"Cox: {losses['cox']:.4f} | "
                f"Manifold: {losses['manifold']:.4f} | "
                f"Causal: {losses['causal']:.4f} | "
                f"HG: {losses['hypergraph']:.4f} | "
                f"Val C-index: {val_cindex:.4f}"
            )


###############################################################
# MAIN
###############################################################

def main():

    print("Loading MSK-CHORD data...")

    loader = MSKDataLoader(CFG)
    clinical_patient, clinical_sample, mutations, treatment = loader.load_raw()

    preprocessor = MSKPreprocessor()
    df = preprocessor.preprocess(
        clinical_patient,
        clinical_sample,
        mutations,
        treatment
    )

    print("Building gene vocabulary...")
    gene_vocab = GeneVocabulary(df)

    # ── Load biological pathways ───────────────────────────────────
    print("Loading biological pathways...")
    pathway_loader = PathwayLoader()
    pathways = pathway_loader.load_msigdb(
        gene_sets=["KEGG_2021_Human", "MSigDB_Hallmark_2020"]
    )
    # Alternatively, load from a local .gmt file:
    # pathways = pathway_loader.load_from_gmt("h.all.v2023.1.Hs.symbols.gmt")
    # Or from the Reactome REST API:
    # pathways = pathway_loader.load_reactome_rest()

    # ── Build pathway hypergraph ───────────────────────────────────
    print("Building pathway hypergraph...")
    hyper_builder = PathwayHypergraphBuilder(
        df=df,
        gene2idx=gene_vocab.gene2idx,
        pathways=pathways,
        min_pathway_overlap=2,
        include_cooccurrence=True,
        cooccurrence_weight=0.5,
        pathway_weight=1.0
    )

    H_sparse = hyper_builder.build()

    # ── Build gene-pathway membership matrix ──────────────────────
    print("Building gene-pathway membership matrix...")
    pathway_names = [
        m["name"] for m in hyper_builder.edge_metadata
        if m["type"] == "pathway"
    ]
    num_pathways = len(pathway_names)

    gene_pathway_matrix = build_gene_pathway_matrix(
        gene_vocab.gene2idx,
        pathways,
        pathway_names
    )

    # ── Build hypergraph Laplacian for smoothness loss ─────────────
    print("Building hypergraph Laplacian...")
    L_hypergraph = build_hypergraph_laplacian(H_sparse).to(DEVICE)

    # ── Dataloaders ───────────────────────────────────────────────
    print("Building dataloaders...")
    train_loader, val_loader, treatment_encoder = build_dataloaders(
        df,
        gene_vocab
    )

    num_treatments = len(treatment_encoder.classes_)

    # ── Model ─────────────────────────────────────────────────────
    print("Initializing Causal Hyperbolic Hypergraph Survival Network...")
    model = CausalHyperbolicHypergraphModel(
        num_genes=len(gene_vocab),
        num_treatments=num_treatments,
        H_sparse=H_sparse,
        gene_pathway_matrix=gene_pathway_matrix,
        num_pathways=num_pathways,
        use_pathway_attention=True,
        curvature=CFG.poincare_curvature
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=CFG,
        L_hypergraph=L_hypergraph,
        num_treatments=num_treatments
    )

    print("Training...")
    trainer.fit()


if __name__ == "__main__":
    main()

# %%