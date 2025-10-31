import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# -------------------------
# Small GCN implementation
# -------------------------
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        x: [N, in_dim]
        adj: [N, N] adjacency (should include self loops)
        returns: [N, out_dim]
        """
        # Simple normalized adjacency multiplication: A_hat = D^{-1/2} A D^{-1/2}
        deg = adj.sum(dim=1)  # [N]
        deg_inv_sqrt = deg.clamp(min=1e-9).pow(-0.5)
        D_inv_sqrt = deg_inv_sqrt.unsqueeze(1)  # [N,1]
        A_norm = D_inv_sqrt * adj * D_inv_sqrt.t()  # broadcasting to [N,N]
        out = A_norm @ x  # [N, in_dim]
        return self.lin(out)  # [N, out_dim]

# -------------------------
# VGAE model
# -------------------------
class VGAE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        decoder_type: str = "dot",  # "dot" or "bilinear"
        device: str = "cpu"
    ):
        super().__init__()
        self.device = device
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        # Two heads for mu and logvar
        self.gcn_mu = GCNLayer(hidden_dim, latent_dim)
        self.gcn_logvar = GCNLayer(hidden_dim, latent_dim)

        self.decoder_type = decoder_type
        if decoder_type == "bilinear":
            # learnable bilinear matrix for directed/better scoring
            self.bilinear = nn.Bilinear(latent_dim, latent_dim, 1, bias=False)
            
        # Instead of simple bilinear, use expressive relation head
        self.relation_head = nn.Sequential(
            nn.Linear(4 * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1)
        )

    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        returns mu, logvar  (each [N, latent_dim])
        """
        h = F.relu(self.gcn1(x, adj))
        mu = self.gcn_mu(h, adj)
        logvar = self.gcn_logvar(h, adj)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    # def decode_scores(self, z: torch.Tensor) -> torch.Tensor:
    #     """
    #     For efficiency, returns a score matrix of shape [N, N].
    #     For 'dot': score_ij = z_i · z_j
    #     For 'bilinear': score_ij = bilinear(z_i, z_j) => returned as matrix [N,N]
    #     """
    #     N = z.size(0)
    #     if self.decoder_type == "dot":
    #         # inner product
    #         scores = z @ z.t()  # [N,N]
    #     else:
    #         # compute pairwise bilinear (slower for large N)
    #         # Efficient vectorized bilinear using outer expansion
    #         # bilinear returns shape [num_pairs, 1]
    #         zi = z.unsqueeze(1).expand(-1, N, -1)  # [N, N, D]
    #         zj = z.unsqueeze(0).expand(N, -1, -1)  # [N, N, D]
    #         pair = self.bilinear(zi.reshape(-1, z.size(1)), zj.reshape(-1, z.size(1)))
    #         scores = pair.view(N, N)  # [N,N]
    #     return scores
    
    def decode_scores(self, z):
        """
        Computes pairwise link scores using [zi, zj, zj-zi, zi*zj].
        Returns a dense [N, N] matrix of logits.
        """
        N, D = z.shape

        zi = z.unsqueeze(1).expand(-1, N, -1)  # [N, N, D]
        zj = z.unsqueeze(0).expand(N, -1, -1)  # [N, N, D]

        pair_emb = torch.cat([zi, zj, zj - zi, zi * zj], dim=-1)  # [N, N, 4D]
        scores = self.relation_head(pair_emb).squeeze(-1)  # [N, N]
        return scores

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [N, in_dim] (node features)
        adj: [N, N] adjacency (0/1 floats) - should include self loops if desired
        returns:
            recon_scores: [N, N] raw scores (not sigmoid-ed)
            mu: [N, latent_dim]
            logvar: [N, latent_dim]
        """
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        scores = self.decode_scores(z)  # raw logits for links
        return scores, mu, logvar

# -------------------------
# Loss helpers
# -------------------------
# def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
#     # KL per node per latent dimension summed over nodes and dims
#     # KL(N(mu, sigma) || N(0, I)) = 0.5 * sum( mu^2 + sigma^2 - 1 - log(sigma^2) )
#     kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return kld

# def reconstruction_loss_from_scores(scores: torch.Tensor, adj: torch.Tensor, pos_weight: Optional[float] = None) -> torch.Tensor:
#     """
#     BCE with logits over all node pairs. For symmetric graphs you may want to only use upper triangle.
#     pos_weight: scalar weight for positive class if graph is sparse (helps class imbalance).
#     """
#     # Flatten
#     logits = scores.view(-1)
#     labels = adj.view(-1)
#     if pos_weight is not None:
#         # use weighted BCE
#         loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=torch.tensor(pos_weight, device=logits.device))
#     else:
#         loss = F.binary_cross_entropy_with_logits(logits, labels)
#     return loss

# # -------------------------
# # Utilities to build adjacency
# # -------------------------
# def build_adj_from_pairs(num_nodes: int, relationships: torch.Tensor, directed: bool = True, self_loops: bool = True, device: str = "cpu") -> torch.Tensor:
#     """
#     relationships: [num_pairs, 2] (integer indices). value 1 means link (following), 0 means no link.
#     Returns adjacency [N, N] float tensor.
#     If relationships contains only positive pairs, you can still call this by providing those pairs.
#     """
#     adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
#     if relationships.numel() == 0:
#         if self_loops:
#             adj.fill_diagonal_(1.0)
#         return adj

#     # relationships could be shape [num_pairs, 2] with pairs only (we treat them as ones)
#     # If relationships is provided as [num_pairs, 3] where last col is value, handle that
#     if relationships.size(1) == 3:
#         idx = relationships[:, :2].long()
#         vals = relationships[:, 2].float()
#         for (i, j), v in zip(idx.tolist(), vals.tolist()):
#             adj[i, j] = v
#             if not directed:
#                 adj[j, i] = v
#     else:
#         # assume pairs -> value 1
#         idx = relationships.long()
#         for i, j in idx.tolist():
#             adj[i, j] = 1.0
#             if not directed:
#                 adj[j, i] = 1.0

#     if self_loops:
#         adj.fill_diagonal_(1.0)
#     return adj

def build_adj_from_pairs(num_nodes, relationships, labels, directed=True, self_loops=True, device="cpu"):
    """Builds adjacency matrix using relationship pairs and binary labels."""
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    for (i, j), lbl in zip(relationships.tolist(), labels.tolist()):
        adj[i, j] = lbl
        if not directed:
            adj[j, i] = lbl
    if self_loops:
        adj.fill_diagonal_(1.0)
    return adj


def kl_divergence(mu, logvar):
    """KL divergence between N(mu, sigma) and N(0, I)."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def reconstruction_loss(scores, adj):
    """Binary cross-entropy between predicted logits and true adjacency."""
    return F.binary_cross_entropy_with_logits(scores.view(-1), adj.view(-1))