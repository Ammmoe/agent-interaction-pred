import torch
import torch.nn as nn
import torch.optim as optim


# class DroneRelationModel(nn.Module):
#     def __init__(self, context_dim, hidden_dim=128, device="cuda"):
#         super().__init__()
#         self.device = device
#         self.input_dim = 4 + context_dim  # (x,y,z,type) + context embedding
#         self.hidden_dim = hidden_dim

#         self.gru = nn.GRU(input_size=self.input_dim, hidden_size=hidden_dim, batch_first=True)

#         # Relation classifier head
#         self.relation_head = nn.Sequential(
#             nn.Linear(2 * hidden_dim, hidden_dim),
#             # nn.ReLU(),
#             # nn.Linear(hidden_dim, 1),
#             # nn.ReLU(),
#             # nn.Linear(hidden_dim // 2, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, current_features, context_embeddings, relationships):
#         """
#         Args:
#             current_features: [num_drones, 4]
#             context_embeddings: [num_drones, context_dim]
#             relationships: [num_pairs, 2] (pairs of drone indices)
#         Returns:
#             preds: [num_pairs]
#         """
#         x = torch.cat([current_features, context_embeddings], dim=1)  # [N, input_dim]
#         x = x.unsqueeze(1)  # GRU expects sequence
#         _, h = self.gru(x)  # h: [1, N, hidden_dim]
#         node_emb = h.squeeze(0)  # [N, hidden_dim]

#         src = node_emb[relationships[:, 0]]
#         dst = node_emb[relationships[:, 1]]
#         pair_emb = torch.cat([src, dst], dim=1)  # [num_pairs, 2*hidden_dim]
#         preds = self.relation_head(pair_emb).squeeze(1)  # [num_pairs]
#         return preds


import torch
import torch.nn as nn

class DroneRelationModel(nn.Module):
    def __init__(self, context_dim, hidden_dim=None, num_heads=4, device="cpu"):
        super().__init__()
        self.device = device
        if hidden_dim is None:
            hidden_dim = context_dim
        self.hidden_dim = hidden_dim

        # Self-attention block
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # Relation prediction head
        self.relation_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, current_features, context_embeddings, relationships):
        """
        Args:
            current_features: [num_drones, 4]
            context_embeddings: [num_drones, hidden_dim]
            relationships: [num_pairs, 2]
        """
        # Self-attention refinement
        x = context_embeddings.unsqueeze(0)  # add batch dim: [1, num_drones, hidden_dim]
        refined_emb, attn_weights = self.self_attn(x, x, x)  # Self-attention
        node_emb = refined_emb.squeeze(0)  # [num_drones, hidden_dim]

        # Pairwise relation prediction
        src = node_emb[relationships[:, 0]]
        dst = node_emb[relationships[:, 1]]
        pair_emb = torch.cat([src, dst], dim=1)
        preds = self.relation_head(pair_emb).squeeze(1)
        return preds
