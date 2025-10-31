import torch
import torch.nn as nn
import torch.optim as optim


# class DroneRelationModel(nn.Module):
#     def __init__(self, context_dim, hidden_dim=128, device="cpu", num_heads=4):
#         super().__init__()
#         self.device = device
#         self.input_dim = 4 + context_dim
#         self.hidden_dim = hidden_dim

#         # Use batch_first=True for clarity
#         self.self_attn = nn.MultiheadAttention(
#             embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
#         )
#         self.gru = nn.GRU(
#             input_size=self.input_dim, hidden_size=hidden_dim, batch_first=True
#         )

#         self.relation_head = nn.Sequential(
#             nn.Linear(4 * hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, current_features, context_embeddings, relationships):
#         # refined_context_embeddings, attn_weights = self.self_attn(
#         #     context_embeddings.unsqueeze(0),
#         #     context_embeddings.unsqueeze(0),
#         #     context_embeddings.unsqueeze(0)
#         # )
#         # refined_context_embeddings = refined_context_embeddings.squeeze(0)

#         refined_context_embeddings = (
#             context_embeddings  # No self-attention, just use context embeddings
#         )

#         x = torch.cat(
#             [current_features, refined_context_embeddings], dim=1
#         )  # [N, input_dim]
#         x = x.unsqueeze(1)  # [N, 1, input_dim]
#         _, h = self.gru(x)  # h: [1, N, hidden_dim]

#         # node_emb = h.squeeze(0).unsqueeze(0)  # [1, N, hidden_dim]
#         node_emb = h.squeeze(0)  # [N, hidden_dim]

#         # Self-attention across drones
#         # refined_emb, attn_weights = self.self_attn(node_emb, node_emb, node_emb)
#         # node_emb = refined_emb.squeeze(0)  # [N, hidden_dim]

#         src = node_emb[relationships[:, 0]]
#         dst = node_emb[relationships[:, 1]]
#         pair_emb = torch.cat(
#             [src, dst, dst - src, src * dst], dim=-1
#         )  # [num_pairs, 4*decoder_hidden_dim]
#         preds = self.relation_head(pair_emb).squeeze(1)
#         return preds


import torch
import torch.nn as nn

class DroneRelationModel(nn.Module):
    def __init__(self, context_dim, hidden_dim=None, num_heads=1, device="cpu"):
        super().__init__()
        self.device = device
        if hidden_dim is None:
            hidden_dim = context_dim
        self.hidden_dim = hidden_dim

        # Self-attention block
        # self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # Relation prediction head
        self.relation_head = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)

    def forward(self, current_features, context_embeddings, relationships):
        """
        Args:
            current_features: [num_drones, 4]
            context_embeddings: [num_drones, hidden_dim]
            relationships: [num_pairs, 2]
        """
        # Self-attention refinement
        x = context_embeddings.unsqueeze(0)  # add batch dim: [1, num_drones, hidden_dim]
        # refined_emb, attn_weights = self.self_attn(x, x, x)  # Self-attention
        # node_emb = refined_emb.squeeze(0)  # [num_drones, hidden_dim]
        node_emb = x.squeeze(0)  # No self-attention, just use context embeddings

        # Pairwise relation prediction
        src = node_emb[relationships[:, 0]]
        dst = node_emb[relationships[:, 1]]
        bilinear = self.bilinear(src, dst)
        pair_emb = torch.cat([src, dst, dst - src, src * dst], dim=-1)  # [num_pairs, 4*decoder_hidden_dim]
        # pair_emb = torch.cat([dst - src], dim=-1)
        preds = self.relation_head(pair_emb).squeeze(1)
        return preds


# class DroneRelationModel(nn.Module):
#     def __init__(self, context_dim, hidden_dim=64, device="cpu", num_heads=4):
#         super().__init__()
#         self.device = device
#         self.input_dim = 4 + context_dim
#         self.hidden_dim = hidden_dim

#         # GRU over concatenated features
#         self.gru = nn.GRU(input_size=self.input_dim, hidden_size=hidden_dim, batch_first=True)

#         # Relation head outputs 1 logit per pair
#         self.relation_head = nn.Sequential(
#             nn.Linear(4 * hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1)
#         )

#     def forward(self, current_features, context_embeddings, relationships, num_friendly, num_unauth):
#         """
#         current_features: [N, 4]  (all drones)
#         context_embeddings: [N, context_dim]
#         relationships: [num_pairs, 2] (friendly_idx, unauth_idx)
#         num_friendly: number of friendly drones at this timestep
#         num_unauth: number of unauthorized drones
#         """

#         # Concatenate current features with context embeddings
#         x = torch.cat([current_features, context_embeddings], dim=1)  # [N, input_dim]
#         x = x.unsqueeze(1)  # [N, 1, input_dim]

#         # GRU over drones
#         _, h = self.gru(x)
#         node_emb = h.squeeze(0)  # [N, hidden_dim]

#         # Build pair embeddings for friendly -> unauthorized
#         src = node_emb[relationships[:, 0]]  # friendly drones repeated
#         dst = node_emb[relationships[:, 1]]  # unauthorized drones repeated
#         pair_emb = torch.cat([src, dst, src - dst, src * dst], dim=-1)  # [num_pairs, 4*decoder_hidden_dim]

#         # Compute logits per pair
#         logits = self.relation_head(pair_emb).squeeze(1)  # [num_pairs]

#         # Reshape to [num_friendly, num_unauth] for softmax
#         logits = logits.view(num_friendly, num_unauth)   # [num_friendly, num_unauth]
#         probs = torch.softmax(logits, dim=1)            # softmax over unauthorized drones

#         return probs, logits


# class DroneRelationModel(nn.Module):
#     def __init__(self, context_dim, hidden_dim=None, device="cpu"):
#         super().__init__()
#         self.device = device
#         if hidden_dim is None:
#             hidden_dim = context_dim
#         self.hidden_dim = hidden_dim

#         # Relation prediction head
#         self.relation_head = nn.Sequential(
#             nn.Linear(4 * hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1)  # raw logits
#         )

#     def forward(self, current_features, decoder_outputs, relationships, num_friendly, num_unauth):
#         """
#         Args:
#             decoder_outputs: [num_drones, decoder_hidden_dim] (output from trajectory decoder)
#             relationships: [num_pairs, 2] (friendly_idx, unauth_idx)
#             num_friendly: number of friendly drones at this timestep
#             num_unauth: number of unauthorized drones
#         """
#         # Node embeddings are directly the decoder outputs
#         node_emb = decoder_outputs  # [num_drones, decoder_hidden_dim]

#         # Pairwise relation embeddings
#         src = node_emb[relationships[:, 0]]  # friendly drones repeated
#         dst = node_emb[relationships[:, 1]]  # unauthorized drones repeated
#         pair_emb = torch.cat([src, dst, src - dst, src * dst], dim=-1)  # [num_pairs, 4*decoder_hidden_dim]

#         # Compute logits per pair
#         logits = self.relation_head(pair_emb).squeeze(1)  # [num_pairs]

#         # Reshape to [num_friendly, num_unauth] for softmax
#         logits = logits.view(num_friendly, num_unauth)   # [num_friendly, num_unauth]
#         probs = torch.softmax(logits, dim=1)            # softmax over unauthorized drones

#         return probs, logits
