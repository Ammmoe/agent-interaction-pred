"""
attention_bi_gru_predictor.py

Defines a sequence-to-sequence GRU model with attention for predicting multi-agent trajectories.
Supports bidirectional encoder, flexible autoregressive decoding, and variable number of agents.
"""

import torch
from torch import nn


class Attention(nn.Module):
    """
    Additive (Bahdanau-style) attention mechanism.

    Computes a context vector as a weighted sum of encoder outputs for each decoder step.
    Works per agent in the variable-agent TrajPredictor.
    """

    def __init__(self, enc_hidden_size, dec_hidden_size):
        """
        Args:
            enc_hidden_size (int): Size of the encoder hidden states.
            dec_hidden_size (int): Size of the decoder hidden states.
        """
        super().__init__()
        self.attn = nn.Linear(enc_hidden_size + dec_hidden_size, dec_hidden_size)
        self.v = nn.Linear(dec_hidden_size, 1, bias=False)
        self.enc_hidden_size = enc_hidden_size

    def forward(self, hidden, encoder_outputs):
        """
        Args:
            hidden: [batch_size, dec_hidden_size] current decoder hidden state
            encoder_outputs: [batch_size, seq_len, enc_hidden_size] encoder outputs

        Returns:
            attn_weights: [batch_size, seq_len] attention weights over encoder outputs
        """
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)


class TrajPredictor(nn.Module):
    """
    Sequence-to-sequence GRU model with attention for **multi-agent trajectory prediction**.

    Features:
        - Bidirectional encoder GRU
        - Attention over encoder outputs
        - Unidirectional decoder GRU
        - Flexible decoding: teacher forcing or autoregressive
        - Supports **variable number of agents** per input

    Args:
        input_size (int): Number of input features per agent per timestep (2 or 3 for x,y,z).
        enc_hidden_size (int): Number of hidden units in the encoder GRU.
        dec_hidden_size (int): Number of hidden units in the decoder GRU.
        num_layers (int): Number of stacked GRU layers.
    """

    def __init__(
        self, input_size=6, enc_hidden_size=64, dec_hidden_size=64, num_layers=1
    ):
        super().__init__()
        self.input_size = input_size
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.num_layers = num_layers

        # Shared modules for all agents
        self.encoder = nn.GRU(
            input_size,
            enc_hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = Attention(enc_hidden_size * 2, dec_hidden_size)
        self.enc_to_dec = nn.Linear(enc_hidden_size * 2, dec_hidden_size)
        self.decoder = nn.GRU(
            input_size + enc_hidden_size * 2,
            dec_hidden_size,
            num_layers,
            batch_first=True,
        )
        self.fc_out = nn.Linear(dec_hidden_size, input_size)

    def forward(self, src, tgt=None, pred_len=1):
        """
        Forward pass for multi-agent trajectory prediction.

        Args:
            src: [batch_size, seq_len, num_agents * input_size] input past trajectories for all agents
            tgt: optional, [batch_size, pred_len, num_agents * input_size] target future trajectories (for teacher forcing)
            pred_len: int, number of steps to predict if tgt is None

        Returns:
            outputs: [batch_size, pred_len, num_agents * input_size] predicted future trajectories
        """
        _, _, total_features = src.size()
        num_agents = total_features // self.input_size
        src_agents = torch.split(src, self.input_size, dim=2)

        tgt_agents = None
        if tgt is not None:
            tgt_agents = torch.split(tgt, self.input_size, dim=2)

        outputs_per_agent = []
        for agent_idx in range(num_agents):
            # Encoder
            enc_output, hidden = self.encoder(src_agents[agent_idx])
            num_directions = 2
            hidden_cat = (
                torch.cat([hidden[-2], hidden[-1]], dim=1)
                if num_directions == 2
                else hidden[-1]
            )
            hidden_dec = (
                self.enc_to_dec(hidden_cat).unsqueeze(0).repeat(self.num_layers, 1, 1)
            )

            # Decoder
            dec_input = src_agents[agent_idx][:, -1:, :]
            agent_outputs = []
            for t in range(pred_len):
                attn_weights = self.attention(hidden_dec[-1], enc_output)
                context = torch.bmm(attn_weights.unsqueeze(1), enc_output)
                rnn_input = torch.cat((dec_input, context), dim=2)
                dec_out, hidden_dec = self.decoder(rnn_input, hidden_dec)
                pred = self.fc_out(dec_out.squeeze(1))
                agent_outputs.append(pred.unsqueeze(1))
                if tgt_agents is not None:
                    dec_input = tgt_agents[agent_idx][:, t : t + 1, :]
                else:
                    dec_input = pred.unsqueeze(1)

            outputs_per_agent.append(torch.cat(agent_outputs, dim=1))

        # --- Concatenate all agents ---
        outputs = torch.cat(
            outputs_per_agent, dim=2
        )  # [batch, pred_len, num_agents * input_size]
        return outputs
