"""
pretrained_model_loader.py

Loads a pretrained multi-agent trajectory prediction model (e.g., AttentionBiGRUPredictor)
and extracts context embeddings from the attention mechanism for downstream relationship modeling.
"""

from pathlib import Path
from importlib import import_module
import torch
import json
import joblib
import numpy as np
from utils.scaler import scale_per_agent


def load_pretrained_traj_model(experiment_dir, device=None):
    """
    Loads pretrained TrajPredictor model and configuration.

    Args:
        experiment_dir (str or Path): Path to experiment directory containing model + config.
        device (torch.device, optional): Device to load the model to. Defaults to cuda if available.

    Returns:
        model: Loaded PyTorch model in eval mode.
        config: Experiment configuration dictionary.
    """
    experiment_dir = Path(experiment_dir)
    CONFIG_PATH = experiment_dir / "config.json"
    MODEL_PATH = experiment_dir / "last_model.pt"

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Import model module dynamically
    model_module = import_module(config["model_module"])
    ModelClass = getattr(model_module, config["model_class"])
    model = ModelClass(**config["model_params"]).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    return model, config


def extract_context_embeddings(model, traj_data, scaler_X, lookback, features_per_agent, device):
    """
    Runs encoder-attention pipeline and extracts attention context embeddings.

    Args:
        model: Loaded trajectory prediction model (TrajPredictor)
        traj_data: np.ndarray [T, num_agents * features_per_agent]
        scaler_X: Fitted input scaler (joblib)
        lookback (int): Number of timesteps for encoder input
        features_per_agent (int): Input feature dimension per agent
        device: torch.device

    Returns:
        context_embeddings: torch.Tensor [num_agents, hidden_dim] — final context per agent
    """
    # Scale trajectory data per agent (same as training)
    traj_scaled = scale_per_agent(traj_data, scaler_X, features_per_agent)

    # Take only last lookback segment
    input_seq = traj_scaled[-lookback:]
    X_tensor = torch.from_numpy(input_seq.reshape(1, lookback, -1)).float().to(device)

    # Extract encoder outputs and context manually
    _, _, total_features = X_tensor.size()
    num_agents = total_features // model.input_size
    src_agents = torch.split(X_tensor, model.input_size, dim=2)

    context_embeddings = []
    with torch.no_grad():
        for agent_idx in range(num_agents):
            enc_output, hidden = model.encoder(src_agents[agent_idx])
            num_directions = 2
            hidden_cat = (
                torch.cat([hidden[-2], hidden[-1]], dim=1)
                if num_directions == 2
                else hidden[-1]
            )
            hidden_dec = model.enc_to_dec(hidden_cat).unsqueeze(0).repeat(
                model.num_layers, 1, 1
            )

            # --- Compute attention and context vector ---
            attn_weights = model.attention(hidden_dec[-1], enc_output)
            context = torch.bmm(attn_weights.unsqueeze(1), enc_output).squeeze(1)
            context_embeddings.append(context)

        context_embeddings = torch.cat(context_embeddings, dim=0)  # [num_agents, enc_hidden*2]
        return context_embeddings


def extract_decoder_embeddings(model, traj_data, scaler_X, lookback, features_per_agent, device):
    """
    Extract per-agent embeddings from the decoder GRU step.
    """
    traj_scaled = scale_per_agent(traj_data, scaler_X, features_per_agent)
    input_seq = traj_scaled[-lookback:]
    X_tensor = torch.from_numpy(input_seq.reshape(1, lookback, -1)).float().to(device)

    _, _, total_features = X_tensor.size()
    num_agents = total_features // model.input_size
    src_agents = torch.split(X_tensor, model.input_size, dim=2)

    agent_embeddings = []
    for agent_idx in range(num_agents):
        # Encoder
        enc_output, hidden = model.encoder(src_agents[agent_idx])
        num_directions = 2
        hidden_cat = (
            torch.cat([hidden[-2], hidden[-1]], dim=1)
            if num_directions == 2
            else hidden[-1]
        )
        hidden_dec = model.enc_to_dec(hidden_cat).unsqueeze(0).repeat(model.num_layers, 1, 1)

        # Decoder step
        dec_input = src_agents[agent_idx][:, -1:, :]  # last step
        rnn_input = torch.cat((dec_input, enc_output[:, -1:, :]), dim=2)  # optional: use attention context
        dec_out, hidden_dec = model.decoder(rnn_input, hidden_dec)

        # Take final hidden state as embedding
        agent_embeddings.append(hidden_dec[-1])  # [1, hidden_dim]

    agent_embeddings = torch.cat(agent_embeddings, dim=0)  # [num_agents, hidden_dim]
    return agent_embeddings
