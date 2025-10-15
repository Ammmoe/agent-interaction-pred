"""
Scaling utilities for multi-agent trajectory data.

This module provides helper functions to normalize or inverse-normalize
trajectory features on a per-agent basis using scikit-learn scalers
(e.g., MinMaxScaler, StandardScaler).

Each agent's features (e.g., x, y, z coordinates) are grouped together,
and the scaler is applied consistently across all agents by flattening
the agent dimension before scaling and reshaping back afterwards.

Functions
---------
scale_per_agent(data, scaler, num_features_per_agent, fit=False, inverse=False)
    Scale or inverse-scale multi-agent data while preserving the per-agent
    feature grouping. Useful for training and inference in models where the
    number of agents may vary.
"""


def scale_per_agent(data, scaler, num_features_per_agent, fit=False, inverse=False):
    """
    Scale or inverse-scale per agent.

    Args:
        data: np.ndarray, shape (..., num_agents * per_agent_features)
        scaler: MinMaxScaler fitted on per-agent features
        num_features_per_agent: int, e.g. 3 for (x,y,z)
        fit: whether to fit the scaler
        inverse: if True, apply inverse_transform instead of transform
    """
    orig_shape = data.shape
    data_reshaped = data.reshape(-1, num_features_per_agent)  # collapse agents

    if fit:
        scaler.fit(data_reshaped)

    if inverse:
        data_scaled = scaler.inverse_transform(data_reshaped)
    else:
        data_scaled = scaler.transform(data_reshaped)

    return data_scaled.reshape(orig_shape)
