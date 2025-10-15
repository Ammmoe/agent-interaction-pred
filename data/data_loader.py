import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class DroneGraphDataset(Dataset):
    def __init__(self, trajectory_csv, relationship_csv, lookback=50, device='cpu'):
        self.lookback = lookback
        self.device = device

        # Load CSVs
        self.traj_df = pd.read_csv(trajectory_csv)
        self.rel_df = pd.read_csv(relationship_csv)

        # Encode roles (e.g., unauthorized=0, friendly=1)
        self.traj_df['role_id'] = LabelEncoder().fit_transform(self.traj_df['role'])

        # Sort by flight and time
        self.traj_df = self.traj_df.sort_values(['flight_id', 'time_stamp', 'drone_id']).reset_index(drop=True)
        self.rel_df = self.rel_df.sort_values(['flight_id', 'time_stamp']).reset_index(drop=True)

        # Group by flight for multi-flight datasets
        self.flights = list(self.traj_df['flight_id'].unique())
        self.flight_data = {fid: df for fid, df in self.traj_df.groupby('flight_id')}
        self.rel_data = {fid: df for fid, df in self.rel_df.groupby('flight_id')}

        # Precompute available timesteps per flight
        self.flight_timesteps = {fid: sorted(df['time_stamp'].unique()) for fid, df in self.flight_data.items()}
        self.valid_indices = self._build_indices()

    def _build_indices(self):
        """Return all valid (flight_id, current_time_index) pairs with enough lookback"""
        indices = []
        for fid, timesteps in self.flight_timesteps.items():
            for i in range(self.lookback, len(timesteps)):
                indices.append((fid, i))
        return indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        fid, i = self.valid_indices[idx]
        timesteps = self.flight_timesteps[fid]
        past_times = timesteps[i - self.lookback:i]
        current_time = timesteps[i]

        # Build feature tensors for lookback window
        past_features = []
        for t in past_times:
            df_t = self.flight_data[fid][self.flight_data[fid]['time_stamp'] == t]
            feats = df_t[['pos_x', 'pos_y', 'pos_z', 'role_id']].values
            past_features.append(feats)
        past_features = np.stack(past_features)  # [lookback, num_drones, feature_dim]

        # Current features
        current_df = self.flight_data[fid][self.flight_data[fid]['time_stamp'] == current_time]
        current_features = current_df[['pos_x', 'pos_y', 'pos_z', 'role_id']].values

        # Relationship labels at current timestep
        rel_t = self.rel_data[fid][self.rel_data[fid]['time_stamp'] == current_time]
        rel_pairs = rel_t[['drone_id', 'target_id']].values
        rel_labels = (rel_t['relationship'] == 'following').astype(np.float32).values

        return {
            'context_window': torch.tensor(past_features, dtype=torch.float32, device=self.device),
            'current_features': torch.tensor(current_features, dtype=torch.float32, device=self.device),
            'relationships': torch.tensor(rel_pairs, dtype=torch.long, device=self.device),
            'labels': torch.tensor(rel_labels, dtype=torch.float32, device=self.device),
            'flight_id': fid,
            'time_stamp': current_time
        }
