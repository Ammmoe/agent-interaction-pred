import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class DroneGraphDataset(Dataset):
    def __init__(self, trajectory_csv, relationship_csv, lookback=50, device="cpu"):
        self.lookback = lookback
        self.device = device

        # Load CSVs
        self.traj_df = pd.read_csv(trajectory_csv)
        self.rel_df = pd.read_csv(relationship_csv)

        # Encode roles (e.g., unauthorized=0, friendly=1)
        self.traj_df["role_id"] = LabelEncoder().fit_transform(self.traj_df["role"])

        # Sort by flight and time
        self.traj_df = self.traj_df.sort_values(
            ["flight_id", "time_stamp", "drone_id"]
        ).reset_index(drop=True)
        self.rel_df = self.rel_df.sort_values(["flight_id", "time_stamp"]).reset_index(
            drop=True
        )

        # --- Compute accelerations if not present ---
        if not {"acc_x", "acc_y", "acc_z"}.issubset(self.traj_df.columns):
            self.traj_df[["acc_x", "acc_y", "acc_z"]] = self._compute_accelerations(
                self.traj_df
            )

        # Group by flight for multi-flight datasets
        self.flights = list(self.traj_df["flight_id"].unique())
        self.flight_data = {fid: df for fid, df in self.traj_df.groupby("flight_id")}
        self.rel_data = {fid: df for fid, df in self.rel_df.groupby("flight_id")}

        # Precompute available timesteps per flight
        self.flight_timesteps = {
            fid: sorted(df["time_stamp"].unique())
            for fid, df in self.flight_data.items()
        }
        self.valid_indices = self._build_indices()

    def _compute_accelerations(self, df):
        """Compute per-drone acceleration using finite differencing with time step normalization."""
        acc_list = np.zeros((len(df), 3), dtype=np.float32)

        for fid, flight_df in df.groupby("flight_id"):
            for drone_id, drone_df in flight_df.groupby("drone_id"):
                v = drone_df[["vel_x", "vel_y", "vel_z"]].values
                t = drone_df["time_stamp"].values

                a = np.zeros_like(v)
                if len(v) > 1:
                    dt = np.diff(t)
                    # Prevent division by zero or near-zero time intervals
                    dt[dt == 0] = 1e-6
                    a[1:] = (v[1:] - v[:-1]) / dt[:, None]
                acc_list[drone_df.index] = a

        return pd.DataFrame(acc_list, columns=["acc_x", "acc_y", "acc_z"])

    def _build_indices(self):
        indices = []
        for fid, timesteps in self.flight_timesteps.items():
            for i in range(self.lookback, len(timesteps)):
                current_time = timesteps[i]
                df_current = self.flight_data[fid][
                    self.flight_data[fid]["time_stamp"] == current_time
                ]
                if len(df_current) == 6:  # only keep full timesteps
                    indices.append((fid, i))
                else:
                    print(
                        f"[Skipping] Flight {fid}, timestep {current_time} has {len(df_current)} drones"
                    )
        return indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        fid, i = self.valid_indices[idx]
        timesteps = self.flight_timesteps[fid]
        past_times = timesteps[i - self.lookback : i]
        current_time = timesteps[i]

        # --- Build feature tensors for lookback window ---
        past_features = []
        for t in past_times:
            df_t = self.flight_data[fid][self.flight_data[fid]["time_stamp"] == t]
            feats = df_t[
                [
                    "pos_x", "pos_y", "pos_z",
                    "vel_x", "vel_y", "vel_z",
                    "acc_x", "acc_y", "acc_z",
                    "role_id",
                ]
            ].values
            past_features.append(feats)
        past_features = np.stack(past_features)  # [lookback, num_drones, feature_dim]

        # --- Current features ---
        current_df = self.flight_data[fid][self.flight_data[fid]["time_stamp"] == current_time]
        current_features = current_df[
            [
                "pos_x", "pos_y", "pos_z",
                "vel_x", "vel_y", "vel_z",
                "acc_x", "acc_y", "acc_z",
                "role_id",
            ]
        ].values

        # --- Relationship labels ---
        rel_t = self.rel_data[fid][self.rel_data[fid]["time_stamp"] == current_time]
        rel_pairs = rel_t[["drone_id", "target_id"]].values
        rel_labels = (rel_t["relationship"] == "following").astype(np.float32).values

        return {
            "context_window": torch.tensor(past_features, dtype=torch.float32, device=self.device),
            "current_features": torch.tensor(current_features, dtype=torch.float32, device=self.device),
            "relationships": torch.tensor(rel_pairs, dtype=torch.long, device=self.device),
            "labels": torch.tensor(rel_labels, dtype=torch.float32, device=self.device),
            "flight_id": fid,
            "time_stamp": current_time,
        }


# class DroneGraphDataset(Dataset):
#     def __init__(self, trajectory_csv, relationship_csv, lookback=50, device="cpu"):
#         self.lookback = lookback
#         self.device = device

#         # Load CSVs
#         self.traj_df = pd.read_csv(trajectory_csv)
#         self.rel_df = pd.read_csv(relationship_csv)

#         # Encode roles (e.g., unauthorized=0, friendly=1)
#         self.traj_df["role_id"] = LabelEncoder().fit_transform(self.traj_df["role"])

#         # Sort by flight and time
#         self.traj_df = self.traj_df.sort_values(
#             ["flight_id", "time_stamp", "drone_id"]
#         ).reset_index(drop=True)
#         self.rel_df = self.rel_df.sort_values(["flight_id", "time_stamp"]).reset_index(
#             drop=True
#         )

#         # Group by flight for multi-flight datasets
#         self.flights = list(self.traj_df["flight_id"].unique())
#         self.flight_data = {fid: df for fid, df in self.traj_df.groupby("flight_id")}
#         self.rel_data = {fid: df for fid, df in self.rel_df.groupby("flight_id")}

#         # Precompute available timesteps per flight
#         self.flight_timesteps = {
#             fid: sorted(df["time_stamp"].unique())
#             for fid, df in self.flight_data.items()
#         }
#         self.valid_indices = self._build_indices()

#     def _build_indices(self):
#         indices = []
#         for fid, timesteps in self.flight_timesteps.items():
#             for i in range(self.lookback, len(timesteps)):
#                 current_time = timesteps[i]
#                 df_current = self.flight_data[fid][
#                     self.flight_data[fid]["time_stamp"] == current_time
#                 ]
#                 if len(df_current) == 6:  # only keep full timesteps
#                     indices.append((fid, i))
#                 else:
#                     print(
#                         f"[Skipping] Flight {fid}, timestep {current_time} has {len(df_current)} drones"
#                     )
#         return indices

#     def __len__(self):
#         return len(self.valid_indices)

#     def __getitem__(self, idx):
#         fid, i = self.valid_indices[idx]
#         timesteps = self.flight_timesteps[fid]
#         past_times = timesteps[i - self.lookback:i]
#         current_time = timesteps[i]

#         # Build feature tensors for lookback window
#         past_features = []
#         for t in past_times:
#             df_t = self.flight_data[fid][self.flight_data[fid]['time_stamp'] == t]
#             feats = df_t[['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'role_id']].values
#             past_features.append(feats)
#         past_features = np.stack(past_features)  # [lookback, num_drones, feature_dim]

#         # Current features
#         current_df = self.flight_data[fid][self.flight_data[fid]['time_stamp'] == current_time]
#         current_features = current_df[['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'role_id']].values

#         # Relationship labels at current timestep
#         rel_t = self.rel_data[fid][self.rel_data[fid]['time_stamp'] == current_time]
#         rel_pairs = rel_t[['drone_id', 'target_id']].values
#         rel_labels = (rel_t['relationship'] == 'following').astype(np.float32).values

#         return {
#             'context_window': torch.tensor(past_features, dtype=torch.float32, device=self.device),
#             'current_features': torch.tensor(current_features, dtype=torch.float32, device=self.device),
#             'relationships': torch.tensor(rel_pairs, dtype=torch.long, device=self.device),
#             'labels': torch.tensor(rel_labels, dtype=torch.float32, device=self.device),
#             'flight_id': fid,
#             'time_stamp': current_time
#         }

    # def __getitem__(self, idx):
    #     fid, i = self.valid_indices[idx]
    #     timesteps = self.flight_timesteps[fid]
    #     past_times = timesteps[i - self.lookback:i]
    #     current_time = timesteps[i]

    #     # Build feature tensors for lookback window
    #     past_features = []
    #     for t in past_times:
    #         df_t = self.flight_data[fid][self.flight_data[fid]['time_stamp'] == t]
    #         feats = df_t[['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'role_id']].values
    #         past_features.append(feats)
    #     past_features = np.stack(past_features)  # [lookback, num_drones, feature_dim]

    #     # Current features
    #     current_df = self.flight_data[fid][self.flight_data[fid]['time_stamp'] == current_time]
    #     current_features = current_df[['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'role_id']].values
    #     drone_ids = current_df['drone_id'].values

    #     # Map global drone_id -> local index
    #     drone_id_to_idx = {drone_id: i for i, drone_id in enumerate(drone_ids)}

    #     # Identify friendly vs unauthorized drones (local indices)
    #     friendly_idx = [drone_id_to_idx[drone_id] for drone_id in current_df[current_df['role_id'] == 1]['drone_id']]
    #     unauth_idx = [drone_id_to_idx[drone_id] for drone_id in current_df[current_df['role_id'] == 0]['drone_id']]

    #     num_friendly = len(friendly_idx)
    #     num_unauth = len(unauth_idx)

    #     # Build relationships: all friendly -> all unauthorized pairs
    #     relationships = []
    #     target_indices = []
    #     for f_local_idx in friendly_idx:
    #         found_target = False
    #         for u_local_idx, u_global_idx in enumerate(unauth_idx):
    #             relationships.append([f_local_idx, u_global_idx])

    #             # Check if this unauthorized drone is actually followed
    #             rel_row = self.rel_data[fid][
    #                 (self.rel_data[fid]['time_stamp'] == current_time) &
    #                 (self.rel_data[fid]['drone_id'] == drone_ids[f_local_idx]) &
    #                 (self.rel_data[fid]['target_id'] == drone_ids[u_global_idx])
    #             ]
    #             if not rel_row.empty and rel_row['relationship'].values[0] == 'following':
    #                 target_indices.append(u_local_idx)  # index within unauthorized drones
    #                 found_target = True

    #         if not found_target:
    #             target_indices.append(0)  # optional: default if no following

    #     return {
    #         'context_window': torch.tensor(past_features, dtype=torch.float32, device=self.device),
    #         'current_features': torch.tensor(current_features, dtype=torch.float32, device=self.device),
    #         'relationships': torch.tensor(np.array(relationships), dtype=torch.long, device=self.device),
    #         'target_indices': torch.tensor(target_indices, dtype=torch.long, device=self.device),
    #         'num_friendly': num_friendly,
    #         'num_unauth': num_unauth,
    #         'flight_id': fid,
    #         'time_stamp': current_time
    #     }

    # def __getitem__(self, idx):
    #     fid, i = self.valid_indices[idx]
    #     timesteps = self.flight_timesteps[fid]
    #     past_times = timesteps[i - self.lookback : i]
    #     current_time = timesteps[i]

    #     # -------------------------------
    #     # 1️⃣ Build lookback feature tensors
    #     # -------------------------------
    #     past_features = []
    #     for t in past_times:
    #         df_t = self.flight_data[fid][self.flight_data[fid]["time_stamp"] == t]
    #         feats = df_t[["pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z", "role_id"]].values
    #         past_features.append(feats)
    #     past_features = np.stack(past_features)  # [lookback, num_drones, feature_dim]

    #     # -------------------------------
    #     # 2️⃣ Current timestep features
    #     # -------------------------------
    #     current_df = self.flight_data[fid][
    #         self.flight_data[fid]["time_stamp"] == current_time
    #     ]
    #     current_features = current_df[["pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z", "role_id"]].values
    #     drone_ids = current_df["drone_id"].values  # global IDs

    #     # Map global drone_id → local index for tensor indexing
    #     drone_id_to_idx = {d: i for i, d in enumerate(drone_ids)}

    #     # -------------------------------
    #     # 3️⃣ Identify friendlies vs unauthorized
    #     # -------------------------------
    #     friendly_df = current_df[current_df["role_id"] == 0]
    #     unauth_df = current_df[current_df["role_id"] == 1]

    #     friendly_ids = friendly_df["drone_id"].values
    #     unauth_ids = unauth_df["drone_id"].values

    #     friendly_idx = [drone_id_to_idx[d] for d in friendly_ids]
    #     unauth_idx = [drone_id_to_idx[d] for d in unauth_ids]

    #     num_friendly = len(friendly_idx)
    #     num_unauth = len(unauth_idx)

    #     if num_friendly == 0 or num_unauth == 0:
    #         raise ValueError(
    #             f"No friendly or unauthorized drones at time {current_time} in flight {fid}"
    #         )

    #     # -------------------------------
    #     # 4️⃣ Build relationships and target labels
    #     # -------------------------------
    #     relationships = []
    #     target_indices = []

    #     for f_local_idx, f_id in zip(friendly_idx, friendly_ids):
    #         found_target = False
    #         for u_local_pos, u_id in enumerate(unauth_ids):
    #             u_local_idx = drone_id_to_idx[u_id]
    #             relationships.append([f_local_idx, u_local_idx])

    #             # Step 1: select rows at current timestamp
    #             mask_time = np.isclose(
    #                 self.rel_data[fid]["time_stamp"].values,
    #                 float(current_time),
    #                 atol=1e-4,
    #             )
    #             rows_at_time = self.rel_data[fid][mask_time]

    #             # Step 2: filter by friendly drone (global ID)
    #             rows_for_friendly = rows_at_time[rows_at_time["drone_id"].values == f_id]

    #             # Step 3: filter by unauthorized target (global ID)
    #             rel_row = rows_for_friendly[rows_for_friendly["target_id"].values == u_id]

    #             if not rel_row.empty and rel_row["relationship"].values[0] == "following":
    #                 target_indices.append(u_local_pos)  # correct index within unauthorized drones
    #                 found_target = True

    #         if not found_target:
    #             target_indices.append(0)


    #     # -------------------------------
    #     # 5️⃣ Convert to tensors
    #     # -------------------------------
    #     return {
    #         "context_window": torch.tensor(
    #             past_features, dtype=torch.float32, device=self.device
    #         ),
    #         "current_features": torch.tensor(
    #             current_features, dtype=torch.float32, device=self.device
    #         ),
    #         "relationships": torch.tensor(
    #             np.array(relationships), dtype=torch.long, device=self.device
    #         ),
    #         "target_indices": torch.tensor(
    #             target_indices, dtype=torch.long, device=self.device
    #         ),
    #         "num_friendly": num_friendly,
    #         "num_unauth": num_unauth,
    #         "flight_id": fid,
    #         "time_stamp": current_time,
    #     }
