import math
import random

from einops import repeat
from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd



TRAJ_ID_COL = 'traj_id'
USER_COL = 'user_id'
X_COL = 'lng'
Y_COL = 'lat'
T_COL = 'timestamp'
DT_COL = 'delta_t'
ROAD_COL = 'road'
FEATURE_PAD = 0
ST_MAP = {
    "spatial": [0, 1],
    "temporal": [2, 3]
}

KNOWN_TOKEN = 0
MASK_TOKEN = 1
START_TOKEN = 2
END_TOKEN = 3
UNKNOWN_TOKEN = 4
PAD_TOKEN = 5


def coord_transform_GPS_UTM(traj, UTM_region, origin_coord = "latlong", dest_coord = "utm"):
    from pyproj import Proj, transform
    
    if origin_coord == "latlong":
        origin = Proj(proj="latlong", datum="WGS84")
        dest = Proj(proj="utm", zone=UTM_region, datum="WGS84")  # 成都位于 UTM 第48N区
        
    elif origin_coord == "utm":
        dest = Proj(proj="latlong", datum="WGS84")
        origin = Proj(proj="utm", zone=UTM_region, datum="WGS84")  # 成都位于 UTM 第48N区

    else:
        raise NotImplementedError(f'coord type error')

    if traj.ndim == 2:
        easting, northing = transform(origin, dest, traj[:,0], traj[:,1]) # type: ignore
        traj[:,0] = easting
        traj[:,1] = northing
    elif traj.ndim == 3:
        easting, northing = transform(origin, dest, traj[:,:,0], traj[:,:,1]) # type: ignore
        traj[:,:,0] = easting
        traj[:,:,1] = northing
    return traj


class TrajFMDataset(Dataset):
    def __init__(self, traj_df, UTM_region, scale, spatial_middle_coord=None):
        """
        Dataset supporter for the Trajectory Foundation Model.

        Args:
            traj_df (pd.DataFrame): contains points of all trajectories.
        """
        super().__init__()

        self.traj_df = traj_df
        
        self.UTM_region = UTM_region
        self.scale = scale
        # self.traj_df['timestamp'] = self.traj_df['time'].apply(lambda x: x.timestamp())
        self.traj_ids = self.traj_df[TRAJ_ID_COL].unique()

        spatial_border = traj_df[[X_COL, Y_COL]]
        self.spatial_border = [spatial_border.min().tolist(), spatial_border.max().tolist()]
        
        if spatial_middle_coord is None:
            self.middle_coord = np.array([[(self.spatial_border[0][0] + self.spatial_border[1][0])/2, (self.spatial_border[0][1] + self.spatial_border[1][1])/2]]) #中心点经纬度坐标
            self.spatial_middle_coord = coord_transform_GPS_UTM(self.middle_coord, self.UTM_region)
        else:
            self.spatial_middle_coord = spatial_middle_coord
        # 进行缩放操作
        traj_gps = traj_df[[X_COL, Y_COL]].values
        traj_utm = (coord_transform_GPS_UTM(traj_gps, self.UTM_region) - self.spatial_middle_coord) / self.scale 
        
        self.traj_df[[X_COL, Y_COL]] = pd.DataFrame(traj_utm)

    def __len__(self):
        return self.traj_ids.shape[0]

    def __getitem__(self, index):
        one_traj = self.traj_df[self.traj_df[TRAJ_ID_COL] == self.traj_ids[index]].copy()
        one_traj[DT_COL] = one_traj[T_COL] - one_traj[T_COL].iloc[0]
        return one_traj


# MOD
class TULPadder:
    """Collate function for Trajectory User Classification task."""
    
    def __init__(self, num_users):
        self.num_users = num_users
    
    def __call__(self, raw_batch):
        input_batch, output_batch, pos_batch = [], [], []
        
        for traj in raw_batch:
            # Extract trajectory features, including the user ID directly
            traj_feats = traj[[X_COL, Y_COL, T_COL, DT_COL]].to_numpy()  # (L, F+1)
            user_id = traj[USER_COL].iloc[0] 
            L_out = len(traj)
            
            # Prepare input tensor
            input_row = np.stack([traj_feats, np.ones_like(traj_feats) * KNOWN_TOKEN], axis = -1)  # (L, F, 2)
            
            # Prepare output tensor (single one-hot encoding)
            output_row = np.zeros((1, self.num_users))  # (1, num_users)
            output_row[0, user_id] = 1  # One-hot encoding for the user ID
            
            # Prepare position tensor
            pos_row = np.array([[i, 0] for i in range(L_out)])  # (L_out, 2) for position
            
            # Append to batches
            input_batch.append(input_row)
            output_batch.append(output_row)
            pos_batch.append(pos_row)

        input_batch = torch.tensor(pad_batch_3d(input_batch)).float()  # (Batch, max_len, F, 2)
        output_batch = np.array(output_batch)
        output_batch = torch.tensor(output_batch).float().squeeze(1) # (Batch, 1, num_users)
        pos_batch = torch.tensor(pad_batch_2d(pos_batch)).long()  # (Batch, max_len, 2)
        return input_batch, output_batch, pos_batch
# MOD


def fetch_task_padder(padder_name, padder_params):
    
    if padder_name == 'tul':
        task_padder = TULPadder(**padder_params)
    else:
        raise NotImplementedError(f'No Padder named {padder_name}')

    return task_padder


def pad_batch_3d(batch):
    """
    Pad the batch to the maximum length of the batch.

    Args:
        batch (list): the batch of arrays to pad, [(L1, F, 2), (L2, F, 2), ...].

    Returns:
        np.array: the padded array.
    """
    max_len = max([arr.shape[0] for arr in batch])
    padded_batch = np.stack((
        np.full((len(batch), max_len, batch[0].shape[1]), FEATURE_PAD, dtype=float),
        np.full((len(batch), max_len, batch[0].shape[1]), PAD_TOKEN, dtype=float)
    ), axis=-1)
    for i, arr in enumerate(batch):
        padded_batch[i, :arr.shape[0]] = arr

    return padded_batch


def pad_batch_2d(batch):
    """
    Pad the batch to the maximum length of the batch.

    Args:
        batch (list): the batch of arrays to pad, [(L1, 2), (L2, 2), ...].

    Returns:
        np.array: the padded array.
    """
    max_len = max([arr.shape[0] for arr in batch])
    padded_batch = np.stack((
        np.full((len(batch), max_len), FEATURE_PAD, dtype=float),
        np.full((len(batch), max_len), FEATURE_PAD, dtype=float)
    ), axis=-1)
    for i, arr in enumerate(batch):
        padded_batch[i, :arr.shape[0]] = arr

    return padded_batch