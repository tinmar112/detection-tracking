import numpy as np
import torch

from object3d import Object3D  #type: ignore


class Pillarization:

    def __init__(self,
                 x_min: float, x_max: float,
                 y_min: float, y_max: float,
                 z_min: float, z_max: float,
                 delta_x: float, delta_y: float):

        self._x_min = x_min
        self._x_max = x_max
        self._y_min = y_min
        self._y_max = y_max
        self._z_min = z_min
        self._z_max = z_max
        self._delta_x = delta_x
        self._delta_y = delta_y

        self._N_x, self._N_y = int((x_max-x_min)/delta_x), int((y_max-y_min)/delta_y)

    def crop(self, raw_lidar: np.ndarray) -> np.ndarray:

        x, y, z, r = raw_lidar[:, 0], raw_lidar[:, 1], raw_lidar[:, 2], raw_lidar[:, 3]

        mask = (
        (x >= self._x_min) &
        (x < self._x_max) &
        (y >= self._y_min) &
        (y < self._y_max) &
        (z >= self._z_min) &
        (z < self._z_max) &
        np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(r)
        )

        cropped_lidar = raw_lidar[mask]

        return cropped_lidar

    def assign_pillars(self, cropped_lidar: np.ndarray) -> np.ndarray:

        pillar_x = np.floor((cropped_lidar[:, 0]-self._x_min)/self._delta_x).astype(np.int32)
        pillar_y = np.floor((cropped_lidar[:, 1]-self._y_min)/self._delta_y).astype(np.int32)

        return np.column_stack((pillar_x, pillar_y, cropped_lidar))

    def create_pillars(self, indexed_lidar: np.ndarray,
                       Np_max: int = 32) -> np.ndarray:

        if indexed_lidar.size == 0:
            return np.zeros((self._N_y, self._N_x, Np_max, 4), dtype=np.float32)

        x_idx = indexed_lidar[:, 0].astype(np.int64)
        y_idx = indexed_lidar[:, 1].astype(np.int64)
        points = indexed_lidar[:, 2:].astype(np.float32)

        # Flatten (row, col) into one pillar id
        pillar_id = y_idx * self._N_x + x_idx

        # Group points by pillar
        order = np.argsort(pillar_id, kind="stable")
        pillar_id = pillar_id[order]
        points = points[order]

        counts = np.bincount(pillar_id, minlength=self._N_x*self._N_y)
        starts = np.concatenate(([0], np.cumsum(counts[:-1])))

        # Local index inside each pillar
        local_idx = np.arange(pillar_id.size, dtype=np.int64) - starts[pillar_id]

        # Keep only first Np_max points per pillar
        keep = local_idx < Np_max
        pillar_id = pillar_id[keep]
        points = points[keep]
        local_idx = local_idx[keep]

        rows = pillar_id // self._N_x
        cols = pillar_id % self._N_x

        pillars = np.zeros((self._N_y, self._N_x, Np_max, 4), dtype=np.float32)
        pillars[rows, cols, local_idx] = points

        return pillars

    def augment_pillars(self, pillars: np.ndarray,
                        pillar_coords: np.ndarray) -> np.ndarray:

        x_c = pillars[:,:, 0] - pillars[:,:, 0].mean()
        y_c = pillars[:,:, 1] - pillars[:,:, 1].mean()
        z_c = pillars[:,:, 2] - pillars[:,:, 2].mean()

        x_center = (self._x_min + self._delta_x * pillar_coords[:, 1] + self._delta_x/2).astype(np.float32)
        y_center = (self._y_min + self._delta_y * pillar_coords[:, 0] + self._delta_y/2).astype(np.float32)

        # shape: (n_pillars,) -> repeat to (n_pillars, Np_max)
        x_center = np.repeat(x_center[:, None], pillars.shape[1], axis=1)
        y_center = np.repeat(y_center[:, None], pillars.shape[1], axis=1)

        x_p = pillars[:,:, 0] - x_center
        y_p = pillars[:,:, 1] - y_center

        new_features = np.stack([x_c, y_c, z_c, x_p, y_p], axis=-1).astype(np.float32)
        return np.concatenate([pillars, new_features], axis=-1).astype(np.float32)

    def scatter_bev(self, pillar_features: torch.Tensor, pillar_coords: np.ndarray):

        coords = torch.as_tensor(pillar_coords, device=pillar_features.device, dtype=torch.long)

        bev = torch.zeros(
            64, self._N_y, self._N_x,
            device=pillar_features.device,
            dtype=pillar_features.dtype,
        )

        bev[:, coords[:, 0], coords[:, 1]] = pillar_features.T
        
        return bev

    def object3d_to_targets(self,
                            objects: list[Object3D],
                            filter: str,
                            conv_matrix: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        cls_target = np.zeros((self._N_y, self._N_x), dtype=np.float32)
        reg_target = np.zeros((6, self._N_y, self._N_x), dtype=np.float32)
        yaw_target = np.zeros((2, self._N_y, self._N_x),dtype=np.float32)

        for obj in objects:

            if obj.type != filter:
                continue

            # camera to LIDAR coordinates
            xyz_cam = np.array([[obj.loc_x, obj.loc_y, obj.loc_z, 1.]], dtype=np.float32).T

            xyz_lidar_h = conv_matrix @ xyz_cam
            xyz_lidar = (xyz_lidar_h[:3,:]/xyz_lidar_h[3,:]).ravel()

            x, y, z = xyz_lidar

            # Check whether object centre lies inside BEV ROI
            if not (self._x_min <= x < self._x_max and self._y_min <= y < self._y_max):
                continue

            # convert centre to BEV cell
            grid_x = int(np.floor((x - self._x_min) / self._delta_x))

            grid_y = int(np.floor((y - self._y_min) / self._delta_y))

            if not (0 <= grid_x < self._N_x and 0 <= grid_y < self._N_y):
                continue

            cls_target[grid_y, grid_x] = 1.0
            # [x, y, z, width, length, height]
            reg_target[:, grid_y, grid_x] = np.array([x, y, z, obj.width, obj.length, obj.height], dtype=np.float32)

            yaw = -obj.rotation_y - np.pi / 2
            yaw = (yaw + np.pi) % (2 * np.pi) - np.pi

            yaw_target[:, grid_y, grid_x] = np.array([np.sin(yaw), np.cos(yaw)], dtype=np.float32)

        return cls_target, reg_target, yaw_target
