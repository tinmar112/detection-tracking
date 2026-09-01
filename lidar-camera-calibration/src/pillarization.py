import numpy as np

def crop(raw_lidar: np.ndarray,
         x_min: float, x_max: float,
         y_min: float, y_max: float,
         z_min: float, z_max: float) -> np.ndarray:

    x, y, z, r = raw_lidar[:, 0], raw_lidar[:, 1], raw_lidar[:, 2], raw_lidar[:, 3]

    mask = (
    (x >= x_min) &
    (x < x_max) &
    (y >= y_min) &
    (y < y_max) &
    (z >= z_min) &
    (z < z_max) &
    np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(r)
)

    cropped_lidar = raw_lidar[mask]

    return cropped_lidar

def assign_pillars(cropped_lidar: np.ndarray,
                   x_min: float, delta_x: float,
                   y_min: float, delta_y: float) -> np.ndarray:

    pillar_x = np.floor((cropped_lidar[:, 0]-x_min)/delta_x).astype(np.int32)
    pillar_y = np.floor((cropped_lidar[:, 1]-y_min)/delta_y).astype(np.int32)

    return np.column_stack((pillar_x, pillar_y, cropped_lidar))


def create_pillars(indexed_lidar: np.ndarray,
                   N_x: int, N_y: int, Np_max: int = 32) -> np.ndarray:

    if indexed_lidar.size == 0:
        return np.zeros((N_y, N_x, Np_max, 4), dtype=np.float32)

    x_idx = indexed_lidar[:, 0].astype(np.int64)
    y_idx = indexed_lidar[:, 1].astype(np.int64)
    points = indexed_lidar[:, 2:].astype(np.float32)

    # Flatten (row, col) into one pillar id
    pillar_id = y_idx * N_x + x_idx

    # Group points by pillar
    order = np.argsort(pillar_id, kind="stable")
    pillar_id = pillar_id[order]
    points = points[order]

    counts = np.bincount(pillar_id, minlength=N_x * N_y)
    starts = np.concatenate(([0], np.cumsum(counts[:-1])))

    # Local index inside each pillar
    local_idx = np.arange(pillar_id.size, dtype=np.int64) - starts[pillar_id]

    # Keep only first Np_max points per pillar
    keep = local_idx < Np_max
    pillar_id = pillar_id[keep]
    points = points[keep]
    local_idx = local_idx[keep]

    rows = pillar_id // N_x
    cols = pillar_id % N_x

    pillars = np.zeros((N_y, N_x, Np_max, 4), dtype=np.float32)
    pillars[rows, cols, local_idx] = points

    return pillars

def augment_pillars(pillars: np.ndarray, pillar_coords: np.ndarray, x_min: float, delta_x: float, y_min: float, delta_y: float) -> np.ndarray:

    x_c = pillars[:,:, 0] - pillars[:,:, 0].mean()
    y_c = pillars[:,:, 1] - pillars[:,:, 1].mean()
    z_c = pillars[:,:, 2] - pillars[:,:, 2].mean()

    x_center = x_min + delta_x * pillar_coords[:, 1] + delta_x/2
    y_center = y_min + delta_y * pillar_coords[:, 0] + delta_y/2

    # shape: (n_pillars,) -> repeat to (n_pillars, Np_max)
    x_center = np.repeat(x_center[:, None], pillars.shape[1], axis=1)
    y_center = np.repeat(y_center[:, None], pillars.shape[1], axis=1)

    x_p = pillars[:,:, 0] - x_center
    y_p = pillars[:,:, 1] - y_center

    new_features = np.stack([x_c, y_c, z_c, x_p, y_p], axis=-1)
    return np.concatenate([pillars, new_features], axis=-1)
