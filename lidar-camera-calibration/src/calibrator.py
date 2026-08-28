import numpy as np


class Calibrator:

    def __init__(self) -> None:

        self.matrix = np.empty((3,4))
        
    def load_matrices(self, path :str) -> None:

        with open(file=path, mode='r') as doc:
            lines = doc.readlines()

        p2 = np.array(lines[2].split(':')[1].strip().split(),
                      dtype=np.float32).reshape(3,4)

        r0 = np.array(lines[4].split(':')[1].strip().split(),
                      dtype=np.float32).reshape(3,3)
        r0_rect = np.eye(4, dtype=np.float32) # convert to the same 4x4 rotation matrix
        r0_rect[:3, :3] = r0

        tr = np.array(lines[5].split(':')[1].strip().split(),
                      dtype=np.float32).reshape(3,4)
        tr_velo_to_cam = np.eye(4, dtype=np.float32) # add a homogeneous row to make 4x4
        tr_velo_to_cam[:3, :] = tr
        
        self.matrix = p2 @ r0_rect @ tr_velo_to_cam

    def load_lidar(self, path: str) -> tuple:

        self.lidar_points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)

        # convert to image pixels

        pts, reflectance = self.lidar_points[:, :3], self.lidar_points[:, 3]
        n = pts.shape[0]
        h_pts = np.hstack([pts, np.ones((n, 1))]) # to homogeneous coordinates
        im_pts = self.matrix @ h_pts.T # (3x4) @ (4, N) -> needs (x,y,z,1) vertically

        u = im_pts[0] / im_pts[2] # back to cartesian
        v = im_pts[1] / im_pts[2]
        #self.lidar_pixels = np.stack([u, v, reflectance], axis=1)

        return im_pts, u, v, reflectance
    