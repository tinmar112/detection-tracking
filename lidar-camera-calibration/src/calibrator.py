import cv2
import numpy as np


class Calibrator:

    def __init__(self, path_im: str,
                 path_lidar: str,
                 path_calib: str) -> None:

        self._path_im = path_im
        self._path_lidar = path_lidar
        self._path_calib = path_calib

        self.matrix = np.empty((3,4))

    def load_image(self) -> None:

        self.img = cv2.imread(self._path_im)
        self.H, self.W = self.img.shape[:2] #type: ignore
        
    def load_matrices(self) -> None:

        with open(file=self._path_calib, mode='r') as doc:
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

    def project_lidar(self) -> None:

        raw_points = np.fromfile(self._path_lidar, dtype=np.float32).reshape(-1, 4)

        # convert to image pixels

        pts, reflectance = raw_points[:, :3], raw_points[:, 3]
        n = pts.shape[0]
        h_pts = np.hstack([pts, np.ones((n, 1))]) # to homogeneous coordinates
        im_pts = self.matrix @ h_pts.T # (3x4) @ (4, N) -> needs (x,y,z,1) vertically

        u = im_pts[0] / im_pts[2] # back to cartesian
        v = im_pts[1] / im_pts[2]

        # only keep valid points
        depth = im_pts[2] # z in camera frame (metres)
        valid = (depth > 0) & (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        u_v, v_v, r_v = u[valid], v[valid], reflectance[valid]

        self.lidar_points = im_pts
        self.u = u_v
        self.v = v_v
        self.r = r_v

    def load_lidar(self) -> tuple:

        self.load_image()
        self.load_matrices()
        self.project_lidar()

        return self.lidar_points, self.u, self.v, self.r
    