import cv2
import numpy as np


class Frame:

    def __init__(self, 
                 frame_id: str,
                 path_im: str,
                 path_lidar: str,
                 path_calib: str) -> None

        self._frame_id = frame_id
        self._path_im = path_im
        self._path_lidar = path_lidar
        self._path_calib = path_calib

        self.calib_matrix = np.empty((3,4))

    def load_image(self) -> None:

        self.img = cv2.imread(self._path_im+self._frame_id+'.png')
        self.H, self.W = self.img.shape[:2] #type: ignore
        
    def load_calibration(self) -> None:

        with open(file=self._path_calib+self._frame_id+'.txt', mode='r') as doc:
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
        
        self.calib_matrix = p2 @ r0_rect @ tr_velo_to_cam

    def project_lidar(self) -> None:

        raw_points = np.fromfile(self._path_lidar+self._frame_id+'.bin', dtype=np.float32).reshape(-1, 4)

        # convert to image pixels

        pts, reflectance = raw_points[:, :3], raw_points[:, 3]
        n = pts.shape[0]
        h_pts = np.hstack([pts, np.ones((n, 1))]) # to homogeneous coordinates
        im_pts = self.calib_matrix @ h_pts.T # (3x4) @ (4, N) -> needs (x,y,z,1) vertically

        u = im_pts[0] / im_pts[2] # back to cartesian
        v = im_pts[1] / im_pts[2]

        # only keep valid points: ahead of vehicle + finite & not NaN
        depth = im_pts[2] # z in camera frame (metres)
        valid = (depth > 0) & (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H) & np.isfinite(reflectance)
        d_v, u_v, v_v, r_v = depth[valid], u[valid], v[valid], reflectance[valid]

        self.im_pts = im_pts # points x,y,z in metres
        self.depth = d_v
        self.u = u_v
        self.v = v_v
        self.r = r_v

    def stats(self):
        print(f'Frame ID: {self._frame_id}')
        print(f'Image Resolution: {self.W}x{self.H}')
        print(f'Raw Lidar: {self.im_pts.shape[1]} points')
        print(f'Filtered Lidar: {len(self.u)} points')
        print('Depth:')
        print(f'\t Min: {self.depth.min()} m')
        print(f'\t Max: {self.depth.max()} m')
        print(f'\t Median: {np.median(self.depth)} m')

    def load_camera_lidar(self, verbose: bool) -> tuple:

        self.load_image()
        self.load_calibration()
        self.project_lidar()

        if verbose:
            self.stats()

        return self.raw_lidar, self.u, self.v, self.r
    