from typing import Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np

from object_extractor import ObjectExtractor


class Frame:

    def __init__(self, 
                 frame_id: str,
                 path_img: str,
                 path_lidar: str,
                 path_calib: str,
                 path_label: str) -> None:

        self._frame_id = frame_id
        self._path_img = path_img
        self._path_lidar = path_lidar
        self._path_calib = path_calib
        self._path_label = path_label

    def load_image(self) -> None:

        self.img = cv2.imread(self._path_img+self._frame_id+'.png')
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

        self.lidar_to_cam = r0_rect @ tr_velo_to_cam
        self.lidar_to_pixels = p2 @ self.lidar_to_cam

    def project_lidar(self) -> None:

        try:

            raw_points = np.fromfile(self._path_lidar+self._frame_id+'.bin', dtype=np.float32).reshape(-1, 4)

            # convert to image pixels

            pts, reflectance = raw_points[:, :3], raw_points[:, 3]
            n = pts.shape[0]
            h_pts = np.hstack([pts, np.ones((n, 1))]) # to homogeneous coordinates
            im_pts = self.lidar_to_pixels @ h_pts.T # (3x4) @ (4, N) -> needs (x,y,z,1) vertically

            u = im_pts[0] / im_pts[2] # back to cartesian
            v = im_pts[1] / im_pts[2]

            # only keep valid points: ahead of vehicle + finite & not NaN
            depth = im_pts[2] # z in camera frame (metres)
            valid = (depth > 0) & (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H) & np.isfinite(reflectance)
            d_v, u_v, v_v, r_v = depth[valid], u[valid], v[valid], reflectance[valid]

            self.raw_lidar = raw_points # points x,y,z in lidar coords
            self.transformed_lidar = im_pts # points x,y,z in camera coords
            self.valid_lidar = d_v
            self.u = u_v
            self.v = v_v
            self.r = r_v

        except AttributeError:
            print('The calibration matrix must be loaded before projecting lidar coordinates onto the image.')

    def load_labels(self) -> None:

        extractor = ObjectExtractor(path_label=self._path_label)
        self.objects = extractor.extract(frame_id=self._frame_id)

    def load(self, verbose: bool) -> None:

        self.load_image()
        self.load_calibration()
        self.project_lidar()
        self.load_labels()

        if verbose:
            self.stats()
    
    def stats(self):
        print(
            f'Frame ID: {self._frame_id}\n'
            f'Image Resolution: {self.W}x{self.H}\n'
            f'Raw Lidar: {self.transformed_lidar.shape[1]} points\n'
            f'Filtered Lidar: {len(self.u)} points\n'
            f'Depth:\n'
            f'\t Min: {self.depth.min()} m\n'
            f'\t Max: {self.depth.max()} m\n'
            f'\t Median: {np.median(self.depth)} m'
            )

    def display(self, boxes: bool = True) -> None:

        r_norm = np.clip(self.r, 0, 1)

        colors = cv2.applyColorMap((r_norm * 255).astype(np.uint8), cv2.COLORMAP_JET).reshape(-1, 3)
    
        # draw each point as a filled circle on a copy
        overlay = self.img.copy() #type: ignore

        for i in range(len(self.u)):
            cv2.circle(overlay, (int(self.u[i]), int(self.v[i])),
                    radius=2, color=colors[i].tolist(), thickness=-1)
    
        if boxes: # then draw bounding boxes

            for object in self.objects:
                object.draw_2D(img=overlay)
    
        # blend so points don't fully hide the road scene
        out = cv2.addWeighted(overlay, 0.7, self.img, 0.3, 0) #type: ignore

        cv2.imshow("lidar overlay", out); cv2.waitKey(0)
    
    def plot_bev(self, intensity: Literal["z", "r"], boxes: bool = True) -> None:

        if intensity not in ("z", "r"):
            raise ValueError(f"Unsupported intensity: {intensity!r}")

        x_min, x_max = 0, 50
        y_min, y_max = -25, 25
        plt.xlim((x_min,x_max))
        plt.ylim((y_min,y_max))

        if intensity == 'z':
            plt.scatter(self.raw_lidar[:,0], self.raw_lidar[:,1],
                        s=0.5, c=self.raw_lidar[:,2], cmap='jet')
            plt.colorbar().set_label('Height $z$ (m)')
        else:
            plt.scatter(self.raw_lidar[:,0], self.raw_lidar[:,1],
                        s=0.5, c=self.raw_lidar[:,3], cmap='viridis') # reflectance
            plt.colorbar().set_label('Reflectance $r$')

        plt.xlabel('$x$ (m): ahead of vehicle')
        plt.ylabel('$y$ (m): left/right of vehicle')
        plt.title('LIDAR points (LIDAR coordinate system)\n Vehicle is at the origin.')

        if boxes:

            for object in self.objects:
                corners = object.corners_3D(conv_matrix=np.linalg.inv(self.lidar_to_cam)) # switching to lidar coordinates
                x = corners[0, [0, 1, 2, 3, 0, 4, 5, 6, 7, 4]]
                y = corners[1, [0, 1, 2, 3, 0, 4, 5, 6, 7, 4]]
                plt.plot(x, y, color='red', linewidth=3.0)

        plt.show()
