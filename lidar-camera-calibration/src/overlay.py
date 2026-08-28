import cv2
import numpy as np

from calibrator import Calibrator


def overlay(path_im: str, path_lidar:str , path_calib: str) -> None:

    img = cv2.imread(path_im)
    H, W = img.shape[:2]

    calibrator = Calibrator()
    calibrator.load_matrices(path_calib)

    proj, u, v, refl = calibrator.load_lidar(path_lidar)


    # keep only points in front of the camera and inside the image
    depth = proj[2]                              # z in camera frame (metres)
    valid = (depth > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    u_v, v_v, d_v, r_v = u[valid], v[valid], depth[valid], refl[valid]

    # 2. map reflectance (0..1) to a color — here a blue→red colormap
    r_norm = np.clip(r_v, 0, 1)
    colors = cv2.applyColorMap(
        (r_norm * 255).astype(np.uint8), cv2.COLORMAP_JET
    ).reshape(-1, 3)                             # (M, 3) BGR

    # 3. draw each point as a filled circle on a copy
    overlay = img.copy()
    for i in range(len(u_v)):
        cv2.circle(overlay, (int(u_v[i]), int(v_v[i])),
                radius=2, color=colors[i].tolist(), thickness=-1)

    # 4. blend so points don't fully hide the road scene
    out = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)

    cv2.imshow("lidar overlay", out); cv2.waitKey(0)
