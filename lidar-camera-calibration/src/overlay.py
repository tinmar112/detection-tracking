import cv2
import numpy as np

from calibrator import Calibrator


def overlay(path_im: str, path_lidar:str , path_calib: str) -> None:

    calibrator = Calibrator(path_im=path_im, path_lidar=path_lidar, path_calib=path_calib)

    calibrator.load_matrices()
    _, u, v, r = calibrator.load_lidar()

    # map reflectance to a colour
    r_norm = np.clip(r, 0, 1)
    colors = cv2.applyColorMap(
        (r_norm * 255).astype(np.uint8), cv2.COLORMAP_JET
    ).reshape(-1, 3)                             # (M, 3) BGR

    # draw each point as a filled circle on a copy
    overlay = calibrator.img.copy() #type: ignore
    for i in range(len(u)):
        cv2.circle(overlay, (int(u[i]), int(v[i])),
                radius=2, color=colors[i].tolist(), thickness=-1)

    # blend so points don't fully hide the road scene
    out = cv2.addWeighted(overlay, 0.7, calibrator.img, 0.3, 0) #type: ignore

    cv2.imshow("lidar overlay", out); cv2.waitKey(0)
