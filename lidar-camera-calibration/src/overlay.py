import cv2
import numpy as np
from frame import Frame


def overlay(frame_id: str, path_im: str, path_lidar:str , path_calib: str, path_label: str) -> None:

    frame = Frame(frame_id=frame_id, path_im=path_im, path_lidar=path_lidar, path_calib=path_calib, path_label=path_label)

    _, u, v, r = frame.load_frame(verbose=True)

    # map reflectance to a colour
    r_norm = np.clip(r, 0, 1)
    colors = cv2.applyColorMap(
        (r_norm * 255).astype(np.uint8), cv2.COLORMAP_JET
    ).reshape(-1, 3)                             # (M, 3) BGR

    # draw each point as a filled circle on a copy
    overlay = frame.img.copy() #type: ignore
    for i in range(len(u)):
        cv2.circle(overlay, (int(u[i]), int(v[i])),
                radius=2, color=colors[i].tolist(), thickness=-1)

    # blend so points don't fully hide the road scene
    out = cv2.addWeighted(overlay, 0.7, frame.img, 0.3, 0) #type: ignore

    cv2.imshow("lidar overlay", out); cv2.waitKey(0)
