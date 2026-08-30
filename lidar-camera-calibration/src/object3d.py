import cv2
import numpy as np


class Object3D:

    def __init__(self,
                 frame_id: str,
                 type: str,
                 truncated: float,
                 occluded: int,
                 alpha: float,
                 bbox_x1: float,
                 bbox_y1: float,
                 bbox_x2: float,
                 bbox_y2: float,
                 height: float,
                 width: float,
                 length: float,
                 loc_x: float,
                 loc_y: float,
                 loc_z: float,
                 rotation_y: float
                 ) -> None:

            self.frame_id = frame_id
            self.type = type
            self.truncated = truncated
            self.occluded = occluded
            self.alpha = alpha
            self.bbox_x1 = bbox_x1
            self.bbox_x2 = bbox_x2
            self.bbox_y1 = bbox_y1
            self.bbox_y2 = bbox_y2
            self.height = height
            self.width = width
            self.length = length
            self.loc_x = loc_x
            self.loc_y = loc_y
            self.loc_z = loc_z
            self.rotation_y = rotation_y

    def __str__(self) -> str:
        return (
            f"Object3D(frame_id={self.frame_id}, type={self.type}, "
            f"truncated={self.truncated}, occluded={self.occluded}, "
            f"alpha={self.alpha}, bbox=({self.bbox_x1}, {self.bbox_y1})-"
            f"({self.bbox_x2}, {self.bbox_y2}), size=({self.height}, {self.width}, "
            f"{self.length}), loc=({self.loc_x}, {self.loc_y}, {self.loc_z}), "
            f"rotation_y={self.rotation_y})"
        )

    def draw(self, img: np.ndarray) -> None:
        """Draws the 2D bounding box on a given image with object type label."""

        cv2.rectangle(img, 
                    (int(self.bbox_x1), int(self.bbox_y1)),
                    (int(self.bbox_x2), int(self.bbox_y2)),
                    color=(0, 255, 0), # green
                    thickness=2)
        
        # Draw label
        cv2.putText(img,
                    self.type,
                    (int(self.bbox_x1), int(self.bbox_y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 255, 0),
                    thickness=1)
