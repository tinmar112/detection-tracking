import argparse

def arguments() -> argparse.Namespace:
    """Parses command line arguments for Lidar-Camera Calibration."""

    DEFAULT_PATH_IMG = './data/data_object_image_2/training/image_2/'
    DEFAULT_PATH_LIDAR = './data/data_object_velodyne/training/velodyne/'
    DEFAULT_PATH_CALIB = './data/data_object_calib/training/calib/'
    DEFAULT_PATH_LABEL = './data/data_object_label_2/training/label_2/'

    parser = argparse.ArgumentParser(description = "LIDAR-Camera Calibration",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--frame_id","-f",type=str,
                        help="Frame ID to be displayed. ")
    
    parser.add_argument("--img","-i",type=str,
                        default=DEFAULT_PATH_IMG,
                        help="File from which the image will be loaded. ")

    parser.add_argument("--lidar","-l",type=str,
                        default=DEFAULT_PATH_LIDAR,
                        help="File from which the LIDAR points will be loaded. ")

    parser.add_argument("--calib","-c",type=str,
                        default=DEFAULT_PATH_CALIB,
                        help="File from which the calibration matrix will be loaded. ")

    parser.add_argument("--objects","-o",type=str,
                        default=DEFAULT_PATH_LABEL,
                        help="File from which the object labels will be loaded. ")
    
    return parser.parse_args()
