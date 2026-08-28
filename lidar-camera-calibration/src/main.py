from calibrator import Calibrator
from overlay import overlay

if __name__ == '__main__':

    calibrator = Calibrator()
    path_calib = '../data/data_object_calib/training/calib/000055.txt'
    calibrator.load_matrices(path=path_calib)

    path_lidar = '../data/data_object_velodyne/training/velodyne/000055.bin'
    calibrator.load_lidar(path_lidar)

    path_im = '../data/data_object_image_2/training/image_2/000055.png'

    overlay(path_im, path_lidar, path_calib)
