from overlay import overlay

if __name__ == '__main__':

    frame_id = '000150'

    path_calib = '../data/data_object_calib/training/calib/'
    path_lidar = '../data/data_object_velodyne/training/velodyne/'
    path_im = '../data/data_object_image_2/training/image_2/'

    overlay(frame_id, path_im, path_lidar, path_calib)
