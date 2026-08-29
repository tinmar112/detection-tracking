from overlay import overlay

if __name__ == '__main__':

    frame_id = '000150'

    path_calib = f'../data/data_object_calib/training/calib/{frame_id}.txt'
    path_lidar = f'../data/data_object_velodyne/training/velodyne/{frame_id}.bin'
    path_im = f'../data/data_object_image_2/training/image_2/{frame_id}.png'

    overlay(path_im, path_lidar, path_calib)
