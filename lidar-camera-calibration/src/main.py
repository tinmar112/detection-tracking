from frame import Frame

if __name__ == '__main__':

    frame_id = '000400'

    path_calib = './data/data_object_calib/training/calib/'
    path_lidar = './data/data_object_velodyne/training/velodyne/'
    path_img = './data/data_object_image_2/training/image_2/'
    path_label = './data/data_object_label_2/training/label_2/'

    frame = Frame(frame_id, path_img, path_lidar, path_calib, path_label)
    frame.load(verbose=False)
    frame.display(boxes=True)
