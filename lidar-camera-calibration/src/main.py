from arguments import arguments
from frame import Frame

if __name__ == '__main__':

    args = arguments()

    frame = Frame(args.frame_id, args.img, args.lidar, args.calib, args.objects)
    frame.load(verbose=True)
    frame.display(boxes=True)
    frame.plot_bev(intensity='r', boxes=True)
