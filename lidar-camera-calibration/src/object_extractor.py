from object3d import Object3D

class ObjectExtractor:

    def __init__(self, path_label: str):

        self.path_label = path_label

    def extract(self, frame_id: str) -> list[Object3D]:

        objects = []

        with open(file=self.path_label+frame_id+'.txt') as file:
            lines = file.readlines()

        for line in lines:

            data_str = line.strip().split()

            dtype = data_str[0]

            data = [float(d) for d in data_str[1:]]
            
            truncated, occluded, alpha = data[0], int(data[1]), data[2]
            bbox = data[3:7]
            dims = data[7:10]
            loc = data[10:13]
            rotation_y = data[13]

            objects.append(
                Object3D(
                    frame_id=frame_id,
                    type=dtype,
                    truncated=truncated,
                    occluded=occluded,
                    alpha=alpha,
                    bbox_x1=bbox[0], bbox_y1=bbox[1], bbox_x2=bbox[2], bbox_y2=bbox[3],
                    height=dims[0], width=dims[1], length=dims[2],
                    loc_x=loc[0], loc_y=loc[1], loc_z=loc[2],
                    rotation_y=rotation_y
                    )
            )

        return objects
    