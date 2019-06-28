import json
import h5py
from skimage import io
import os
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy
from random import shuffle


def get_bboxes(image_index, scene_graphs):
    scene_graphs = scene_graphs[image_index]
    print(f"scene graph: {scene_graphs}")
    scene_graphs["objects"]
    print(f"scene graph objects: {}")

    bboxes = h5["bboxes"][h5_index]
    return bboxes[:objects_num]


def convert_points_to_box(points, color, alpha):
    # upper_left_point = (points[0], points[1])
    lower_left_point = (points[0], points[3])
    width = points[2] - points[0]
    height = points[1] - points[3]
    text_pos = (lower_left_point[0]+width/2, lower_left_point[1]+height/2)
    return Rectangle(lower_left_point, width, height, ec=(*color, 1),
                     fc=(*color, alpha)), text_pos


if __name__ == "__main__":
    images_path = "./images"
    scene_graphs_path = "./real_objects/train_sceneGraphs.json"
    id_images_in_miniGQA_path = "./id_images_in_miniGQA.json"
    MAX_OBJECTS = 5

    with open(id_images_in_miniGQA_path, "r") as f:
        id_images_in_miniGQA = json.load(f)

    id_images_in_miniGQA = ["861", "916", "997", "1018", "1042"]  # TODO

    with open(scene_graphs_path) as f:
        scene_graphs = json.load(f)

    shuffle(id_images_in_miniGQA)

    for image_id in id_images_in_miniGQA[:1]:
        bboxes = get_bboxes(image_id, scene_graphs)
        # print(f"bboxes: {bboxes}")
        image = io.imread(os.path.join(images_path, image_id + ".jpg"))
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1 = plt.subplot(121)
        io.imshow(image)
        ax1.autoscale(enable=True)
        ax1.title.set_text(f'{MAX_OBJECTS} objects')
        ax1.set_axis_off()
        for idx, bbox in enumerate(bboxes[:MAX_OBJECTS]):
            # print(f"bbox: {bbox}")
            color = numpy.random.rand(3)
            box, text_pos = convert_points_to_box(
                bbox, color, .4)
            ax1.text(text_pos[0], text_pos[1],
                     f"obj {idx+1}", bbox=dict(facecolor=color, alpha=0.5))
            ax1.add_patch(box)
        ax2 = plt.subplot(122)
        ax2.title.set_text(f'Original image')
        ax2.set_axis_off()
        io.imshow(image)
        plt.show()
