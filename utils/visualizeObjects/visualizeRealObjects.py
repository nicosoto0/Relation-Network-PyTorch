"""
Permite visualizar las bounding boxes que se definen en el scene-graph.
"""
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
    bboxes = []
    names = []
    for object_id in scene_graphs['objects']:
        # print(f"object: {scene_graphs['objects'][object_id]}")
        x = scene_graphs['objects'][object_id]["x"]
        y = scene_graphs['objects'][object_id]["y"]
        w = scene_graphs['objects'][object_id]["w"]
        h = scene_graphs['objects'][object_id]["h"]
        names.append(scene_graphs['objects'][object_id]["name"])
        bbox = (x, y, w, h)
        bboxes.append(bbox)
        # bboxes = h5["bboxes"][h5_index]
    return bboxes, names


def convert_points_to_box(points, color, alpha):
    # upper_left_point = (points[0], points[1])
    lower_left_point = (points[0], points[1])
    width = points[2]
    height = points[3]
    text_pos = (lower_left_point[0]+width/2, lower_left_point[1]+height/2)
    return Rectangle(lower_left_point, width, height, ec=(*color, 1),
                     fc=(*color, alpha)), text_pos


if __name__ == "__main__":
    images_path = "C:/Users/benjavides/Desktop/miniGQA2"
    scene_graphs_train_path = "C:/Users/benjavides/Desktop/Relation-Network-PyTorch/data/scene_graph/train_sceneGraphs.json"
    scene_graphs_val_path = "C:/Users/benjavides/Desktop/Relation-Network-PyTorch/data/scene_graph/val_sceneGraphs.json"
    id_images_in_miniGQA2_path = "./gqa2_image_ids.json"
    MAX_OBJECTS = 24

    with open(id_images_in_miniGQA2_path, "r") as f:
        id_images_in_miniGQA = json.load(f)

    with open(scene_graphs_train_path, "r") as f:
        scene_graphs = json.load(f)

    with open(scene_graphs_val_path, "r") as f:
        scene_graphs_val = json.load(f)
        scene_graphs.update(scene_graphs_val)

    shuffle(id_images_in_miniGQA)

    for image_id in id_images_in_miniGQA:
        bboxes, names = get_bboxes(image_id, scene_graphs)
        # print(f"bboxes: {bboxes}")
        image = io.imread(os.path.join(images_path, image_id + ".jpg"))
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1 = plt.subplot(121)
        io.imshow(image)
        ax1.autoscale(enable=True)
        ax1.set_axis_off()
        print(f"Mostrando imagen: {image_id}.jpg")
        for idx, bbox in enumerate(bboxes[:MAX_OBJECTS]):
            # print(f"bbox: {bbox}")
            ax1.title.set_text(f'{min(len(bboxes), MAX_OBJECTS)} objects')
            # fig.suptitle(f'{image_id}.jpg', fontsize=16)
            color = numpy.random.rand(3)
            box, text_pos = convert_points_to_box(
                bbox, color, .4)
            ax1.text(text_pos[0], text_pos[1],
                     f"{idx+1} {names[idx]}", bbox=dict(facecolor=color, alpha=0.5))
            ax1.add_patch(box)
        ax2 = plt.subplot(122)
        ax2.title.set_text(f'Original image')
        ax2.set_axis_off()
        io.imshow(image)
        plt.show()
