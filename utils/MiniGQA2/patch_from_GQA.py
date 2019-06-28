"""
Recorre los id's de miniGQA, si una imagen no tiene sceneGrapgh, entonces busca otra que si tenga en GQA,
luego mueve esta imagen parche a images_output_path
"""
import json
import shutil
import os
import random
from tqdm import tqdm


def get_indexes(images_path):
    indexes = os.listdir(images_path)
    return {index.split(".")[0] for index in indexes}


def has_scene_graph(image_id, scene_graphs):
    try:
        scene_graphs[image_id]
        return True
    except KeyError:
        return False


def find_patch(gqa_images_ids, miniGqa_images_ids, patch_images_ids):
    picked = False
    while not picked:
        picked_id = random.choice(gqa_images_ids)
        if picked_id not in patch_images_ids and picked_id not in miniGqa_images_ids:
            picked = True
    return picked_id


if __name__ == "__main__":
    gqa_path = "./gqa"
    id_images_in_miniGQA_path = "./id_images_in_miniGQA.json"
    scene_graphs_path = "./real_objects/train_sceneGraphs.json"
    images_output_path = "./images"
    gqa2_image_ids_output_path = "./gqa2_image_ids.json"

    COPY_IMAGES = True

    with open(id_images_in_miniGQA_path, "r") as f:
        id_images_in_miniGQA = set(json.load(f))

    with open(scene_graphs_path) as f:
        scene_graphs = json.load(f)

    gqa_images_ids = get_indexes(gqa_path)

    pbar = tqdm(total=len(id_images_in_miniGQA))
    patch_images_ids = set()
    for image_id in id_images_in_miniGQA:
        if not has_scene_graph(image_id, scene_graphs):
            patch = find_patch(
                gqa_images_ids, id_images_in_miniGQA, patch_images_ids)
            patch_images_ids.add(patch)
        pbar.update()
    pbar.close()

    with open(gqa2_image_ids_output_path) as f:
        gqa2_image_ids = list(id_images_in_miniGQA).extend(
            list(patch_images_ids))
        json.dump(gqa2_image_ids, f)

    if COPY_IMAGES:
        pbar = tqdm(total=len(patch_images_ids))
        for image_id in patch_images_ids:
            file_path = gqa_path + "/" + image_id + ".jpg"
            shutil.copy(file_path, images_output_path)
            pbar.update()
        pbar.close()
