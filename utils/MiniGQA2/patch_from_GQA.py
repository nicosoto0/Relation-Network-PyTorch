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
    return [index.split(".")[0] for index in indexes]


def has_scene_graph(image_id, scene_graphs):
    if image_id in scene_graphs:
        return True
    else:
        return False
    # try:
    #     scene_graphs[image_id]
    #     return True
    # except KeyError:
    #     return False


def find_patch(gqa_images_ids, miniGqa_images_ids, patch_images_ids, scene_graphs):
    picked = False
    while not picked:
        # index = random.randint(0, len(gqa_images_ids)-1)
        picked_id = random.choice(gqa_images_ids)
        # picked_id = gqa_images_ids[index]
        if picked_id not in patch_images_ids and picked_id not in miniGqa_images_ids:
            if picked_id in scene_graphs:
                picked = True
    return picked_id


if __name__ == "__main__":
    gqa_path = "./gqa"
    id_images_in_miniGQA_path = "./id_images_in_miniGQA.json"
    scene_graphs_train_path = "./train_sceneGraphs.json"
    scene_graphs_val_path = "./val_sceneGraphs.json"
    minigqa_path = "G:/Archivos/Downloads/AI/Relation Net No Git/utils/visualizeObjects/images"
    
    images_output_path = "C:/Users/benjavides/Desktop/miniGQA2"
    gqa2_image_ids_output_path = "./gqa2_image_ids.json"
    gqa2_patch_image_ids_output_path = "./gqa2_patch_image_ids.json"
    id_images_in_miniGQA_with_scene_graph_output_path = "./id_images_in_miniGQA_with_scene_graph.json"

    COPY_IMAGES = True
    random.seed(1)

    with open(id_images_in_miniGQA_path, "r") as f:
        id_images_in_miniGQA = set(json.load(f))
        print(f"imagenes en miniGQA: {len(id_images_in_miniGQA)}")

    with open(scene_graphs_train_path, "r") as f:
        scene_graphs = set(json.load(f).keys())
    
    with open(scene_graphs_val_path, "r") as f:
        scene_graphs_val = set(json.load(f).keys())
        scene_graphs.update(scene_graphs_val)

    gqa_images_ids = get_indexes(gqa_path)

    pbar = tqdm(total=len(id_images_in_miniGQA))
    patch_images_ids = set()
    no_scene_graph_counter = 0
    id_images_in_miniGQA_with_scene_graph = []
    for image_id in id_images_in_miniGQA:
        if not has_scene_graph(image_id, scene_graphs):
            patch = find_patch(
                gqa_images_ids, id_images_in_miniGQA, patch_images_ids, scene_graphs)
            patch_images_ids.add(patch)
            no_scene_graph_counter += 1
            # print(f"no_scene_graph_counter: {no_scene_graph_counter}")
        else:
            id_images_in_miniGQA_with_scene_graph.append(image_id)
        pbar.update()
    pbar.close()

    print(f"imagenes en miniGQA con scene graph: {len(id_images_in_miniGQA_with_scene_graph)}")
    print(f"imagenes en miniGQA sin scene graph: {len(id_images_in_miniGQA) - len(id_images_in_miniGQA_with_scene_graph)}")
    print(f"imagenes parche: {len(patch_images_ids)}")

    with open(gqa2_image_ids_output_path, "w") as f:
        gqa2_image_ids = list(id_images_in_miniGQA_with_scene_graph) + list(patch_images_ids)
        # print(f"gqa2_image_ids: {gqa2_image_ids}")
        json.dump(gqa2_image_ids, f)
    with open(gqa2_patch_image_ids_output_path, "w") as f:
        json.dump(list(patch_images_ids), f)
    with open(id_images_in_miniGQA_with_scene_graph_output_path, "w") as f:
        json.dump(list(id_images_in_miniGQA_with_scene_graph), f)

    if COPY_IMAGES:
        #Copy patch images
        pbar = tqdm(total=len(patch_images_ids))
        for image_id in patch_images_ids:
            file_path = gqa_path + "/" + image_id + ".jpg"
            shutil.copy(file_path, images_output_path)
            pbar.update()
        pbar.close()
        # Copy miniGQA images with scene graph
        pbar = tqdm(total=len(id_images_in_miniGQA_with_scene_graph))
        for image_id in id_images_in_miniGQA_with_scene_graph:
            file_path = minigqa_path + "/" + image_id + ".jpg"
            shutil.copy(file_path, images_output_path)
            pbar.update()
        pbar.close()
