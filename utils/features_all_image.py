import h5py
import json
import os
import sys
import numpy as np
from tqdm import tqdm
from torch import nn
import torchvision.models as models
from skimage import io, transform
from skimage.color import gray2rgb
from skimage.util import crop, pad     
from matplotlib import pyplot as plt
import  torch


if __name__ == "__main__":

    # --- set device ---
    if torch.cuda.is_available():
        print('Using ', torch.cuda.device_count() ,' GPU(s)')
        mode = 'cuda'
    else:
        print("WARNING: No GPU found. Using CPUs...")
        mode = 'cpu'
    device = torch.device(mode)


    # MiniGQA1.0
    #id_images_in_miniGQA_path = "./data/miniGQA/id_images_in_miniGQA.json"
    #images_miniGQA_path = "G:/Archivos/Downloads/AI/Relation Net No Git/utils/visualizeObjects/images/"

    # MiniGQA patch data
    images_miniGQA_path = "C:/Users/benjavides/Desktop/miniGQA2/"
    id_images_in_miniGQA_path = "H:/MiniGQA2/gqa2_image_ids.json"

    train_graph_path = "./data/scene_graph/train_sceneGraphs.json"
    val_graph_path = "./data/scene_graph/val_sceneGraphs.json"

    save_features_path = "./data/GQA_full_image_features/"

    resnet = models.resnet50().to(device)
    
    print("reading ids...")
    with open(id_images_in_miniGQA_path, "r") as f:
        image_ids = json.load(f)

    print("reading train graph...")    
    with open(train_graph_path, "r") as f:
        train_graph = json.load(f)

    print("reading val graph...")     
    with open(val_graph_path, "r") as f:
        val_graph = json.load(f)

    seen_id = []
    not_seen = []

    # --- GET valid image ids ---
    print("searching for valid ids...")

    #pbar = tqdm(total=len(image_ids))
    #for image_id in image_ids[0:20]:
    for image_id in image_ids:
        if image_id in train_graph:
            seen_id.append(image_id)

        elif image_id in val_graph:
            seen_id.append(image_id)

        else:
            not_seen.append(image_id)
        
        #pbar.update()

    #pbar.close()
    print(f"len(image_ids) ->{len(image_ids)}")
    print(f"len(seen_id) ->{len(seen_id)}")
    print(f"len(seen_id)/len(image_ids) ->{len(seen_id)/len(image_ids)}")

    # --- GET corp bb ---
    print("getting feature of objects of each image ...")
    pbar = tqdm(total=len(seen_id))
    for image_id in seen_id:

        if image_id in train_graph:
            scene = train_graph[image_id]

        else: #image_id in val_graph:
            scene = val_graph[image_id]

        # ---- GET and CROP Image ---
        img_name = images_miniGQA_path + image_id + ".jpg"
        image = io.imread(img_name)
        if len(image.shape) == 2:
            image = gray2rgb(image)
           
        img_resize = transform.resize(image, (224, 224, 3), anti_aliasing=False)
            
        img_tensor = torch.from_numpy(np.array(img_resize)).float().to(device)
        img_tensor = img_tensor.permute(2, 0, 1)
        img_tensor = torch.unsqueeze(img_tensor, 0)
        #print(f"object_tensor.size() -> {object_tensor.size()}")
        
        object_features = resnet(img_tensor)
        #print(f"object_features.size() -> {object_features.size()}")

        copy_tensor = object_features.clone()
        doble_object_features = torch.cat((object_features, copy_tensor))
        #print(f"all_objects.size() -> {doble_object_features.size()}")

        torch.save(doble_object_features,
                   save_features_path + str(image_id) + '.pt')

        # --- Load tensor example ---
        #objects_features2 = torch.load( save_features_path + str(image_id) + '.pt')
        #print(f"objects_features2.size() -> {objects_features2.size()}")

        pbar.update()
    pbar.close()


