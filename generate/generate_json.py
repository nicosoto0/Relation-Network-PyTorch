# Imports
import json
import random
from os.path import exists
from os import mkdir
from tqdm import tqdm 


def generate_json_miniGQA(train_fraction=0.7, test_fraction=0.2, val_fraction=0.1, include=["train","test","val"]):
    """
    Argumentos:
        * fractions: fracciones de train, test y val. Deben sumar 1, aunque val se deduce
        * include: qué jsons generar. Útil cuando hay problemas de memoria.
    """
    
    s = train_fraction + test_fraction + val_fraction
    assert 1.0 - s <= 0.000001, f"fractions should add up to 1, got {s}"
    
    SEED = 1
    random.seed(SEED)
    
    json_path = "utils/MiniGQA2/jsons"
    miniGQA2_folder = "data/miniGQA2"
    
    train_path = "miniGQA2_question_train.json"
    test_path     = "miniGQA2_question_test.json"
    val_path      = "miniGQA2_question_val.json"
    
    if not exists(miniGQA2_folder):
        mkdir(miniGQA2_folder)
    
    images_from_miniGQA1_filename = "id_images_in_miniGQA_with_scene_graph.json"
    images_from_GQA_filename = "gqa2_patch_image_ids.json"
    GQA_questions_filepath = "data/GQA/questions1.2"
    
    print(f"Reading {images_from_miniGQA1_filename}...")
    with open(f"{json_path}/{images_from_miniGQA1_filename}") as f:
        miniGQA = json.load(f)
    print(f"Reading {images_from_GQA_filename}...")    
    with open(f"{json_path}/{images_from_GQA_filename}") as f:
        from_GQA = json.load(f)      

    mini_train_size = int(len(miniGQA) * train_fraction)
    mini_test_size = int(len(miniGQA) * test_fraction)
    gqa_train_size = int(len(from_GQA) * train_fraction)
    gqa_test_size = int(len(from_GQA) * test_fraction)
    
    random.shuffle(miniGQA)
    random.shuffle(from_GQA)  
    
    print("Generating sets...")
    
    miniGQAs, from_GQAs, miniGQA2s, paths = [], [], [], []
    if "train" in include:
        miniGQA_train, from_GQA_train = set(miniGQA[:mini_train_size]), set(from_GQA[:gqa_train_size])
        miniGQAs.append(miniGQA_train)
        from_GQAs.append(from_GQA_train)
        miniGQA2s.append({})
        paths.append(train_path)
        
    if "test" in include:
        miniGQA_test,  from_GQA_test  = set(miniGQA[mini_train_size:mini_train_size + mini_test_size]), set(from_GQA[gqa_train_size: gqa_train_size + gqa_test_size])
        miniGQAs.append(miniGQA_test)
        from_GQAs.append(from_GQA_test)
        miniGQA2s.append({})
        paths.append(test_path)                
        
    if "val" in include:
        miniGQA_val,   from_GQA_val   = set(miniGQA[mini_train_size + mini_test_size:]), set(from_GQA[gqa_train_size + gqa_test_size:])
        miniGQAs.append(miniGQA_val)
        from_GQAs.append(from_GQA_val)
        miniGQA2s.append({})   
        paths.append(val_path)
             
    del miniGQA
    del from_GQA
    
    for source in ["val_all_questions.json", "test_all_questions.json",
                   "train_all_questions/train_all_questions_0.json",
                   "train_all_questions/train_all_questions_1.json",
                   "train_all_questions/train_all_questions_2.json",
                   "train_all_questions/train_all_questions_3.json",
                   "train_all_questions/train_all_questions_4.json",
                   "train_all_questions/train_all_questions_5.json",
                   "train_all_questions/train_all_questions_6.json",
                   "train_all_questions/train_all_questions_7.json",
                   "train_all_questions/train_all_questions_8.json",
                   "train_all_questions/train_all_questions_9.json"
                   ]:
        
        with open(f"{GQA_questions_filepath}/{source}", 'r') as f:
            GQA = json.load(f)
        
        print(f"processing {source}...")
        pbar = tqdm(total=len(GQA))
        for question_id in GQA:
            image_id = GQA[question_id]["imageId"]
    
            for miniGQA_ids, from_GQA_ids, miniGQA2 in zip(miniGQAs, from_GQAs, miniGQA2s):
                
                    for from_miniGQA, ids in [
                        (True, miniGQA_ids),
                        (False, from_GQA_ids)]:
                    
                        if image_id in ids:
                            question = GQA[question_id]
                        
                            miniGQA2[question_id] = {"question":   question["question"],
                                                    "answer":      question["answer"],
                                                    "imageId":     question["imageId"],
                                                    "group":       question["groups"]["global"],
                                                    "types":       question ["types"],
                                                    "fromMiniGQA": from_miniGQA}
            pbar.update()
        pbar.close()
        
        del GQA
    
    for miniGQA2, filename in zip(miniGQA2s, paths):
        
        print(f"saving {len(miniGQA2)} questions")
                          
        with open(f"{miniGQA2_folder}/{filename}", 'w') as f:
            json.dump(miniGQA2, f)    
            
if __name__ == "__main__":
    generate_json_miniGQA()
            
            
                
            
    