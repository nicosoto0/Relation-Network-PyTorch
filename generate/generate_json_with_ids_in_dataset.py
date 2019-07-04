"""
Genera un archivo json con una lista de todos los ids en un dataset.
Correr esto luego de haber generado los archivos de generate_json.py
"""
import json

if __name__ == "__main__":
    train_questions_path = "./data/miniGQA3/miniGQA3_question_train.json"
    test_questions_path = "./data/miniGQA3/miniGQA3_question_test.json"
    validation_questions_path = "./data/miniGQA3/miniGQA3_question_val.json"

    output_path = "./data/miniGQA3/id_images_in_miniGQA3.json"

    with open(train_questions_path, "r") as f:
        questions = json.load(f)
    with open(test_questions_path, "r") as f:
        questions.update(json.load(f))
    with open(validation_questions_path, "r") as f:
        questions.update(json.load(f))
    
    image_ids_in_dataset = set()

    for questions_id in questions:
        quest_dictionary = questions[questions_id]
        image_ids_in_dataset.add(quest_dictionary["imageId"])
    
    with open(output_path, "w") as f:
        json.dump(list(image_ids_in_dataset), f)