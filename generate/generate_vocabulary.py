import json
import sys
import h5py
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
       
def keys(f):
    return [key for key in f.keys()]

def generate_vocabulary(training_questions_path, testing_questions_path, validation_questions_path, output_voc_path, mode="question"):
    """
    Argumentos:
        * paths: path de origen de training, testing y val
        * output_voc_path: dónde se guardará el vocabulario
        * mode = "question" o "answer". Qué vocabulario generar
        
    NOTE: Se espera que los paths (inputs y output) sean jsons
    """
    
    assert mode == "question" or mode == "answer", "Mode must be either 'question' or 'asnwer'"
    
    question_mode = True if "question" else False
    print(f"Generating {mode} dictionary")
    dictionary_set = set()
    
    tokenizer = RegexpTokenizer(r'\w+')
    
    for question_paths in [training_questions_path,
                           testing_questions_path,
                           validation_questions_path]:
    
        #Scan training questions
        questions = load_dict(question_paths)
        question_ids = keys(questions)
        print(f"processing {len(question_ids)} questions")
        pbar = tqdm(total=len(question_ids))
        for question_id in question_ids:
            content = questions[question_id][mode]
            words = tokenizer.tokenize(content)
            if question_mode:
                for word in words:
                    dictionary_set.add(word)
            else:
                dictionary_set.add(content)
            pbar.update()
        pbar.close()

    output = list(dictionary_set)
    print(f"Size dict: {len(output)}")
    
    with open(output_voc_path, "w") as f:
        json.dump(output, f)
    
    print(f"{mode} dictionary generated!")
    return output

def load_dict(dict_path):
    print(f"Loading {dict_path}..")
    with open(dict_path) as f:
        dictionary = json.load(f)
    return dictionary

if __name__ == "__main__":
    training_questions_path = "data/miniGQA2/miniGQA2_question_train.json"
    testing_questions_path = "data/miniGQA2/miniGQA2_question_test.json"
    validation_questions_path = "data/miniGQA2/miniGQA2_question_val.json"
    
    output_question = "data/miniGQA2/miniGQA2_question_vocabulary.json"
    output_answer = "data/miniGQA2/miniGQA2_answer_vocabulary.json"
    
    generate_vocabulary(training_questions_path, testing_questions_path, validation_questions_path, output_answer, question_mode=False)
    generate_vocabulary(training_questions_path, testing_questions_path, validation_questions_path, output_question, question_mode=True)