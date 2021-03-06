import torch
from sklearn.model_selection import train_test_split
import random
import pickle
import time
import json


def save_dict(dictionary):
    with open(saving_path_dict, 'wb') as f:
        pickle.dump(dictionary, f)

def load_dict():
    with open(saving_path_dict, 'rb') as f:
        dictionary = pickle.load(f)

    return dictionary

def get_answer(output, target, vocabulary=None, return_answer=False):
    '''
    :param output: tensor representing output of the model
    :param vocabulary: vocabulary of all the words

    :return correct: average accuracy
    :return answer: if vocabulary is not None, the string representation of the outputs, else None
    '''

    with torch.no_grad():
        idx = torch.argmax(output, dim=1)
        correct = (idx == target).sum().item()
        correct /= float(output.size(0))
            
        if vocabulary is not None:
            # print("elementos en vocabulary: ", len(vocabulary))
            # for id in idx:
            #     print(f"id: {id}")
            #     print(f"id item: {id.item()}")
            answer = [vocabulary[id.item()] if id.item() < len(vocabulary) else "-" for id in idx]

            # answer = []
            # for id in idx:
            #     if id.item() < len(vocabulary)
            #     answer.append
        else:
            answer = None

        if return_answer:
            return correct, answer, (idx == target).tolist()

        return correct, answer, []
    

def split_train_validation(stories, labels, perc_validation=0.2, shuffle=True):
    '''
    :param stories: stories structure already vectorized
    :param labels: list of task ids of the stories

    :return train_stories: 100*(1-perc_validation)% of stories
    :return validation_stories: 100*perc_validation% stories
    '''

    # stratify maintains the same percentage of babi tasks for train and validation
    train_stories, validation_stories = train_test_split(stories, test_size=perc_validation, shuffle=shuffle, stratify=labels)

    return train_stories, validation_stories

def random_idx_gen(start,end):
    indices = list(range(start,end))

    while True:
        random.shuffle(indices)
        for el in indices:
            yield el

def save_models(models, path):
    '''
    :param models: iterable of (models to save, name)
    :param paths: saving path
    '''
    dict_m = {}
    for model, name in models:
        dict_m[name] = model.state_dict()
        print(f"Saved model: {name}!")
    torch.save(dict_m, path)


def emergency_save(models, wasError=True):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if wasError:
        print("Error but saving models first")
        filename = f"saved_models/error_{timestr}.tar"
    else:
        print("Aborting but saving models first")
        filename = f"saved_models/aborted_{timestr}.tar"
    save_models(models, filename)

def load_models(models, path):
    '''
    :param models: iterable of models to save
    :param paths: iterable of saving paths
    '''

    checkpoint = torch.load(path)
    for model, name in models:
        model.load_state_dict(checkpoint[name])
        print(f"modelo {name} cargado!")


def save_training_state(avg_train_losses, avg_train_accuracies, val_losses, val_accuracies, path):
    json_file = {"avg_train_losses": avg_train_losses,
                 "avg_train_accuracies": avg_train_accuracies,
                 "val_losses": val_losses,
                 "val_accuracies": val_accuracies}
    with open(path, "w") as f:
        json.dump(json_file, f)

def load_training_state(path):
    with open(path, 'r') as f:
        json_file = json.load(f)
    return [json_file["avg_train_losses"], json_file["avg_train_accuracies"], json_file["val_losses"], json_file["val_accuracies"]]


saving_path_dict = 'saved_models/dict.data'
saving_path_rn = 'saved_models/rn.tar'
saving_path_rrn = 'saved_models/rrn.tar'
names_models = ['LSTM', 'RN', 'RRN', 'MLP']

files_names_train_en_valid = [
    'qa1_train.txt',
    'qa2_train.txt',
    'qa3_train.txt',
    'qa4_train.txt',
    'qa5_train.txt',
    'qa6_train.txt',
    'qa7_train.txt',
    'qa8_train.txt',
    'qa9_train.txt',
    'qa10_train.txt',
    'qa11_train.txt',
    'qa12_train.txt',
    'qa13_train.txt',
    'qa14_train.txt',
    'qa15_train.txt',
    'qa16_train.txt',
    'qa17_train.txt',
    'qa18_train.txt',
    'qa19_train.txt',
    'qa20_train.txt'
]

files_names_val_en_valid = [
    'qa1_valid.txt',
    'qa2_valid.txt',
    'qa3_valid.txt',
    'qa4_valid.txt',
    'qa5_valid.txt',
    'qa6_valid.txt',
    'qa7_valid.txt',
    'qa8_valid.txt',
    'qa9_valid.txt',
    'qa10_valid.txt',
    'qa11_valid.txt',
    'qa12_valid.txt',
    'qa13_valid.txt',
    'qa14_valid.txt',
    'qa15_valid.txt',
    'qa16_valid.txt',
    'qa17_valid.txt',
    'qa18_valid.txt',
    'qa19_valid.txt',
    'qa20_valid.txt'
]

files_names_test_en_valid = [
    'qa1_test.txt',
    'qa2_test.txt',
    'qa3_test.txt',
    'qa4_test.txt',
    'qa5_test.txt',
    'qa6_test.txt',
    'qa7_test.txt',
    'qa8_test.txt',
    'qa9_test.txt',
    'qa10_test.txt',
    'qa11_test.txt',
    'qa12_test.txt',
    'qa13_test.txt',
    'qa14_test.txt',
    'qa15_test.txt',
    'qa16_test.txt',
    'qa17_test.txt',
    'qa18_test.txt',
    'qa19_test.txt',
    'qa20_test.txt'
]


files_names_test_en = [
    'qa1_single-supporting-fact_test.txt',
    'qa2_two-supporting-facts_test.txt',
    'qa3_three-supporting-facts_test.txt',
    'qa4_two-arg-relations_test.txt',
    'qa5_three-arg-relations_test.txt',
    'qa6_yes-no-questions_test.txt',
    'qa7_counting_test.txt',
    'qa8_lists-sets_test.txt',
    'qa9_simple-negation_test.txt',
    'qa10_indefinite-knowledge_test.txt',
    'qa11_basic-coreference_test.txt',
    'qa12_conjunction_test.txt',
    'qa13_compound-coreference_test.txt',
    'qa14_time-reasoning_test.txt',
    'qa15_basic-deduction_test.txt',
    'qa16_basic-induction_test.txt',
    'qa17_positional-reasoning_test.txt',
    'qa18_size-reasoning_test.txt',
    'qa19_path-finding_test.txt',
    'qa20_agents-motivations_test.txt',
]

files_names_train_en = [
    'qa1_single-supporting-fact_train.txt',
    'qa2_two-supporting-facts_train.txt',
    'qa3_three-supporting-facts_train.txt',
    'qa4_two-arg-relations_train.txt',
    'qa5_three-arg-relations_train.txt',
    'qa6_yes-no-questions_train.txt',
    'qa7_counting_train.txt',
    'qa8_lists-sets_train.txt',
    'qa9_simple-negation_train.txt',
    'qa10_indefinite-knowledge_train.txt',
    'qa11_basic-coreference_train.txt',
    'qa12_conjunction_train.txt',
    'qa13_compound-coreference_train.txt',
    'qa14_time-reasoning_train.txt',
    'qa15_basic-deduction_train.txt',
    'qa16_basic-induction_train.txt',
    'qa17_positional-reasoning_train.txt',
    'qa18_size-reasoning_train.txt',
    'qa19_path-finding_train.txt',
    'qa20_agents-motivations_train.txt'
]
