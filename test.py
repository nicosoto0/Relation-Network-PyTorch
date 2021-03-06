#!/usr/bin/env python
import torch
from utils.generate_dictionary import load_dict

# """
# def test():
#     yield 1,2,3

# batch = test()
# val_1, val_2, val_3 = next(batch)

# print(val_1, val_2, val_3)

# print(["hola" if i < 5 else "chao" for i in range(10)])
# """

# question_v = []

# # <PADDING> Index: len(dictionary_question)
# padding_symbol_Index = 1
# MAX_QUESTION_LENGTH = 4

# questions = ["como estas?", "que día es?", "cuál lapiz es tuyo?"]

# for question in questions:

#     # words = question.split(" ")
#     # q_v = torch.rand(1, MAX_QUESTION_LENGTH)
#     # lst = [0 if i < len(words) else padding_symbol_Index for i in range(MAX_QUESTION_LENGTH)]
#     # torch.cat([lst], out=q_v)
#     # question_v.append(q_v)
    
#     words = question.split(" ")
    
#     q_v = [0 if i < len(words) else padding_symbol_Index for i in range(MAX_QUESTION_LENGTH)]
#     # q_v = [dictionary_question.index(words[i]) for i in range(MAX_QUESTION_LENGTH) if i < len(words) else paddingIndex]
#     question_v.append(q_v)

# print(question_v)

# b = torch.FloatTensor(question_v)

# """
# a = []
# for i in range(100000):
#     a.append(torch.rand(1, 100, 100)

# b = torch.Tensor(100000, 100, 100)
# torch.cat(a, out=b)
# """


# print(b)
# print(b.shape)

# import signal
# import sys
# def signal_handler(sig, frame):
#     print('You pressed Ctrl+C!')
#     sys.exit(0)
    
# def does():
#     while True:
#         pass
# signal.signal(signal.SIGINT, signal_handler)
# print('Press Ctrl+C')
# does()


# from functools import partial
# import win32api

# def signalHandler(sig, par=None):
#     print("Test!!!")
#     par()
#     raise KeyboardInterrupt

# win32api.SetConsoleCtrlHandler(partial(signalHandler,partial(print,"hola")), 1)
# while True:
#     pass


questions_path = "./data/miniGQA/training_question_ids.json"
questions = load_dict(questions_path)
questions_ids = questions.keys()
all_categories = []

print(f"amount of data: {len(questions_ids)}")

group_list = []
type_list =[]

for question_id in questions_ids:
    q_type = questions[question_id]["group"]
    q_group = questions[question_id]["types"]
    #category = {"group": questions[question_id]["group"], "types":questions[question_id]["types"]}
    #all_categories.append(category)

    if not(q_group  in  group_list):
        group_list.append(q_group)

    if not(q_type in type_list):
        type_list.append(q_type)

print(f"len(group_list) {len(group_list) }")
print(f"len(type_list) {len(type_list) }")
print("-- group_list --")
print(group_list)

print("-- type_list --")
print(type_list)


            # # Obtain results for each group and type
            # for question, correct_answer in zip(category_batch, correct_answers):
                
            #     group = question["group"] # e.g. -> all color questions
            #     if group is not None:
            #         group_rights, group_total = groups.get(group, (0, 0))
            #         groups[group] = (group_rights + correct_answer, group_total + 1)
            #     else:
            #         group_rights, group_total = groups.get("None", (0, 0))
            #         groups["None"] = (group_rights + correct_answer, group_total + 1)
                
            #     for typ in question["types"]: # -> e.g. semantic
            #         type_category = question["types"][typ] # -> e.g. query
            #         if type_category is not None:
            #             category_rights, category_total = types[typ].get(type_category, (0, 0))
            #             types[typ][type_category] = (category_rights + correct_answer, category_total + 1)
            # batch_number += 1




                