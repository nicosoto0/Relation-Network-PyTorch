
import json

if __name__ == "__main__":

    questions_path = "C:/Users/benjavides/Desktop/Relation-Network-PyTorch/data/miniGQA/testing_question_ids.json"

    with open(questions_path, "r") as f:
        questions = json.load(f)
    
    yes=0
    no=0
    other=0

    for image_id in questions:
        question_dict = questions[image_id]
        answer = question_dict["answer"]
        if answer == "yes":
            yes+=1
        elif answer == "no":
            no+=1
        else:
            other+=1
    
    total = yes+no+other
    print(f"yes:{yes} no:{no} other:{other}")
    print(f"yes:{yes*100/total} no:{no*100/total} other:{other*100/total}")
#yes:41.828 no:56.558 other:28.898