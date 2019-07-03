import torch

idx    = torch.tensor([1,0,0,0,0,0,1])
target = torch.tensor([0,0,1,0,1,0,1])

correct_answers = (idx == target).tolist()

print(correct_answers)
for ans in correct_answers:
    print(ans)
    print(type(ans))