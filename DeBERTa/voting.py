import torch
from sklearn.metrics import f1_score

acc = 0
prompt_1 = torch.load('./DeBERTa_prompt1.pth')
prompt_2 = torch.load('./DeBERTa_prompt2.pth')
prompt_3 = torch.load('./DeBERTa_prompt3.pth')

# choose the best prompt and give it more votes(more than 1 and less than 2)
prompt_vote = prompt_1 + prompt_2*1.5 + prompt_3

truth = torch.load('./test_truth.pth')
_, pred = torch.max(prompt_vote, dim=1)
_, real = torch.max(truth, dim=1)
for i in range(len(prompt_1)):
    if pred[i] == real[i]:
        acc = acc+1

acc_rate = acc / 1474
f1 = f1_score(real, pred,average='macro')

