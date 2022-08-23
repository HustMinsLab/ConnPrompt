# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaForMaskedLM, RobertaTokenizer


class RoBERTa_MLM(nn.Module):
    def __init__(self, args):
        super(RoBERTa_MLM, self).__init__()

        self.RoBERTa_MLM = RobertaForMaskedLM.from_pretrained('roberta-base')
        for param in self.RoBERTa_MLM.parameters():
            param.requires_grad = True
        
        self.vocab_size = args.vocab_size
        self.num_class = args.num_class
        
    def forward(self, arg, mask_arg, token_mask_indices, Token_id, len_comp, len_cont, len_expa, len_temp):
        out_arg = self.RoBERTa_MLM(arg, mask_arg)[0].cuda()  # [batch, arg_len, vocab]
        
        out_vocab = torch.zeros(len(arg), self.vocab_size).cuda()
        for i in range(len(arg)):
            out_vocab[i] = out_arg[i][token_mask_indices[i]]  # [arg_len, vocab]
        
        out_ans = out_vocab[:, Token_id] # Tensor.cuda()

        # Verbalizer
        pred_word = torch.argmax(out_ans, dim=1).tolist() # list

        pred = torch.IntTensor(len(arg), self.num_class).cuda()
        for tid, idx in enumerate(pred_word, 0):
            if idx <= (len_comp - 1):
                pred[tid] = torch.IntTensor([1, 0, 0, 0])
            elif  (len_comp - 1) < idx <= (len_comp + len_cont - 1):
                pred[tid] = torch.IntTensor([0, 1, 0, 0])
            elif  (len_comp + len_cont - 1) < idx <= (len_comp + len_cont + len_expa - 1):
                pred[tid] = torch.IntTensor([0, 0, 1, 0])
            elif  (len_comp + len_cont + len_expa - 1) < idx <= (len_comp + len_cont + len_expa + len_temp - 1):
                pred[tid] = torch.IntTensor([0, 0, 0, 1])
                
        return pred, out_ans
