# -*- coding: utf-8 -*-

"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
"""


import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import f1_score
from load_data import load_data
from prepro_data import prepro_data_train, prepro_data_dev, prepro_data_test
from transformers import DebertaTokenizer, AdamW, get_linear_schedule_with_warmup
from parameter import parse_args

from model import DeBERTa_MLM

torch.cuda.empty_cache()
args = parse_args()  # load parameters


# set seed for random number
def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


setup_seed(args.seed)

# load DeBERTa model
model_name = 'microsoft/deberta-base'
tokenizer = DebertaTokenizer.from_pretrained(model_name)

# load data tsv file
train_data, dev_data, test_data = load_data()

# get arg_1 arg_2 label from data
train_arg_1, train_arg_2, train_label, train_conn_label, train_conn = prepro_data_train(train_data)
dev_arg_1, dev_arg_2, dev_label, dev_conn = prepro_data_dev(dev_data)
test_arg_1, test_arg_2, test_label, test_conn = prepro_data_test(test_data)


label_conn = torch.LongTensor(train_conn_label)
label_tr = torch.LongTensor(train_label)
label_de = torch.LongTensor(dev_label)
label_te = torch.LongTensor(test_label)
print('Data loaded')

Comp = ['similarly', 'but', 'however', 'although']
Cont = ['for', 'if', 'because', 'so']
Expa = ['instead', 'by', 'thereby', 'specifically', 'and']
Temp = ['simultaneously', 'previously', 'then']  

len_comp = len(Comp)
len_cont = len(Cont)
len_expa = len(Expa)
len_temp = len(Temp)

# corresponding ids of the above connectives when tokenizing
Token_id = [11401, 53, 959, 1712,
            13, 114, 142, 98,
            1386, 30, 12679, 4010, 8,
            11586, 1433, 172]

# to limit the length of argument_1
def arg_1_prepro(arg_1):
    arg_1_new = []
    for each_string in arg_1:
        encode_dict = tokenizer.encode_plus(
            each_string,
            add_special_tokens=False,
            padding='max_length',
            max_length=args.arg1_len,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=False,
            return_tensors='pt')
        decode_input = tokenizer.decode(encode_dict['input_ids'][0]).replace('[PAD]', '')
        arg_1_new.append(decode_input)
    return arg_1_new


train_arg_1 = arg_1_prepro(train_arg_1)
dev_arg_1 = arg_1_prepro(dev_arg_1)
test_arg_1 = arg_1_prepro(test_arg_1)


def get_batch(text_data1, text_data2, indices):
    indices_ids = []
    indices_mask = []
    mask_indices = []  # the place of '[MASK]' in 'input_ids'

    for idx in indices:
        encode_dict = tokenizer.encode_plus(
            text_data1[idx] + ' [MASK] ' + text_data2[idx],                 # Prompt 1
            # text_data1[idx] + '[SEP] ' + ' [MASK] ' + text_data2[idx],    # Prompt 2
            # ' [MASK] ' + text_data1[idx] + '[SEP] ' + text_data2[idx],    # Prompt 3
            add_special_tokens=True,
            padding='max_length',
            max_length=args.len_arg,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt')
        indices_ids.append(encode_dict['input_ids'])
        indices_mask.append(encode_dict['attention_mask'])
        try: mask_indices.append(np.argwhere(np.array(encode_dict['input_ids']) == 50264)[0][1])  # id of [MASK] is 50264
        except IndexError:
            print(encode_dict['input_ids'])
            print(np.argwhere(np.array(encode_dict['input_ids']) == 50264))

    batch_ids = torch.cat(indices_ids, dim=0)
    batch_mask = torch.cat(indices_mask, dim=0)

    return batch_ids, batch_mask, mask_indices


# ---------- network ----------
net = DeBERTa_MLM(args).cuda()

# AdamW
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.wd},
    {'params': [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.lr)

criterion = nn.CrossEntropyLoss().cuda()

# creat file to save model and result
file_out = open('./' + args.file_out + '.txt', "w")


print('epoch_num:', args.num_epoch)
print('epoch_num:', args.num_epoch, file=file_out)
print('wd:', args.wd)
print('wd:', args.wd, file=file_out)
print('initial_lr:', args.lr)
print('initial_lr:', args.lr, file=file_out)


##################################  epoch  #################################
for epoch in range(args.num_epoch):
    print('Epoch: ', epoch + 1)
    print('Epoch: ', epoch + 1, file=file_out)
    all_indices = torch.randperm(args.train_size).split(args.batch_size)
    loss_epoch = 0.0
    acc = 0.0
    f1_pred = torch.IntTensor([]).cuda()
    f1_truth = torch.IntTensor([]).cuda()
    start = time.time()

    print('lr:', optimizer.state_dict()['param_groups'][0]['lr'])
    print('lr:', optimizer.state_dict()['param_groups'][0]['lr'], file=file_out)

    ############################################################################
    ##################################  train  #################################
    ############################################################################
    net.train()
    for i, batch_indices in enumerate(all_indices, 1):
        # get a batch of wordvecs
        batch_arg, mask_arg, token_mask_indices = get_batch(train_arg_1, train_arg_2, batch_indices)

        batch_arg = batch_arg.cuda()

        mask_arg = mask_arg.cuda()

        y = Variable(label_tr[batch_indices]).cuda()
        y_conn = label_conn[batch_indices].cuda()

        # fed data into network
        out_sense, out_ans = net(batch_arg, mask_arg, token_mask_indices, Token_id, len_comp, len_cont, len_expa, len_temp)
        
        _, pred_ans = torch.max(out_ans, dim=1)
        _, truth_ans = torch.max(y_conn, dim=1)
        
        _, pred = torch.max(out_sense, dim=1)
        _, truth = torch.max(y, dim=1)
        
        num_correct = (pred == truth).sum()
        acc += num_correct.item()
        f1_pred = torch.cat((f1_pred, pred), 0)
        f1_truth = torch.cat((f1_truth, truth), 0)

        # loss
        loss = criterion(out_ans, truth_ans)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # report
        loss_epoch += loss.item()
        if i % (3000 // args.batch_size) == 0:
            print('loss={:.4f}, acc={:.4f}, F1_score={:.4f}'.format(loss_epoch / (3000 // args.batch_size), acc / 3000,
                                                                    f1_score(f1_truth.cpu(), f1_pred.cpu(),
                                                                              average='macro')), file=file_out)
            print('loss={:.4f}, acc={:.4f}, F1_score={:.4f}'.format(loss_epoch / (3000 // args.batch_size), acc / 3000,
                                                                    f1_score(f1_truth.cpu(), f1_pred.cpu(),
                                                                              average='macro')))
            loss_epoch = 0.0
            acc = 0.0
            f1_pred = torch.IntTensor([]).cuda()
            f1_truth = torch.IntTensor([]).cuda()
    end = time.time()
    print('Training Time: {:.2f}s'.format(end - start))

    ############################################################################
    ##################################  dev  ###################################
    ############################################################################
    all_indices = torch.randperm(args.dev_size).split(args.batch_size)
    loss_epoch = []
    acc = 0.0
    f1_pred = torch.IntTensor([]).cuda()
    f1_truth = torch.IntTensor([]).cuda()

    net.eval()
    for batch_indices in all_indices:
        # get a batch of wordvecs
        batch_arg, mask_arg, token_mask_indices = get_batch(dev_arg_1, dev_arg_2, batch_indices)

        batch_arg = batch_arg.cuda()

        mask_arg = mask_arg.cuda()

        y = Variable(label_de[batch_indices]).cuda()

        # fed data into network
        out_sense, out_ans = net(batch_arg, mask_arg, token_mask_indices, Token_id, len_comp, len_cont, len_expa, len_temp)
        _, pred = torch.max(out_sense, dim=1)
        _, truth = torch.max(y, dim=1)
        num_correct = (pred == truth).sum()
        acc += num_correct.item()
        f1_pred = torch.cat((f1_pred, pred), 0)
        f1_truth = torch.cat((f1_truth, truth), 0)

    # report
    print('Dev Acc={:.4f}, Dev F1_score={:.4f}'.format(acc / args.dev_size, f1_score(f1_truth.cpu(), f1_pred.cpu(),
                                                                                     average='macro')), file=file_out)
    print('Dev Acc={:.4f}, Dev F1_score={:.4f}'.format(acc / args.dev_size,  f1_score(f1_truth.cpu(), f1_pred.cpu(),
                                                                                      average='macro')))

    ############################################################################
    ##################################  test  ##################################
    ############################################################################
    all_indices = torch.randperm(args.test_size).split(args.batch_size)
    loss_epoch = []
    acc = 0.0
    f1_pred = torch.IntTensor([]).cuda()
    f1_truth = torch.IntTensor([]).cuda()
    net.eval()

    # Just for Multi-Prompt case
    '''
    test_pred = torch.zeros(1474, 4)
    test_truth = torch.zeros(1474, 4)
    '''

    for batch_indices in all_indices:
        # get a batch of wordvecs
        batch_arg, mask_arg, token_mask_indices = get_batch(test_arg_1, test_arg_2, batch_indices)

        batch_arg = batch_arg.cuda()

        mask_arg = mask_arg.cuda()

        y = Variable(label_te[batch_indices]).cuda()
        
        # fed data into network
        out_sense, out_ans = net(batch_arg, mask_arg, token_mask_indices, Token_id, len_comp, len_cont, len_expa, len_temp)

        # Just for Multi-Prompt case
        # choose the appropriate Prompt(1 2 3) and other parameters
        # save outputs of the model
        # all test_truth are the same
        '''
        # -------------------------
        for i in range(len(batch_indices)):
            test_pred[batch_indices[i]] = out_sense[i]
            test_truth[batch_indices[i]] = y[i]
        if epoch == 5:                        # epoch(variable) + 1 = epoch(real)
            torch.save(test_pred, './BERT_prompt1.pth')
            # torch.save(test_truth, './test_truth.pth')   # once is ok
        # ------------------------
        '''

        _, pred = torch.max(out_sense, dim=1)
        _, truth = torch.max(y, dim=1)
        num_correct = (pred == truth).sum()
        acc += num_correct.item()
        f1_pred = torch.cat((f1_pred, pred), 0)
        f1_truth = torch.cat((f1_truth, truth), 0)

    # report
    print('Test Acc={:.4f}, Test F1_score={:.4f}'.format(acc / args.test_size, f1_score(f1_truth.cpu(), f1_pred.cpu(),
                                                                                    average='macro')), file=file_out)
    print('Test Acc={:.4f}, Test F1_score={:.4f}'.format(acc / args.test_size, f1_score(f1_truth.cpu(), f1_pred.cpu(),
                                                                                    average='macro')))

file_out.close()
