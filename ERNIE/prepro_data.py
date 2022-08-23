# -*- coding: utf-8 -*-

import re


"""
conn_label = ['by being', 'on the one hand', 'in response', 'particularly', 'on the whole', 'as of now', 'while', 'although', 'while being', 'still',
               'next', 'specifically', 'if we are', 'if one is to', 'thereby', 'ultimately', 'as a consequence', 'because it was', 'nonetheless', 'with',
               'with the purpose of', 'for one thing', 'despite this', 'accordingly', 'if she is', 'all the while', 'namely', 'eventually', 'more specifically', 'given',
               'separately', 'in the end', 'granted', 'and', 'as a result of being', 'to this end', 'in general', 'though', 'consequently', 'in summary',
               'likewise', 'in sum', 'if one were', 'whereas', 'when', 'it is because', 'with the goal', 'if I am', 'at the time', 'in order to',
               'because of being', 'in being', 'were one to', 'during that time', 'in order for them', 'if you are', 'thus being', 'insofar as', 'now', 'through',
               'for the reason that', 'for that purpose', 'as', 'for example', 'by means of', 'on account of being', 'in comparison', 'further', 'since then', 'thus',
               'since', 'simultaneously', 'furthermore', 'as a result of having', 'conversely', 'on the other hand', 'instead', 'in particular', 'in comparison to the fact', 'as part of that',
               'if one is', 'alternatively', 'besides', 'after', 'soon', 'considering that', 'at the same time', 'nevertheless', "what's more", 'in other words',
               'also', 'first', 'additionally', 'before', 'that is', 'as it turns out', 'but', 'in short', 'even though', 'similarly',
               'with the goal of', 'as a result of', 'in addition', 'in', 'after being', 'in the meantime', 'rather', 'yet', 'thereafter', 'in this case',
               'if we were', 'before that', 'for instance', 'upon', 'inasmuch as', 'however', 'afterwards', 'despite', 'in contrast', 'by contrast',
               'This is because', 'later', 'if it was', 'in addition to being', 'overall', 'given that', 'for', 'therefore', 'after all', 'so that',
               'despite being', 'previously', 'in order to be', 'then', 'otherwise', 'finally', 'or', 'indeed', 'in fact', 'prior to this',
               'if they were', 'more to the point', 'if they are', 'in addition to', 'for the purpose of', 'in order', 'plus', 'moreover', 'as such', 'by comparison',
               'if it is', 'at that time', 'by', 'if', 'for that reason', 'because of that', 'so as', 'if he is', 'after having', 'meanwhile',
               'subsequently', 'earlier', 'on the contrary', 'as a result', 'hence', 'regardless', 'in more detail', 'because', 'incidentally', 'as evidence',
               'generally', 'so', 'if it were', 'because of']
"""

class_label = [['Comparison'], ['Contingency'], ['Expansion'], ['Temporal']]

subtype_label = ['Comparison.Similarity', 'Comparison.Contrast', 'Comparison.Concession+SpeechAct.Arg2-as-denier+SpeechAct', 'Comparison.Concession.Arg2-as-denier', 'Comparison.Concession.Arg1-as-denier', 
                 'Contingency.Purpose.Arg2-as-goal', 'Contingency.Purpose.Arg1-as-goal', 'Contingency.Condition+SpeechAct', 'Contingency.Condition.Arg2-as-cond', 'Contingency.Condition.Arg1-as-cond', 'Contingency.Cause+SpeechAct.Result+SpeechAct', 'Contingency.Cause+SpeechAct.Reason+SpeechAct', 'Contingency.Cause+Belief.Result+Belief', 'Contingency.Cause+Belief.Reason+Belief', 'Contingency.Cause.Result', 'Contingency.Cause.Reason', 
                 'Expansion.Substitution.Arg2-as-subst', 'Expansion.Manner.Arg2-as-manner', 'Expansion.Manner.Arg1-as-manner', 'Expansion.Level-of-detail.Arg2-as-detail', 'Expansion.Level-of-detail.Arg1-as-detail', 'Expansion.Instantiation.Arg2-as-instance', 'Expansion.Instantiation.Arg1-as-instance', 'Expansion.Exception.Arg2-as-excpt', 'Expansion.Exception.Arg1-as-excpt', 'Expansion.Equivalence', 'Expansion.Disjunction', 'Expansion.Conjunction', 
                 'Temporal.Synchronous', 'Temporal.Asynchronous.Succession', 'Temporal.Asynchronous.Precedence']

subtype_label_word = ['similarly', 'but', 'but', 'however', 'although',
                      'for', 'for', 'if', 'if', 'if', 'because', 'so', 'because', 'so', 'because', 'so', 
                      'instead', 'by', 'thereby', 'specifically', 'specifically', 'specifically', 'specifically', 'and', 'and', 'and', 'and', 'and',
                      'simultaneously', 'previously', 'then']

ans_word = ['similarly', 'but', 'however', 'although',
            'for', 'if', 'because', 'so',
            'instead', 'by', 'thereby', 'specifically', 'and',
            'simultaneously', 'previously', 'then']


def prepro_data_train(train_file_list):
    train_idx = []
    train_label = []
    train_arg_1 = []
    train_arg_2 = []
    train_conn = []
    train_label_list = []
    train_label_conn = []

    for line in train_file_list:
        train_idx.append(line[0].strip('\n').split(' '))
        train_label.append(line[4].strip('\n').split(' '))
        line[6] = re.sub(r'[^A-Za-z0-9 ]+', '', line[6])
        line[7] = re.sub(r'[^A-Za-z0-9 ]+', '', line[7])
        train_arg_1.append(line[6].strip('\n'))
        train_arg_2.append(line[7].strip('\n'))
        train_conn.append(line[8].strip('\n'))

    for idx, word in enumerate(train_conn, 0):
        if word not in ans_word:
            subtpye_index = subtype_label.index(train_file_list[idx][9])
            word = subtype_label_word[subtpye_index]
        
        list0 = [0]*16
        list0[ans_word.index(word)] = 1
        train_label_conn.append(list0)

    for cla in train_label:
        if cla == class_label[0]:
            train_label_list.append([1, 0, 0, 0])
        elif cla == class_label[1]:
            train_label_list.append([0, 1, 0, 0])
        elif cla == class_label[2]:
            train_label_list.append([0, 0, 1, 0])
        elif cla == class_label[3]:
            train_label_list.append([0, 0, 0, 1])
        else:
            print('error')

    return train_arg_1, train_arg_2, train_label_list, train_label_conn, train_conn


def prepro_data_dev(dev_file_list):
    dev_idx = []
    dev_label_1 = []
    dev_label_2 = []
    dev_arg_1 = []
    dev_arg_2 = []
    dev_conn_1 = []
    dev_conn_2 = []
    dev_label_list = []
    dev_label_conn_list = []

    for line in dev_file_list:
        dev_idx.append(line[0].strip('\n').split(' '))
        dev_label_1.append(line[4].strip('\n').split(' '))
        dev_label_2.append(line[5].strip('\n').split(' '))
        line[7] = re.sub(r'[^A-Za-z0-9 ]+', '', line[7])
        line[8] = re.sub(r'[^A-Za-z0-9 ]+', '', line[8])
        dev_arg_1.append(line[7].strip('\n'))
        dev_arg_2.append(line[8].strip('\n'))
        dev_conn_1.append(line[9].strip('\n'))
        dev_conn_2.append(line[11].strip('\n'))

    for cla in dev_label_1:
        if cla == class_label[0]:
            dev_label_list.append([1, 0, 0, 0])
        elif cla == class_label[1]:
            dev_label_list.append([0, 1, 0, 0])
        elif cla == class_label[2]:
            dev_label_list.append([0, 0, 1, 0])
        elif cla == class_label[3]:
            dev_label_list.append([0, 0, 0, 1])
        else:
            print('error')

    return dev_arg_1, dev_arg_2, dev_label_list, dev_conn_1


def prepro_data_test(test_file_list):
    test_idx = []
    test_label_1 = []
    test_label_2 = []
    test_arg_1 = []
    test_arg_2 = []
    test_conn_1 = []
    test_conn_2 = []
    test_label_list = []
    test_label_conn_list = []

    for line in test_file_list:
        test_idx.append(line[0].strip('\n').split(' '))
        test_label_1.append(line[4].strip('\n').split(' '))
        test_label_2.append(line[5].strip('\n').split(' '))
        line[7] = re.sub(r'[^A-Za-z0-9 ]+', '', line[7])
        line[8] = re.sub(r'[^A-Za-z0-9 ]+', '', line[8])
        test_arg_1.append(line[7].strip('\n'))
        test_arg_2.append(line[8].strip('\n'))
        test_conn_1.append(line[9].strip('\n'))
        test_conn_2.append(line[11].strip('\n'))

    for cla in test_label_1:
        if cla == class_label[0]:
            test_label_list.append([1, 0, 0, 0])
        elif cla == class_label[1]:
            test_label_list.append([0, 1, 0, 0])
        elif cla == class_label[2]:
            test_label_list.append([0, 0, 1, 0])
        elif cla == class_label[3]:
            test_label_list.append([0, 0, 0, 1])
        else:
            print('error')

    return test_arg_1, test_arg_2, test_label_list, test_conn_1
