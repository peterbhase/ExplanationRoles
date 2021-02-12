import argparse
import numpy as np
import pandas as pd
import torch
import copy
import os
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, RobertaTokenizer, T5Tokenizer
import matplotlib.pyplot as plt
import faiss
import time
import json
from write_synthetic_data import write_synthetic_data
from models import T5Wrapper, DistilBertForSequenceClassification, BertForSequenceClassification, RobertaForSequenceClassification, RobertaNeuralProcess, RobertaModel, AlbertForSequenceClassification
from models.modeling_optimal_synthetic_model import OptimalSyntheticModel

'''
general utilities
'''

def make_experiment_sheet(experiment, params, num_seeds):
    sheet_folder = 'result_sheets'
    if not os.path.exists(sheet_folder): 
        os.mkdir(sheet_folder)
    sheet_path = os.path.join(sheet_folder, experiment + '.csv')
    cols = {param : [] for param in params}
    data = pd.DataFrame(cols)
    if not os.path.exists(sheet_path):
        data.to_csv(sheet_path, index=False)

def write_experiment_result(experiment, params):
    sheet_path = os.path.join('result_sheets', experiment + '.csv')
    data = pd.read_csv(sheet_path)
    dev_acc = round(np.load('tmp_dev_acc.npy').item(),3)
    test_acc = round(np.load('tmp_test_acc.npy').item(),3)
    params.update({f'dev_acc' : dev_acc})
    params.update({f'test_acc' : test_acc})
    new_data = data.append(params, ignore_index=True)
    new_data.to_csv(sheet_path, index=False)

def balanced_array(size, prop):
    # make array of 0s and 1s of len=size and mean ~= prop
    array = np.zeros(size)
    where_ones = np.random.choice(np.arange(size), size=int(size*prop), replace=False)
    array[where_ones] = 1
    if size==1:
        print("Asking for balanced array of size 1 could easily result in unintended behavior. This defaults to all 0s", end='\r')
    return array

def str2bool(v):
    # used for boolean argparse values
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def safe_decode(tokenizer, sequence, skip_special_tokens=True):
    return tokenizer.decode(filter(lambda x : x>=0, sequence), skip_special_tokens=skip_special_tokens)

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def _pad_seq(seq, length, pad_id):
    assert not len(seq) > length, "seq already too long"
    seq += [pad_id] * (length-len(seq))

def shuffle_lists(lists):
    [np.random.shuffle(_iter) for _iter in lists]

def dataloaders_equal(dataloader1, dataloader2):
    if len(dataloader1) != len(dataloader2): 
        return False
    for batch1, batch2 in zip(dataloader1, dataloader2):
        for item1, item2 in zip(batch1, batch2):
            if not torch.all(item1 == item2):
                print(item1)
                print(item2)
                return False
    return True

def decompose_integers(integers, num_parts, off_elements_zero=False):
    # if not off_elements_zero: randomly decomposes an array of integers (shape: length) into an array (shape: num_parts x length) that sums across dim0 to integers
    # if off_elements_zero: make an array(shape: num_parts x length) of zeros, then distribute integers into each row s.t. each row is 1-sparse and all integers are represented
    if not off_elements_zero:
        idx_to_random_sizes = {
            idx : [len(sub_array) for sub_array in np.split(np.arange(integers[idx]), sorted(np.random.randint(0, integers[idx], size=num_parts-1)))]
            for idx in range(len(integers))
        }
        decomposed_ints = np.concatenate([np.array(idx_to_random_sizes[idx]).reshape(-1,1) for idx in range(len(integers))], axis=1)
    if off_elements_zero:
        assert num_parts in [2,4]
        if num_parts==2:
            mr_idx = np.array([0,2])
            nd_idx = np.array([1,3])
            both_idx = [mr_idx, nd_idx]
            decomposed_ints = np.zeros((num_parts, len(integers)))
            decomposed_ints[0,mr_idx] = integers[mr_idx]
            decomposed_ints[1,nd_idx] = integers[nd_idx]
        elif num_parts==4:
            decomposed_ints = np.zeros((num_parts, len(integers)))
            for i in range(len(integers)):
                decomposed_ints[i,i] = integers[i]
    return decomposed_ints

def add_noise_to_model(sd, model):
    for n,p in model.named_parameters():
        d=np.prod(p.shape)
        noise=torch.normal(mean=torch.zeros(d), std=sd)
        noise = noise.to(p.device).reshape(p.shape)
        p.data.add_(noise)

def rectify_mismatched_embeddings(model, state_dict, tokenizer):
    # hack fix for when tokenizer vocab size doesnt match model vocab size
    if hasattr(model, 'roberta'):
        k = 'roberta.embeddings.word_embeddings.weight'
        module = model.roberta
    elif hasattr(model, 'embeddings'):
        k = 'embeddings.word_embeddings.weight'
        module = model
    else:
        assert False, "not accessing the right embedding in rectify_mismatched_embeddings"
    keys = [k]
    model_vocab_size = module.embeddings.word_embeddings.weight.size(0)
    for k in keys:
        v = state_dict[k]
        v = v[:model_vocab_size,:]
        state_dict[k] = v
    model_vocab_size = module.embeddings.word_embeddings.weight.size(0)
    print(f"Forced new vocab size to {model_vocab_size}")

'''
get experiment name for writing files
'''
def get_experiment_name(args):
    conditioning = 'ELV' if args.use_ELV else ('cat' if args.use_textcat else 'sOnly')
    retrieval_model = os.path.split(args.retrieval_model)[-1]
    single_task = str(args.single_task)[0]
    optimal_retrieval = str(args.use_optimal_retrieval)[0]
    use_retrieval = str(args.use_retrieval)[0]
    give_idx = str(args.use_index)[0]
    mn = str(args.use_mn_indicator)[0]
    if args.use_percent_data < 1:
        n_train = str(args.use_percent_data)
    elif args.use_num_train > 0:
        n_train = str(args.use_num_train)
    elif args.data_name == 'synthetic':
        n_train = str(args.num_train_synthetic)
    else:
        n_train = 'full'
    epochs = args.num_train_epochs
    batch_size = args.train_batch_size * args.grad_accumulation_factor
    relevant_points = args.num_relevant_points if args.num_relevant_points >= 0 else args.context_size + 1
    pretrained_retriever = str(not args.reinitialize_retriever)[0]
    train_retriever = str(args.train_retriever)[0]
    rebuild = str(100*args.rebuild_index_every_perc) if args.rebuild_index_every_perc > 0 else str(args.rebuild_index_every_epoch)[0]
    e_name = f"{args.data_name}_{args.model[:10]}_{retrieval_model[:10]}_e{epochs}b{batch_size}_n{n_train}_{conditioning}_" \
             f"r{relevant_points}_c{args.context_size}_k{args.top_k}" \
             f"_rt{use_retrieval}_tr{train_retriever}_rb{rebuild}" \
             f"_c{args.context_includes}_sd{args.seed}"
    if args.data_name == 'synthetic':
        num_tasks = args.num_train_synthetic // args.num_relevant_points if args.num_tasks < 0 else args.num_tasks
        e_name += f"_opt{optimal_retrieval}_Rprt{pretrained_retriever}_E{args.explanation_kind[:4]}_idx{give_idx}{args.idx_function[:4]}Mn{mn}Sg{single_task}_nTasks{num_tasks}"
    return e_name

'''
dictionaries/functions for mapping related to models/datasets
'''

def get_file_names(data_name):
    if data_name == 'synthetic':
        names = ['train.csv', 'dev.csv', 'test.csv']
    if 'esnli' in data_name.lower():
        names = ['train.csv', 'dev.csv', 'test.csv']
    if data_name == 'semeval':
        names = ['semeval_train_data.json', 'semeval_valid_data.json', 'semeval_test_data.json']
    if data_name == 'tacred':
        names = ['train.json', 'dev.json', 'test.json']
    return names

def get_custom_model_class(name):
    if 't5' in name: 
        model_class = T5Wrapper
    elif 'roberta' in name:
        model_class = RobertaForSequenceClassification
    elif 'albert' in name:
        model_class = AlbertForSequenceClassification
    elif 'distilbert' in name:
        model_class = DistilBertForSequenceClassification
    elif 'bert' in name:
        model_class = BertForSequenceClassification
    elif 'optimal_synthetic' in name:
        model_class = OptimalSyntheticModel
    return model_class

data_name_to_num_labels = {
    'eSNLI' : 3,
    'eSNLI_full' : 3,
    'semeval' : 19,
    'tacred' : 42,
    'synthetic' : 2,
}

tacred_label_to_id = {'no_relation': 0, 'per:title': 1, 'org:top_members/employees': 2, 'per:employee_of': 3, 'org:alternate_names': 4, 'org:country_of_headquarters': 5, 'per:countries_of_residence': 6, 'org:city_of_headquarters': 7, 'per:cities_of_residence': 8, 'per:age': 9, 'per:stateorprovinces_of_residence': 10, 'per:origin': 11, 'org:subsidiaries': 12, 'org:parents': 13, 'per:spouse': 14, 'org:stateorprovince_of_headquarters': 15, 'per:children': 16, 'per:other_family': 17, 'per:alternate_names': 18, 'org:members': 19, 'per:siblings': 20, 'per:schools_attended': 21, 'per:parents': 22, 'per:date_of_death': 23, 'org:member_of': 24, 'org:founded_by': 25, 'org:website': 26, 'per:cause_of_death': 27, 'org:political/religious_affiliation': 28, 'org:founded': 29, 'per:city_of_death': 30, 'org:shareholders': 31, 'org:number_of_employees/members': 32, 'per:date_of_birth': 33, 'per:city_of_birth': 34, 'per:charges': 35, 'per:stateorprovince_of_death': 36, 'per:religion': 37, 'per:stateorprovince_of_birth': 38, 'per:country_of_birth': 39, 'org:dissolved': 40, 'per:country_of_death': 41}
tacred_id_to_label = {v:k for k,v in tacred_label_to_id.items()}
semeval_label_to_id = {"no_relation": 0,  "Cause-Effect(e1,e2)": 1,  "Cause-Effect(e2,e1)": 2,  "Component-Whole(e1,e2)": 3,  "Component-Whole(e2,e1)": 4,  "Content-Container(e1,e2)": 5,  "Content-Container(e2,e1)": 6,  "Entity-Destination(e1,e2)": 7,  "Entity-Destination(e2,e1)": 8,  "Entity-Origin(e1,e2)": 9,  "Entity-Origin(e2,e1)": 10,  "Instrument-Agency(e1,e2)": 11,  "Instrument-Agency(e2,e1)": 12,  "Member-Collection(e1,e2)": 13,  "Member-Collection(e2,e1)": 14,  "Message-Topic(e1,e2)": 15,  "Message-Topic(e2,e1)": 16,  "Product-Producer(e1,e2)": 17,  "Product-Producer(e2,e1)": 18}
semeval_id_to_label = {v:k for k,v in semeval_label_to_id.items()}

def get_make_data_function(name):
    data_name_to_make_data_function = {
        'eSNLI' : make_SNLI_data,
        'eSNLI_full' : make_SNLI_data,
        'semeval' : make_RE_data,
        'tacred' : make_RE_data,
        'synthetic' : make_synthetic_data,
    }
    return data_name_to_make_data_function[name]

'''
SYNTHETIC DATA
'''

def make_synthetic_data(args, retriever, tokenizer, retriever_tokenizer, file_path):
    # reads the synthetic data
    # args.small_data handled in write_synthetic_data
    split_name = os.path.split(file_path)[-1].split('.')[0]
    is_train = (split_name == 'train')
    data = pd.read_csv(file_path)
    sequences = data['s']
    explanations = data['e']
    labels = data['label']
    n = len(data)
    if args.use_percent_data < 1 and is_train: n = int(n*args.use_percent_data)
    n_classes = 2
    data_idx = torch.arange(n) if is_train else -1*torch.ones(n)
    input_ids_list = []
    labels_list = []
    retrieval_ids_list = []
    load_into_retriever = ('train' in file_path and retriever is not None)
    n_truncated=0
    for i in range(n):
        s = sequences[i]
        e = explanations[i]
        label = labels[i]
        s_ids = tokenizer.encode(s, add_special_tokens=True)
        retrieval_ids = retriever_tokenizer.encode(s, add_special_tokens=True)
        if len(s_ids) > args.max_x_len:
            print(f"Truncated {n_truncated} points")
            n_truncated+=1
        _pad_seq(s_ids, args.max_seq_len if args.use_retrieval else args.max_x_len, tokenizer.pad_token_id)
        _pad_seq(retrieval_ids, args.max_seq_len if args.use_retrieval else args.max_x_len, retriever_tokenizer.pad_token_id)            
        if load_into_retriever:
            retriever.add_point(idx=i, x=s, e=e, y=label)
        input_ids_list.append(s_ids)
        labels_list.append(label)
        retrieval_ids_list.append(retrieval_ids)
    input_ids = torch.tensor(input_ids_list).long()
    input_masks = (input_ids!=tokenizer.pad_token_id).float()
    labels = torch.tensor(labels_list).long()
    retrieval_ids = torch.tensor(retrieval_ids_list).long()
    retrieval_masks = (retrieval_ids!=tokenizer.pad_token_id).float()
    if args.print:
        print(f"Example {split_name} inputs:")
        for i in range(args.num_print):
            print(i)
            print(f"  y: {labels[i]} x: {tokenizer.decode(input_ids[i], skip_special_tokens=True)} ... e: {explanations[i]}")
            e_counts = {_int : sum(np.array(sequences[i].split()) == np.array(_int)) for _int in explanations[i].split()[1:]}
            print(f"  int counts: {e_counts}")
    return_data = [data_idx, input_ids, input_masks, labels, retrieval_ids, retrieval_masks]
    return_info = {'n' : n, 'n_classes' : n_classes, f'label_dist' : {i : sum(labels==i).item()/n for i in range(n_classes)}}
    return return_data, return_info


'''
relation extraction utilities
'''

def make_RE_data(args, retriever, tokenizer, retriever_tokenizer, file_path, train_file_path=None):
    # this function got very messy due to a lot of naming and formatting idiosyncracies 
    # data set specific things
    if args.data_name == 'semeval' or 'exp' in file_path:
        x_name = 'sent'
    else:
        x_name = 'token'
    if args.data_name == 'semeval':
        label_to_id = semeval_label_to_id
        id_to_label = semeval_id_to_label
    if args.data_name == 'tacred':
        label_to_id = tacred_label_to_id
        id_to_label = tacred_id_to_label
    n_classes=len(id_to_label)
    # add tokens to tokenizer
    special_tokens_dict = {'additional_special_tokens': ['SUBJ', 'OBJ']}# if args.data_name == 'semeval' else {'additional_special_tokens': ['#subj#', '#obj#']}
    special_tokens = special_tokens_dict['additional_special_tokens']
    subj_token, obj_token = special_tokens
    tokenizer.add_special_tokens(special_tokens_dict)
    retriever_tokenizer.add_special_tokens(special_tokens_dict)
    subj_id, obj_id = tokenizer.convert_tokens_to_ids(special_tokens)
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    is_train = 'train' in file_path
    load_into_retriever = ('exp' in file_path and retriever is not None)

    # define some functions
    def _remove_identifiers_semeval(d):
        sentence = d[x_name]
        special1, special2 = special_tokens
        by_spaces = sentence.split()
        subj_idx = [i for i in range(len(by_spaces)) if special1 in by_spaces[i]]
        obj_idx = [i for i in range(len(by_spaces)) if special2 in by_spaces[i]]
        for subj_id in subj_idx:
            by_spaces[subj_id] = by_spaces[subj_id][:len(special1)]
        for obj_id in obj_idx:
            by_spaces[obj_id] = by_spaces[obj_id][:len(special2)]
        return ' '.join(by_spaces)

    def _remove_identifiers_tacred(d):
        tokens = d['token']
        # anonymize tokens
        ss, se = d['subj_start'], d['subj_end']
        os, oe = d['obj_start'], d['obj_end']
        subj_addin = '-' + d['subj_type'] if 'subj_type' in d else ''
        obj_addin = '-' + d['obj_type'] if 'obj_type' in d else ''
        tokens[ss:se+1] = ['SUBJ'+subj_addin] * (se-ss+1)
        tokens[os:oe+1] = ['OBJ'+obj_addin] * (oe-os+1)
        return ' '.join(tokens)

    def _remove_identifiers_tacred_x(d):
        return d['sent']

    def _remove_identifiers_tacred_exp(d):
        return d['exp']

    def _remove_repeat_specials(s):
        by_spaces = s.split()
        spcl_idx1 = [i for i in range(len(by_spaces)) if special_tokens[0] in by_spaces[i]]
        ss, se = min(spcl_idx1), max(spcl_idx1)
        by_spaces[ss:se+1] = [by_spaces[ss]]
        spcl_idx2 = [i for i in range(len(by_spaces)) if special_tokens[1] in by_spaces[i]]
        os, oe = min(spcl_idx2), max(spcl_idx2)
        by_spaces[os:oe+1] = [by_spaces[os]]
        return ' '.join(by_spaces)
            
    def _tokenize(sent):
        tokens = tokenizer.tokenize(sent)
        ent_idx = [tokens.index(subj_token)+1, tokens.index(obj_token)+1] # add 1 given cls token below
        input_ids = [cls_id] + tokenizer.convert_tokens_to_ids(tokens) + [sep_id]
        return input_ids, ent_idx

    def _retriever_tokenize(sent):
        tokens = retriever_tokenizer.tokenize(sent)
        input_ids = [cls_id] + retriever_tokenizer.convert_tokens_to_ids(tokens) + [sep_id]
        return input_ids
    
    _exp_remove_f = _remove_identifiers_semeval if args.data_name == 'semeval' else _remove_identifiers_tacred_exp
    if args.data_name == 'semeval':
        _train_x_remove_f = _remove_identifiers_semeval
    if args.data_name == 'tacred' and 'exp' in file_path:
        _train_x_remove_f = _remove_identifiers_tacred_x
    if args.data_name == 'tacred' and 'exp' not in file_path:
        _train_x_remove_f = _remove_identifiers_tacred

    # if train file path was provided, make a list of those sentences so we can get the ids right when we're adding explanations (so that a data point's own explanation is never conditioned on)
    if train_file_path is not None:
        with open(train_file_path) as f:
            data_json = json.load(f)
            _train_remove_f = _remove_identifiers_semeval if args.data_name == 'semeval'  else _remove_identifiers_tacred
            train_sentences = [_remove_repeat_specials(_train_remove_f(point)) for point in data_json]

    def safe_list_index(sequence, query):
        for i, s in enumerate(sequence):
            if s==query:
                return i
        return -1
    
    # load things
    with open(file_path) as f:
        data_json = json.load(f)
    n = len(data_json)
    if args.small_data:
        n = int(args.small_size)
    elif args.use_percent_data < 1 and is_train: 
        n = int(n*args.use_percent_data)
    elif args.use_num_train > 0 and is_train:
        n = int(args.use_num_train)
    use_train_idx = np.random.choice(np.arange(len(data_json)), size=n, replace=False)

    # init vars
    input_ids_list = []
    labels_list = []
    retrieval_ids_list = []
    ent_idx_list = []
    explanations = []

    # loop through data
    exp_in_train=0
    for i in use_train_idx:
        point = data_json[i]
        sent = _train_x_remove_f(point)
        # both names used in the same files...
        try:
            label = label_to_id[point['rel']]
        except:
            label = label_to_id[point['relation']]
        input_ids, ent_idx = _tokenize(sent) 
        retrieval_ids = _retriever_tokenize(sent)
        _truncate_seq_pair(input_ids, [], args.max_x_len)
        _truncate_seq_pair(retrieval_ids, [], args.max_x_len)
        _pad_seq(input_ids, length=args.max_seq_len if args.use_retrieval else args.max_x_len, pad_id=tokenizer.pad_token_id)
        _pad_seq(retrieval_ids, length=args.max_seq_len if args.use_retrieval else args.max_x_len, pad_id=tokenizer.pad_token_id)
        if 'exp' in point.keys() and load_into_retriever:
            e = _exp_remove_f(point)
            idx = safe_list_index(train_sentences, sent)
            if idx >= 0:
                exp_in_train+=1
            try:
                y = point['rel']
            except:
                y = point['relation']
            retriever.add_point(idx=idx, x=sent, e=e, y=y)
            explanations.append(e)
        input_ids_list.append(input_ids)
        labels_list.append(label)
        retrieval_ids_list.append(retrieval_ids)
        ent_idx_list.append(ent_idx)

    if 'exp' in point.keys():
        print(f"Out of {len(data_json)} explanations, {exp_in_train} correspond to data points in the train data")

    # make tensors
    data_idx = torch.arange(n).long()
    input_ids = torch.tensor(input_ids_list).long()
    input_masks = torch.tensor(input_ids!=tokenizer.pad_token_id).float()
    labels = torch.tensor(labels_list).long()
    retrieval_ids = torch.tensor(retrieval_ids_list).long()
    retrieval_masks = torch.tensor(retrieval_ids!=tokenizer.pad_token_id).float()
    ent_idx = torch.tensor(ent_idx_list).long()
    # have to clip ent_idx at max_seq_len
    ent_idx = ent_idx.masked_fill(ent_idx>=args.max_x_len, 0)
    if args.print:
        print(f"Example inputs from {file_path}:")
        for i in range(args.num_print):
            ids = [_id for _id in input_ids[i] if _id != tokenizer.pad_token_id]
            print(i)
            print(f"\ty: {id_to_label[labels[i].item()]} x: {tokenizer.decode(ids, skip_special_tokens=False)} ... e: {explanations[i] if 'exp' in point .keys() else ''}")
            ents = [tokenizer.decode(ids[ent_idx[i,j].item()].item(), skip_special_tokens=False) for j in range(2)]
            print(f"\tent tokens: {ents[0]}, {ents[1]} ... e: {explanations[i] if 'exp' in point .keys() else ''}")
    return_data = [data_idx, input_ids, input_masks, labels, retrieval_ids, retrieval_masks, ent_idx]
    return_info = {'n' : n, 'n_classes' : n_classes, f'label_dist' : {i : sum(labels==i).item()/n for i in range(n_classes)}}
    return return_data, return_info


'''
text classification utilities 
'''

def make_SNLI_data(args, retriever, tokenizer, retriever_tokenizer, file_path):
    '''
    read SNLI, make data ready for DataLoader
    '''
    data = pd.read_csv(file_path)
    # locals
    split_name = os.path.split(file_path)[-1].split('.')[0]
    is_train = (split_name == 'train')
    label_map = {0: "neutral", 1: "entailment", 2: "contradiction"}
    # n, n_classes
    n_classes = 3
    n = len(data)
    # adjust size
    if args.small_data:
        n = int(args.small_size)
    elif args.use_percent_data < 1 and is_train: 
        n = int(n*args.use_percent_data)
    elif args.use_num_train > 0 and is_train:
        n = int(args.use_num_train)
    random_idx = np.random.choice(np.arange(len(data)), size=n, replace=False)
    data = data.iloc[random_idx,:].reset_index()
    # load columns
    n_classes = len(label_map)
    ids = data['unique_key']
    premises = data['premise']
    hypotheses = data['hypothesis']
    explanations = data['explanation'] if is_train else data['explanation1']
    labels = data['label']

    # make data
    data_idx = torch.arange(n) if is_train else -1*torch.ones(n)
    input_ids_list = []
    labels_list = []
    retrieval_ids_list = []
    load_into_retriever = ('train' in file_path and retriever is not None)
    for i in range(n):
        premise = premises[i]
        hypothesis = hypotheses[i]
        premise_ids = tokenizer.encode(premise, add_special_tokens=False, max_length=args.max_seq_len, truncation=True)
        hypothesis_ids = tokenizer.encode(hypothesis, add_special_tokens=False, max_length=args.max_seq_len, truncation=True)
        label = labels[i]   
        # make classifier ids
        _truncate_seq_pair(premise_ids, hypothesis_ids, args.max_x_len-3)
        s_ids = premise_ids + [tokenizer.sep_token_id] + hypothesis_ids
        _pad_seq(s_ids, args.max_seq_len, tokenizer.pad_token_id)
        # make retriever ids
        premise_ids = retriever_tokenizer.encode(premise, add_special_tokens=False, max_length=args.max_seq_len, truncation=True)
        hypothesis_ids = retriever_tokenizer.encode(hypothesis, add_special_tokens=False, max_length=args.max_seq_len, truncation=True)
        _truncate_seq_pair(premise_ids, hypothesis_ids, args.max_x_len-3)
        retrieval_ids = premise_ids + [retriever_tokenizer.sep_token_id] + hypothesis_ids
        # make s and s_ids. need an s with the spcl token separating p/h for the retriever
        s = retriever_tokenizer.decode(s_ids, skip_special_tokens=False)
        retrieval_ids = retriever_tokenizer.encode(s, add_special_tokens=True, max_length=args.max_x_len, truncation=True)
        _pad_seq(retrieval_ids, args.max_x_len, retriever_tokenizer.pad_token_id)
        s = retriever_tokenizer.decode(retrieval_ids, skip_special_tokens=False) 
        e = explanations[i]
        if load_into_retriever:
            retriever.add_point(idx=i, x=s, e=e, y=label)
        input_ids_list.append(s_ids)
        labels_list.append(label)
        retrieval_ids_list.append(retrieval_ids)
    input_ids = torch.tensor(input_ids_list).long()
    input_masks = (input_ids!=tokenizer.pad_token_id).float()
    labels = torch.tensor(labels_list).long()
    retrieval_ids = torch.tensor(retrieval_ids_list).long()
    retrieval_masks = (retrieval_ids!=tokenizer.pad_token_id).float()
    print(f"Example {split_name} inputs:")
    for i in range(args.num_print):
        print(f"({i})  y: {labels[i]} x: {tokenizer.decode(input_ids[i], skip_special_tokens=True)} ... e: {explanations[i] if is_train else None}")
    return_data = [data_idx, input_ids, input_masks, labels, retrieval_ids, retrieval_masks]
    return_info = {'n' : n, 'n_classes' : n_classes, f'label_dist' : {i : sum(labels==i).item()/n for i in range(n_classes)}}
    return return_data, return_info

