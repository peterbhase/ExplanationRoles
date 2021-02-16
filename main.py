import os 
import torch
import numpy as np
import pandas as pd
import argparse
import time
import utils
import copy
from torch.utils.data import TensorDataset, DataLoader
from utils import str2bool
from utils import data_name_to_num_labels, get_make_data_function, get_custom_model_class
from report import Report
from models.modeling_optimal_synthetic_model import OptimalSyntheticModel
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers import get_linear_schedule_with_warmup
from retriever import Retriever
import torch.multiprocessing
from scipy import stats
from sklearn.metrics import f1_score
from write_synthetic_data import write_synthetic_data

def load_data(args, retriever, tokenizer, retriever_tokenizer):
    print("Loading data...")
    start_time = time.time()
    data_dir = args.data_dir + '_' + args.experiment_name if args.data_name == 'synthetic' else args.data_dir
    train_name, dev_name, test_name = utils.get_file_names(args.data_name)
    train_path = os.path.join(data_dir, train_name)
    dev_path = os.path.join(data_dir, dev_name)
    test_path = os.path.join(data_dir, test_name)
    make_data_function = get_make_data_function(args.data_name)
    train_dataset, train_info = make_data_function(args, retriever, tokenizer, retriever_tokenizer, file_path = train_path)
    dev_dataset, dev_info =     make_data_function(args, None,      tokenizer, retriever_tokenizer, file_path = dev_path)
    test_dataset, test_info =   make_data_function(args, None,      tokenizer, retriever_tokenizer, file_path = test_path) 
    load_time = (time.time() - start_time) / 60
    print(f"Loading data took {load_time:.2f} minutes")
    print("Data info:")
    for split_name, info in zip(['train','dev','test'], [train_info, dev_info, test_info]):
        n, n_classes, label_dist = info['n'], info['n_classes'], [round(100*x,2) for x in info['label_dist'].values()]
        print(f'  {split_name}: {n} points | {n_classes} classes | label distribution : {label_dist}')
    train_dataloader = DataLoader(TensorDataset(*train_dataset), shuffle=True, batch_size=args.train_batch_size, num_workers = 4, pin_memory = True)    
    dev_dataloader = DataLoader(TensorDataset(*dev_dataset), shuffle=False, batch_size=args.test_batch_size, num_workers = 4, pin_memory = True)    
    test_dataloader = DataLoader(TensorDataset(*test_dataset), shuffle=False, batch_size=args.test_batch_size, num_workers = 4, pin_memory = True)    
    if args.eval_on_train:
        dev_dataloader, test_dataloader = train_dataloader, test_dataloader
    # load separate explanation data for RE into retriever
    if args.task_type == 'RE' and args.use_retrieval:
        exp_file_path = os.path.join(args.data_dir, 'semeval_exp.json' if args.data_name == 'semeval' else 'tacred_exp_orig.json')
        _, exp_info = utils.make_RE_data(args, retriever, tokenizer, retriever_tokenizer, exp_file_path, train_path)
        n, n_classes, label_dist = exp_info['n'], exp_info['n_classes'], [round(100*x,2) for x in exp_info['label_dist'].values()]
        print(f'  Exp info: {n} points | {n_classes} classes | label distribution : {label_dist}')
    return train_dataloader, dev_dataloader, test_dataloader

def load_model(args, device, tokenizer, finetuned_path = None):
    print(f"\nLoading task model: {finetuned_path if finetuned_path is not None else args.model}\n")
    # first, just return the optimal synthetic model if that arg has been used
    if args.use_optimal_model:
        model = OptimalSyntheticModel(args)
        return model.to(device)
    config = AutoConfig.from_pretrained(args.model, num_labels=data_name_to_num_labels[args.data_name], cache_dir=args.cache_dir)
    config.__dict__.update(args.__dict__)
    model_class = utils.get_custom_model_class(args.model)    
    model = model_class.from_pretrained(args.model, config=config, cache_dir = args.cache_dir)
    model.resize_token_embeddings(len(tokenizer))
    if finetuned_path is not None:
        model_state_dict = torch.load(finetuned_path, map_location=lambda storage, loc: storage) # args for preventing memory leakage across gpus
        utils.rectify_mismatched_embeddings(model, model_state_dict, tokenizer)
        model.load_state_dict(model_state_dict) 
    model.to(device)
    model.train()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print("Num trainable model params: %.2fm" % (sum([np.prod(p.size()) for p in model_parameters])/1e6))
    return model

def load_retriever(args, device, task_tokenizer, retriever_tokenizer, finetuned_path=None, stored_index=None, train_use_idx=None):
    print(f"\nLoading retriever: {finetuned_path if finetuned_path is not None else args.retrieval_model}\n")
    config = AutoConfig.from_pretrained(args.retrieval_model)
    config.__dict__.update(args.__dict__)
    model = Retriever.from_pretrained(args.retrieval_model, config=config, cache_dir=args.cache_dir,
                                    task_tokenizer=task_tokenizer, retriever_tokenizer=retriever_tokenizer, stored_index=stored_index, train_use_idx=train_use_idx)    
    model.resize_token_embeddings(len(retriever_tokenizer))
    if args.reinitialize_retriever and finetuned_path is None:
        model.init_weights()
    if finetuned_path is not None:
        model_state_dict = torch.load(finetuned_path, map_location=lambda storage, loc: storage) # args for preventing memory leakage across gpus
        utils.rectify_mismatched_embeddings(model, model_state_dict, retriever_tokenizer)
        model.load_state_dict(model_state_dict) 
    model = model.to(device)
    return model

def load_optimizer(args, model, lr, num_train_optimization_steps, names_params_list = None):
    model.train()
    # for local Bert-based models, if model.use_adaptors, p.requires_grad restricts only to adaptor parameters bc .train() called in load_model
    param_optimizer = [(n, p) for n, p in list(model.named_parameters()) if p.requires_grad] if names_params_list is None else names_params_list
    no_decay = ['bias', 'LayerNorm', 'layer_norm']
    optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                                    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer

def train_or_eval_epoch(args, epoch, device, dataloader, stats_dict, multi_gpu, model, optimizer, scheduler, tokenizer, split_name, # standard args
                        retriever, retriever_optimizer, retriever_scheduler, # retrieval args
                       ):
    '''
    runs one epoch. returns stats_dict. updates model parameters if training
    '''
    # init stat vars
    loss_sum = 0
    acc_sum, mode_acc_sum = 0, 0
    n_data_points = 0
    n_batches = len(dataloader)
    start_time = time.time()
    preds_list=[]
    labels_list=[]
    is_train = (split_name == 'train' and optimizer is not None and not args.use_optimal_model)
    training_classifier = (is_train and args.train_classifier and epoch >= args.freeze_classifier_until)
    training_retriever = (is_train and args.train_retriever and epoch >= args.freeze_retriever_until)
    if args.use_retrieval:
        if is_train and training_retriever:
            retriever.train()
        else:
            retriever.eval()
    if is_train: 
        model.train()
    else:
        model.eval()

    for step, batch in enumerate(dataloader):
        running_time = (time.time()-start_time)
        est_run_time = (running_time/(step+1)*n_batches)
        rolling_acc = 100*acc_sum/n_data_points if step>0 else 0
        print(f"  {split_name.capitalize()} | Batch: {step+1}/{n_batches} | Acc: {rolling_acc:.2f} |"
              f" Loss: {loss_sum/(step+1):.4f} | Speed: {running_time/(n_data_points+1):.3f} secs / point (est. remaining epoch time: {(est_run_time-running_time)/60:.2f} minutes)", end = '\r')
        rebuild_index = (training_retriever and args.rebuild_index_every_perc > -1 and (step+1) % max(1,int(args.rebuild_index_every_perc*n_batches)) == 0)
        
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:

            # unpack batch vars
            batch = [item.to(device) for item in batch]
            data_idx, orig_input_ids, orig_attention_masks, labels, retrieval_input_ids, retrieval_attention_masks = batch[:6]
            ent_idx = None if args.task_type == 'classification' else batch[6]
            batch_size = orig_input_ids.size(0)

            # retrieve
            if args.use_retrieval:
                retrieved_idx, retrieved_labels = retriever.retrieve(retrieval_input_ids, attention_mask=retrieval_attention_masks, ignore_idx=data_idx)
                model_kwargs = retriever.make_new_data(orig_input_ids, orig_attention_masks, retrieved_idx)
                if args.top_k > 1:
                    retrieval_grad_req = torch.enable_grad() if training_retriever else torch.no_grad()
                    with retrieval_grad_req:
                        model_kwargs['retrieval_scores'] = retriever.get_retrieval_scores(retrieval_input_ids, attention_mask=retrieval_attention_masks, retrieval_idx = retrieved_idx)
            if not args.use_retrieval:
                model_kwargs = {'input_ids' : orig_input_ids,
                                'attention_mask' : orig_attention_masks
                }
            model_kwargs['labels'] = labels
            model_kwargs['ent_idx'] = ent_idx

            # forward pass
            outputs = model(**model_kwargs)
            loss = outputs[0] / args.grad_accumulation_factor             
            logprobs = outputs[1] # when k>1, these are marginal logprobs. when k=1, these are just logits
            # compute acc 
            labels = labels.detach().cpu().numpy()
            preds = np.argmax(logprobs.detach().cpu().numpy(), axis=-1)
            labels_list.extend(labels.tolist())
            preds_list.extend(preds.tolist())
            acc_sum += np.sum(preds==labels)
            if args.use_retrieval:
                retrieved_mode, _ = stats.mode(retrieved_labels.reshape(batch_size, -1), axis=1)
                mode_acc_sum += np.sum(retrieved_mode.reshape(-1)==labels.reshape(-1))
            if multi_gpu:
                loss = loss.mean()
            # backward pass
            if is_train:
                loss.backward()
                if (step+1) % args.grad_accumulation_factor == 0 or (step == n_batches-1):
                    if training_classifier:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                    optimizer.zero_grad() # norm clipping avoids catastrophe, but still avoid accumulating the grad if not stepping
                    if retriever_optimizer is not None:
                        if training_retriever:
                            torch.nn.utils.clip_grad_norm_(retriever.parameters(), args.max_grad_norm)
                            retriever_optimizer.step()
                            retriever_scheduler.step()
                        retriever_optimizer.zero_grad()
            # track stats
            loss_sum += loss.item()
            n_data_points += batch_size
            del loss

        if rebuild_index:
            retriever.build_index()
        
        if (args.print and epoch == 0 and step == 0 and is_train) or (args.print and epoch==-1 and step == 0 and split_name in ['train','dev']):
            print(f"\nEXAMPLE INPUTS")
            num_to_print = min(args.num_print, batch_size)
            for i in range(num_to_print):
                print(f"data point: {i} (idx: {data_idx[i].item()})")
                ids = [_id for _id in orig_input_ids[i] if _id != tokenizer.pad_token_id]
                print(f"Orig input: {tokenizer.decode(ids, skip_special_tokens=False)}")
                for k in range(args.top_k):
                    if args.use_ELV:
                        print("Print is `folded` for ELV, so each context is shown in one long string here:")
                    if args.top_k > 1:                         
                        ids = [_id for _id in model_kwargs['input_ids'][i,k].reshape(-1) if _id != tokenizer.pad_token_id]
                        input_str = f"Model input (k: {k}): {tokenizer.decode(ids, skip_special_tokens=False)}"
                        label_logprobs, latent_logprobs = outputs[2], outputs[3]
                        label = labels[i]
                        char_buffer = 65 + 11*args.context_size
                        input_str += ' '*(char_buffer-len(input_str)) + f" \t | lnP(y|x) {logprobs[i,label]:.2f} | lnP(y|x,e) : {label_logprobs[i,k,label]:.2f} | lnP(e|x) : {latent_logprobs[i,k]:.2f}"                        
                    else:
                        ids = [_id for _id in model_kwargs['input_ids'][i].reshape(-1) if _id != tokenizer.pad_token_id]
                        input_str = f"Model input: {tokenizer.decode(ids, skip_special_tokens=False)}"                        
                    print(input_str)
                print(f"Label vs pred: {labels[i]}, {preds[i]}") 

    # summary stats
    loss_mean = loss_sum / n_data_points
    acc_mean = acc_sum / n_data_points
    mode_mean = mode_acc_sum / n_data_points
    f1 = f1_score(np.array(labels_list), np.array(preds_list), labels=range(1*(args.task_type=='RE'),model.config.num_labels), average='micro') # this is exactly accuracy ignoring labels and preds for the no_relation class
    if split_name == 'train': 
        stats_dict[f'{split_name}_loss'] = loss_mean
    stats_dict[f'{split_name}_acc'] = acc_mean * 100
    stats_dict[f'{split_name}_f1'] = f1 * 100
    stats_dict[f'{split_name}_mode_acc'] = mode_mean * 100
    run_time = (time.time() - start_time) / 60
    print(f"  {split_name.capitalize()} time: {run_time:1.2f} min. ")
    
    return stats_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # misc. & debugging
    parser.add_argument('--gpu', type = int, default = 0, help = 'gpu id to use')
    parser.add_argument("--seed", default='0', type=int, help='')  
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--eval_on_train", action='store_true')
    parser.add_argument("--small_data", '-s', action='store_true')
    parser.add_argument("--small_size", '-ss', default=100, type=int, help='')
    parser.add_argument("--print", default = False, type=str2bool, help = 'flag for printing things helpful for debugging / seeing whats happening')
    parser.add_argument("--num_print", default=2, type=int, help='')
    # hyperparams
    parser.add_argument("--patience", default=100, type=int, help='after this many epochs with no dev improvement, break from training')
    parser.add_argument("--train_batch_size", default=10, type=int, help='')
    parser.add_argument("--test_batch_size", default=10, type=int, help='')
    parser.add_argument("--grad_accumulation_factor", default=1, type=int, help='')
    parser.add_argument("--num_train_epochs", default=20, type=int, help='')
    parser.add_argument("--lr", default=1e-5, type=float, help='')  
    parser.add_argument("--retriever_lr", default=1e-5, type=float, help='')  
    parser.add_argument("--warmup_proportion", default=.1, type=float, help='')
    parser.add_argument("--max_x_len", default=100, type=int, help='max length of x in data loading.')  
    parser.add_argument("--max_e_len", default=40, type=int, help='max length of explanations as added to data.')
    parser.add_argument("--max_seq_len", default=140, type=int, help='max length of x plus whatever explanations may be appended to it')
    parser.add_argument("--max_grad_norm", default=1, type=float, help='')  
    parser.add_argument("--use_adaptors", default = False, type=str2bool, help = '')
    parser.add_argument("--adaptor_type", default = 'film', type=str, choices = ['bottleneck', 'film', 'unamortized-film', 'per_layer_film'], 
                                                    help = 'type of adaptor network used. film uses NP to parametrize film layers. unarmortized film uses already parametrized film layers')
    parser.add_argument("--d_adaptor", default=128, type=int, help='dimension of bottleneck hidden state in adaptor networks')
    parser.add_argument("--token_pooling", default = 'mean', choices=['cls','mean','max', 'ent_idx'], type=str, help  = 'ent_idx means pull out and cat vector reps at corresponding ent_idx')
    parser.add_argument("--context_pooling", default = 'mean', choices=['mean','max'], type=str, help  = '')
    # generic paths
    parser.add_argument("--report_dir", default='training_reports/', type=str)
    parser.add_argument("--save_dir", default='', type=str)
    parser.add_argument("--cache_dir", default='', type=str)
    parser.add_argument("--data_dir", default='data/synthetic', type=str)    
    parser.add_argument("--server", default='13', type=str)
    # experiment flags & paths
    parser.add_argument("--model", default='roberta-base', type=str, help='name of pretrained model')
    parser.add_argument("--pretrained_model", default = None, type=str, help = 'path for loading finetuned model')
    parser.add_argument("--pretrained_retriever", default = None, type=str, help = 'path for loading finetuned retriever')
    parser.add_argument("--use_percent_data", '-u', default=1., type=float, help='if < 1, use this percentage of the train data')
    parser.add_argument("--use_num_train", default=-1, type=int, help='if > 0, use this number of train data points')
    parser.add_argument("--experiment_name", '-e', default = None, type=str, help = 'if not none, this overrides the experiment_name')
    parser.add_argument("--ablate_k", default = False, type=str2bool, help  = 'after training, ablate retrieval model performance across k values')
    # retrieval args
    parser.add_argument("--retrieval_model", default='sentence-transformers/roberta-base-nli-stsb-mean-tokens', type=str, help='name of pretrained model')
    parser.add_argument("--use_retrieval", default = True, type=str2bool, help  = 'condition on retrieved context')
    parser.add_argument("--train_classifier", default = True, type=str2bool, help  = 'train the classifier model')
    parser.add_argument("--train_retriever", default = False, type=str2bool, help  = 'train the retriever model')
    parser.add_argument("--freeze_classifier_until", default = -1, type=int, help  = 'update classifier after this epoch if train_classifier')
    parser.add_argument("--freeze_retriever_until", default = -1, type=int, help  = 'update retriever after this epoch if train_retriever')
    parser.add_argument("--backprop_targets", default = False, type=str2bool, help  = 'backprop through embeddings of retrieved documents')
    parser.add_argument("--rebuild_index_every_epoch", default = False, type=str2bool, help  = 'rebuild index every epoch')
    parser.add_argument("--rebuild_index_every_perc", default = -1, type=float, help  = 'rebuild index every percentage of each epoch. set to (-1, 0] to rebuild every single batch')
    parser.add_argument("--fixed_index", default = False, type=str2bool, help  = 'never rebuild index')    
    parser.add_argument("--reinitialize_retriever", default = False, type=str2bool, help  = 're-initialize the retriever weights to begin with random model')
    parser.add_argument("--use_optimal_retrieval", default = False, type=str2bool, help  = 'use the true retrieval function pi:(x,x) -> s or pi:(x,e)->s')    
    parser.add_argument("--precision", default=1, type=float, help='precision on p(idx|x) distribution')
    parser.add_argument("--retrieve_on", default = 'XE', choices=['XX','XE'], type=str,
                        help="similarity for retrieval is either f(x,x) or f(x,e)")
    parser.add_argument("--retrieval_metric", default = 'cosine_sim', choices=['cosine_sim','L2','inner_product'], type=str)
    parser.add_argument("--fast_retrieval", default = False, type=str2bool, help  = 'use FAISS fast retrieval (search voronoi cells)')
    parser.add_argument("--sampling_method", default = 'top_k', choices=['top_k','half_random','importance'], type=str,
                        help="how to get the k_samples context-sets. either get top-k, or get half from the top, half random, or use importance sampling. no NCE bc no ground truth")
    parser.add_argument("--degrade_retriever", default=-1, type=float, help='add eps ~ N(0,sd*I) to retriever parameters')
    parser.add_argument("--exclude_correct_explanations", default = False, type=str2bool, help  = 'when retrieving, never retrieve the exactly correct explanations (i.e. same index)')
    # synthetic data arguments
    parser.add_argument("--use_optimal_model", default = False, type=str2bool, help  = 'used for testing synthetic data is solvable')
    parser.add_argument("--num_tasks", default=-1, type=int, help = 'if > 0, this determines the number of tasks in the data, and hence it determines num_relevant_points per task (see next)')
    parser.add_argument("--num_relevant_points", default=-1, type=int, 
        help='if <0, num_relevant_points=context_size+1.' \
             'used to create a mismatch between context set size and the available relevant points' \
             'NOTE that if num_tasks is not used, this determines the number of tasks/z in the data, since we assume that all explanations are informative'
    )
    parser.add_argument("--single_task", default = False, type=str2bool, help  = 'task solvable as f(x). done by adding missing information to s')
    parser.add_argument("--smooth_idx_to_z", default = False, type=str2bool, help  = ' make f(idx) -> z smooth to ease retrieval learning')
    parser.add_argument("--max_int", default=100, type=int, help='max int to appear in sequences.')
    parser.add_argument("--max_idx", default=-1, type=int, help='if -1, max_idx = max_int**2. else, should be >= n_train. will shrink the space of possible tasks in terms of integer size')
    parser.add_argument("--ordered_mnrd", default = False, type=str2bool, help  = 'if true, instead of randomly sampling mnrd, write them as [range(max_idx)]*4 basically')
    parser.add_argument("--num_train_synthetic", '-n', default=5000, type=int, help='number of train data points for synthetic task')
    parser.add_argument("--use_mn_only", default = False, type=str2bool, help  = 'if true, label always decided by mn rather than rd, hence task solvable as f(x) on the index')
    parser.add_argument("--use_index", default = True, type=str2bool, help  = 'if true, set s[0] as index pointing to other relevant (x,e) data')
    parser.add_argument("--use_mn_indicator", default = True, type=str2bool, help  = 'if true, set s[1] as indicator for whether to use mn or rd')
    parser.add_argument("--explanation_kind", default = 'missing_info', type=str, choices =['missing_info', 'recomposable', 'evidential'], 
        help  = 'this says whether e=missing_info, or e = natural_language(z), and others')
    parser.add_argument("--evidential_eps", default=1, type=int, help='+/- eps noise to appear with evidential explanations.')
    parser.add_argument("--recomposable_pieces", default=2, type=int, help='number of pieces to form a complete explanation.')
    parser.add_argument("--recomposable_additive", default = False, type=str2bool, help  = 'if true, the true mnrd is obtained by summing mnrd across explanations (rather than having 0s in off elements)')
    parser.add_argument("--idx_function", default = 'identity', type=str, choices =['identity','easy','hard','noise'], 
        help  = 'this is the function between the x index and the e index. vary difficulty of learning retrieval')
    parser.add_argument("--weak_feature_correlation", default=.5, type=float, help='train time correlation of weak (non-causal) feature with causal feature')
    parser.add_argument("--explanation_only_causal", default = False, type=str2bool, help  = 'if true, explanation only includes causal integers (m,n) or (r,d)')
    parser.add_argument("--translate_explanation", default = None, choices=['100minusx','xplus5'], type=str,
        help  = 'if true, set mnrd = f(mnrd), requiring model to decode this function')
    parser.add_argument("--disjoint_test_idx", default = False, type=str2bool, help  = 'if true, dev/test idx are not seen during training')
    # latent variables / conditioning
    parser.add_argument("--context_includes", default = 'YXE', choices=['YXE','YX','E','XE','X','Y','YE','empty'], type=str,
                        help="the parts of the context data points to be conditioned on when predicting queries")
    parser.add_argument("--context_size", default=1, type=int, help='number of context points conditioned on')
    parser.add_argument("--top_k", default=1, type=int, help='number of samples of context SETS to marginalize over as latent variables')
    # text concat params
    parser.add_argument("--use_textcat", default = False, type=str2bool, help  = 'condition on context via text concatenation with the query')    
    # ELV params
    parser.add_argument("--use_ELV", default = False, type=str2bool, help  = 'use model from ELV paper for conditioning on explanations')
    # control flow
    parser.add_argument("--do_train", default = True, type=str2bool, help = '')
    parser.add_argument("--do_eval", default = True, type=str2bool, help = '')
    parser.add_argument("--pre_eval", default = False, type=str2bool, help = '')

    # parse + experiment checks
    args = parser.parse_args()
    assert not args.use_retrieval or args.use_ELV or args.use_textcat, "if use_retrieval, then use one of these conditioning mechanisms"
    assert not args.top_k > 1 or args.use_retrieval, "if top_k>1, use retrieval"
    assert not (args.fixed_index and (args.rebuild_index_every_epoch or args.rebuild_index_every_perc > -1))
    assert not (not args.train_retriever and (args.rebuild_index_every_epoch or args.rebuild_index_every_perc > -1))
    assert 'roberta' in args.model, "to use non-roberta models, please remove uses of utils.rectify_mismatched_embeddings"
    
    # add arguments
    args.data_name = os.path.split(args.data_dir)[-1]
    if args.data_name in ['semeval','tacred']:  args.task_type = 'RE'
    if args.data_name in ['eSNLI','eSNLI_full','synthetic']: args.task_type = 'classification'
    
    # GPU + SEED set-up
    n_gpu = torch.cuda.device_count()
    multi_gpu = (n_gpu > 1 and args.gpu == -1) # i.e. multiple gpus available and gpu choice not specified
    if multi_gpu: 
        device = torch.device("cuda") if args.gpu == -1 else torch.device(f'cuda:{args.gpu}')
        assert args.train_batch_size % n_gpu == 0, f"Train batch size will need to be allocated equally across {n_gpu} gpus, but {args.train_batch_size} cannot be"
        assert args.test_batch_size % n_gpu == 0, f"Eval batch size will need to be allocated equally across {n_gpu} gpus, but {args.test_batch_size} cannot be"
    else:
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    args.device = device
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if multi_gpu:
        torch.cuda.manual_seed_all(args.seed)

    # make Report object, stats_dict, and paths
    args.experiment_name = experiment_name = utils.get_experiment_name(args) if args.experiment_name is None else args.experiment_name
    if args.small_data and args.do_train:
        experiment_name += f"_DEBUG"
    model_path = os.path.join(args.save_dir, f'{experiment_name}.pth')
    retriever_path = os.path.join(args.save_dir, f'{experiment_name}_retriever.pth')
    print(f"\nStarting experiment: {experiment_name}") 
    # make pretrained_model_path
    pretrained_model_path = os.path.join(args.save_dir, f'{args.pretrained_model}.pth') if args.pretrained_model else None
    pretrained_retriever_path = os.path.join(args.save_dir, f'{args.pretrained_retriever}.pth') if args.pretrained_retriever else None
    # report + stats
    report_name = f"report_{experiment_name}.txt"
    report_file = os.path.join(args.report_dir, report_name)
    if not os.path.exists(args.report_dir): os.mkdir(args.report_dir)
    score_names = ['train_loss', 'train_acc', 'train_f1', 'train_mode_acc', 'dev_acc', 'dev_f1', 'dev_mode_acc', 'test_acc', 'test_f1', 'test_mode_acc']
    report = Report(args, report_file, experiment_name = experiment_name, score_names = score_names)
    stats_dict = {name : 0 for name in score_names}

    # write data for synthetic task
    if args.data_name == 'synthetic':
        train_use_idx = write_synthetic_data(args)
    else:
        train_use_idx = None

    # LOAD TOKENIZER, DATA, MODEL, OPTIMIZER, and RETRIEVER
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir = args.cache_dir)
    retriever_tokenizer = AutoTokenizer.from_pretrained(args.retrieval_model, cache_dir = args.cache_dir)
    assert type(tokenizer) == type(retriever_tokenizer), "different tokenizers not extensively tested"
    if args.do_train:
        if args.use_retrieval:
            # need tokenizer to be the right size before loading retriever. and retriever has to be loaded before loading the data
            if args.task_type=='RE':
                special_tokens_dict = {'additional_special_tokens': ['SUBJ', 'OBJ']}
                retriever_tokenizer.add_special_tokens(special_tokens_dict)
            retriever = load_retriever(args, device, task_tokenizer=tokenizer, retriever_tokenizer=retriever_tokenizer, finetuned_path = pretrained_retriever_path, train_use_idx=train_use_idx)
        else:
            retriever=None
        train_dataloader, dev_dataloader, test_dataloader = load_data(args, retriever, tokenizer, retriever_tokenizer)
        model = load_model(args, device, tokenizer, finetuned_path = pretrained_model_path)
        num_train_optimization_steps = args.num_train_epochs * int(len(train_dataloader.dataset) / args.train_batch_size / args.grad_accumulation_factor)
        print("NUM OPT STEPS: ", num_train_optimization_steps)
        optimizer = load_optimizer(args, model = model, lr=args.lr, num_train_optimization_steps = num_train_optimization_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps= int(args.warmup_proportion * num_train_optimization_steps), num_training_steps=num_train_optimization_steps)
        if args.use_retrieval:
            retriever_optimizer = load_optimizer(args, model = retriever, lr=args.retriever_lr, num_train_optimization_steps = num_train_optimization_steps)        
            retriever_scheduler = get_linear_schedule_with_warmup(retriever_optimizer, num_warmup_steps= int(args.warmup_proportion * num_train_optimization_steps), num_training_steps=num_train_optimization_steps)
        else:
            retriever_optimizer = retriever_scheduler = None
        if multi_gpu: model = torch.nn.DataParallel(model, device_ids = range(n_gpu))
        if args.rebuild_index_every_perc > 0:
            print(f"REBUILDING INDEX {1//args.rebuild_index_every_perc} TIMES PER EPOCH")

        # build retrieval index (after load_data)
        if args.use_retrieval:
            retriever.build_index()

        # noise the retriever 
        if args.degrade_retriever > 0:
            utils.add_noise_to_model(sd=args.degrade_retriever, model=retriever)

    if args.debug:
        import pdb; pdb.set_trace()

    # pre-training checks
    if args.pre_eval: 
        print("Pre evaluation...")
        pre_stats_dict = train_or_eval_epoch(args=args,
                                            epoch=-1,
                                            device=device,
                                            stats_dict={},
                                            dataloader=dev_dataloader,
                                            multi_gpu=multi_gpu,
                                            model=model,
                                            optimizer=None,
                                            scheduler=None,
                                            tokenizer=tokenizer,
                                            split_name='dev',
                                            retriever=retriever, 
                                            retriever_optimizer=retriever_optimizer, 
                                            retriever_scheduler=retriever_scheduler,
        )
        report.print_epoch_scores(epoch = -1, scores = pre_stats_dict)

    # BEGIN TRAINING
    best_epoch = -1.0
    best_score = -1.0
    start_time = time.time()
    if args.do_train:
        print("\nBeginning training...\n")
        patience=0
        for e in range(args.num_train_epochs):
            print(f"Epoch {e} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            stats_dict = train_or_eval_epoch(args=args,
                                        epoch=e,
                                        device=device,
                                        stats_dict=stats_dict,
                                        dataloader=train_dataloader,
                                        multi_gpu=multi_gpu,
                                        model=model,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        tokenizer=tokenizer,
                                        split_name='train',
                                        retriever=retriever, 
                                        retriever_optimizer=retriever_optimizer, 
                                        retriever_scheduler=retriever_scheduler,
            )
            # rebuild index            
            if args.rebuild_index_every_epoch or args.rebuild_index_every_perc > -1:
                retriever.build_index()
            stats_dict = train_or_eval_epoch(args=args,
                                            epoch=e,
                                            device=device,
                                            stats_dict=stats_dict,
                                            dataloader=dev_dataloader,
                                            multi_gpu=multi_gpu,
                                            model=model,
                                            optimizer=optimizer,
                                            scheduler=scheduler,
                                            tokenizer=tokenizer,
                                            split_name='dev',
                                            retriever=retriever, 
                                            retriever_optimizer=retriever_optimizer, 
                                            retriever_scheduler=retriever_scheduler,
            )
            # get score, write/print results, check for new best
            score = stats_dict['dev_acc']
            report.write_epoch_scores(epoch = e, scores = stats_dict)
            report.print_epoch_scores(epoch = e, scores = stats_dict)
            if score > best_score:
                print(f"  New best model. Saving model at {model_path}\n")
                torch.save(model.state_dict(), model_path)
                if args.use_retrieval:
                    torch.save(retriever.state_dict(), retriever_path)
                best_score, best_epoch = score, e
                patience = 0
            else:
                patience += 1
                if patience >= args.patience:
                    print(f"\n Patience of {args.patience} exceeded at epoch {e}! \n")
                    break
            
    end_time = time.time()
    training_time = (end_time-start_time) / 60
    unit = 'minutes' if training_time < 60 else 'hours'
    training_time = training_time if training_time < 60 else training_time / 60
    time_msg = f"\nTotal training time: {training_time:.2f} {unit}"
    print(time_msg)

    # FINAL EVAL
    if not args.do_train:
        if args.use_retrieval:
            # need tokenizer to be the right size before loading retriever. and retriever has to be loaded before loading the data
            if args.task_type=='RE' and not args.do_train:                
                special_tokens_dict = {'additional_special_tokens': ['SUBJ', 'OBJ']}
                retriever_tokenizer.add_special_tokens(special_tokens_dict)
            retriever = load_retriever(args, device, tokenizer, retriever_tokenizer, finetuned_path = retriever_path, train_use_idx=train_use_idx)
        else:
            retriever=None
        train_dataloader, dev_dataloader, test_dataloader = load_data(args, retriever, tokenizer, retriever_tokenizer)
    # if trained the retriever, reload the weights here
    if args.do_train and args.train_retriever:
        retriever.load_state_dict(  
            load_retriever(args, device, tokenizer, retriever_tokenizer, finetuned_path = retriever_path, train_use_idx=train_use_idx).state_dict()
        )
    # now rebuild index if we are rebuilding the index
    if args.use_retrieval and not args.fixed_index:
        retriever.build_index()
    model = load_model(args, device, tokenizer, finetuned_path = model_path)
    if multi_gpu: model = torch.nn.DataParallel(model)
    
    # final evaluation
    if args.do_eval:
        print("\nGetting final eval results...\n")
        if args.data_name=='synthetic':
            stats_dict = train_or_eval_epoch(args=args,
                                            epoch=-1,
                                            device=device,
                                            stats_dict=stats_dict,
                                            dataloader=train_dataloader,
                                            multi_gpu=multi_gpu,
                                            model=model,
                                            optimizer=None,
                                            scheduler=None,
                                            tokenizer=tokenizer,
                                            split_name='train',
                                            retriever=retriever, 
                                            retriever_optimizer=None, 
                                            retriever_scheduler=None,
            )
        stats_dict = train_or_eval_epoch(args=args,
                                        epoch=-1,
                                        device=device,
                                        stats_dict=stats_dict,
                                        dataloader=dev_dataloader,
                                        multi_gpu=multi_gpu,
                                        model=model,
                                        optimizer=None,
                                        scheduler=None,
                                        tokenizer=tokenizer,
                                        split_name='dev',
                                        retriever=retriever, 
                                        retriever_optimizer=None, 
                                        retriever_scheduler=None,
        )
        stats_dict = train_or_eval_epoch(args=args,
                                        epoch=-1,
                                        device=device,
                                        stats_dict=stats_dict,
                                        dataloader=test_dataloader,
                                        multi_gpu=multi_gpu,
                                        model=model,
                                        optimizer=None,
                                        scheduler=None,
                                        tokenizer=tokenizer,
                                        split_name='test',
                                        retriever=retriever, 
                                        retriever_optimizer=None, 
                                        retriever_scheduler=None,
        )
        train_acc = stats_dict['train_acc']
        dev_acc = stats_dict['dev_acc']
        test_acc = stats_dict['test_acc']
        final_msg = f"Best epoch: {best_epoch} | train acc: {train_acc:.2f} | " \
                    f"dev acc: {dev_acc:.2f} | test acc: {test_acc:.2f} "
        if args.do_train:
            report.write_final_score(args, final_score_str = final_msg, time_msg=time_msg)
        report.print_epoch_scores(epoch = best_epoch, scores = {k:v for k,v in stats_dict.items()})

        # set dev acc and test acc to environment variables for collection across grid searches. see run_tasks.py
        np.save('tmp_dev_acc', dev_acc)
        np.save('tmp_test_acc', test_acc)

        # ablate by k for retrieval models to better understand how retrieval is working
        if args.ablate_k:
            stats_dict = {}
            ks = [1,2,4,6,8,10,12,14,16,18,20]
            print("Ablating k now across: ", ks)
            sheet_path = os.path.join('result_sheets', experiment_name + '_by_k.csv')
            cols = {param : [] for param in ['k','dev_acc']}
            data = pd.DataFrame(cols)
            for k in ks:
                print(f"Starting k={k}")
                retriever.top_k = k
                stats_dict = train_or_eval_epoch(args=args,
                                        epoch=-1,
                                        device=device,
                                        stats_dict=stats_dict,
                                        dataloader=dev_dataloader,
                                        multi_gpu=multi_gpu,
                                        model=model,
                                        optimizer=None,
                                        scheduler=None,
                                        tokenizer=tokenizer,
                                        split_name='dev',
                                        retriever=retriever, 
                                        retriever_optimizer=None, 
                                        retriever_scheduler=None,
                )
                report.print_epoch_scores(epoch = -1, scores = {k:v for k,v in stats_dict.items()})
                results = {'k' : k, 'dev_acc' : stats_dict['dev_acc']}
                data = data.append(results, ignore_index=True)
            data.to_csv(sheet_path, index=False)

