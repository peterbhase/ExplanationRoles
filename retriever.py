import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import faiss
from transformers import BertPreTrainedModel, BertModel, RobertaModel, DistilBertModel
from torch.utils.data import TensorDataset, DataLoader
from models.modeling_pooler import Pooler
from models.modeling_roberta_with_adaptors import RobertaModel

class Retriever(RobertaModel):
    '''
    class for retrieving explanations given queries x
    - this class will store the entire training dataset
    - includes functionality for making new data from retrived idx
    '''
    def __init__(self, config, task_tokenizer, retriever_tokenizer, stored_index=None, train_use_idx=None):
        super().__init__(config)
        self.config=config
        self.__dict__.update(config.__dict__)
        self.sequence_strs = []
        self.explanation_strs = []
        self.sequence_ids = []
        self.labels = []
        self.label_ids = []
        self.explanation_ids = []
        self.representations = []
        self.task_sequence_ids = []
        self.task_explanation_ids = []
        self.task_label_ids = []
        self.index = None # this is the FAISS index
        self.max_idx = config.max_int**2
        if self.use_optimal_retrieval:
            self.idx_to_data = {}
            self.used_data_idx = None
        self.pooler = Pooler(pooling_type='mean')
        self.similarity_function = self.get_similarity_function()
        self.train_dataloader = None
        self.set_tokenizers(task_tokenizer, retriever_tokenizer)
        self.set_stored_index(stored_index)
        self.set_used_data_idx(train_use_idx)
        if self.data_name == 'synthetic':
            self.index_to_train_idx = {index : [] for index in range(1, self.max_idx+1)}
            self.num_relevant_points = self.num_relevant_points if self.num_relevant_points > 0 else self.num_train_synthetic // self.num_tasks

    def set_used_data_idx(self, data_idx):
        # see write_synthetic_data.py
        self.used_data_idx = data_idx

    def set_tokenizers(self, task_tokenizer, retriever_tokenizer):
        self.task_tokenizer = task_tokenizer
        self.task_x_prefix = self.task_tokenizer.encode(' x: ', add_special_tokens=False)
        self.task_y_prefix = self.task_tokenizer.encode(' y: ', add_special_tokens=False)
        self.task_e_prefix = self.task_tokenizer.encode(' e: ', add_special_tokens=False)
        self.retriever_tokenizer = retriever_tokenizer
        self.x_prefix = self.retriever_tokenizer.encode(' x: ', add_special_tokens=False)
        self.y_prefix = self.retriever_tokenizer.encode(' y: ', add_special_tokens=False)
        self.e_prefix = self.retriever_tokenizer.encode(' e: ', add_special_tokens=False)

    def set_stored_index(self, stored_index):
        assert self.index is None, "calling set_stored_index but self.index exists"
        if stored_index is not None:
            self.index = stored_index

    def get_representations(self, input_ids, attention_mask):
        outputs = self(input_ids, attention_mask=attention_mask)
        representations = self.pooler(outputs[0], mask=attention_mask)
        return representations

    def add_point(self, idx : int, x : str, y, e : str,):
        # here idx is location in train data, not the idx between data and explanations
        self.sequence_strs.append(x)
        self.explanation_strs.append(e)
        sequence_ids = self.retriever_tokenizer.encode(x, add_special_tokens=False)
        explanation_ids = self.retriever_tokenizer.encode(e, add_special_tokens=False)
        label_ids = self.retriever_tokenizer.encode(str(y), add_special_tokens=False)
        # truncate sequences
        self._truncate_seq(sequence_ids, self.max_x_len)
        self._truncate_seq(explanation_ids, self.max_e_len)
        # accumulate data
        self.sequence_ids.append(sequence_ids)
        self.explanation_ids.append(explanation_ids)
        self.labels.append(y)
        self.label_ids.append(label_ids)
        if self.use_optimal_retrieval:
            key = self.get_index_from_seq(sequence_ids)
            value = {'idx' : idx, 'x' : sequence_ids, 'e' : explanation_ids, 'y' : label_ids}
            if key in self.idx_to_data:
                self.idx_to_data[key].append(value)
            else:
                self.idx_to_data[key] = [value]
        # add to index_to_train_data
        if self.data_name == 'synthetic':
            index = int(x.split()[0])
            self.index_to_train_idx[index].append(len(self.sequence_ids)-1)
        # now do the same with the task tokenizer    
        sequence_ids = self.task_tokenizer.encode(x, add_special_tokens=False)
        explanation_ids = self.task_tokenizer.encode(e, add_special_tokens=False)
        label_ids = self.task_tokenizer.encode(str(y), add_special_tokens=False)
        # truncate sequences
        self._truncate_seq(sequence_ids, self.max_x_len)
        self._truncate_seq(explanation_ids, self.max_e_len)
        # accumulate data
        self.task_sequence_ids.append(sequence_ids)
        self.task_explanation_ids.append(explanation_ids)
        self.task_label_ids.append(label_ids)    

    def encode_idx(self, idx: int) -> torch.Tensor:
        input_ids = []
        input_ids += self.sequence_ids[idx] if self.retrieve_on == 'XX' else self.explanation_ids[idx]
        input_ids = [self.retriever_tokenizer.cls_token_id] + input_ids + [self.retriever_tokenizer.sep_token_id]
        return input_ids

    def build_dataloader(self):
        # first, build a dataloader
        input_ids_list = []
        for idx in range(len(self.sequence_strs)):
            input_ids = self.encode_idx(idx)
            self._pad_to_length(input_ids, pad_token_id=self.retriever_tokenizer.pad_token_id)
            input_ids_list.append(input_ids)
        input_ids = torch.tensor(input_ids_list).long()
        masks = (input_ids!=self.retriever_tokenizer.pad_token_id).float()
        dataloader = DataLoader(TensorDataset(input_ids, masks), shuffle=False, batch_size=2*self.test_batch_size, num_workers = 4, pin_memory = True)
        self.train_dataloader = dataloader

    def build_representations(self) -> np.ndarray:
        if self.index is None:
            print("Building representations...", end='\r')
        start_time = time.time()
        if self.train_dataloader is None:
            self.build_dataloader()
        n = len(self.sequence_strs)
        representations_list = []
        # now, iterate through dataloader and get representations
        with torch.no_grad():
            for batch in self.train_dataloader:
                batch = [item.to(self.device) for item in batch]
                input_ids, mask = batch
                with torch.no_grad():
                    representations = self.get_representations(input_ids, attention_mask=mask)                
                representations = representations.cpu()
                representations_list.append(representations)
        if self.index is None:
            print(f"\t\t\t took {round((time.time()-start_time)/60,2)} minutes!")
        representations = torch.cat(representations_list).numpy()
        return representations

    def build_index(self):
        # process representations
        self.eval()        
        representations = self.build_representations()
        n = representations.shape[0]
        d = representations.shape[1]
        if self.retrieval_metric == 'cosine_sim':
            faiss.normalize_L2(representations)
        # set up FAISS
        if not self.fast_retrieval:
            index = faiss.IndexFlatL2(d) if self.retrieval_metric == 'L2' else faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)
            index.add(representations)
        if self.fast_retrieval:
            quantizer = faiss.IndexFlatL2(d) if self.retrieval_metric == 'L2' else faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)
            num_cells = 100
            num_to_visit = 10
            index = faiss.IndexIVFFlat(quantizer, d, num_cells)
            index.train(representations)        
        self.index = index

    def get_similarity_function(self):
        # NOTE torch.exp is wrapped around result in .get_retrieval_scores following summing across scores for the set in p(y|x,{e})
        if self.retrieval_metric == 'L2':
            similarity_function = lambda x1,x2: self.precision * -torch.norm(x1-x2, p=2)**2
        if self.retrieval_metric == 'cosine_sim':
            similarity_function = lambda x1,x2: self.precision * torch.dot(x1,x2) / torch.norm(x1,p=2) / torch.norm(x2,p=2)
        if self.retrieval_metric == 'inner_product':
            similarity_function = lambda x1,x2: self.precision * torch.dot(x1,x2)
        if self.use_optimal_retrieval:
            similarity_function = lambda x1,x2: self.precision * (x1==x2).float()
        return similarity_function

    def compute_similarity(self, x1, x2):
        # x1 and x2 should have share the first dim. last dim assumed to be hidden dimensionality.
        similarities = torch.zeros(x1.size(0)).to(self.device)
        for i, (u,v) in enumerate(zip(x1, x2)):
            similarities[i] = self.similarity_function(u,v)
        return similarities

    def retrieve(self, input_ids, attention_mask, ignore_idx=None):
        '''
        returns:
        - idx in self.sequence_ids of retrieved points as array of shape : batch_size x k x context_size
        - labels of retrieved points as array of shape : batch_size x k x context_size
        - if ignore_idx provided (should always provided during train/dev, not in test), these are avoided in the retrieval
        '''
        num_retrieve = self.context_size * self.top_k
        if self.use_optimal_retrieval:
            return_idx = self.optimal_retrieval(input_ids, num_retrieve=num_retrieve)
        else:
            with torch.no_grad():
                return_idx = self.model_based_retrieval(input_ids, attention_mask, num_retrieve=num_retrieve, ignore_idx=ignore_idx)
        return_idx = np.array(return_idx).reshape(input_ids.size(0), self.top_k, self.context_size) 
        # now get labels of those points in flattened form (batch size x k*context_size)
        return_labels = np.array([[[self.labels[idx] for idx in context] for context in top_k] for top_k in return_idx])
        return return_idx, return_labels

    def model_based_retrieval(self, input_ids, attention_mask, num_retrieve, ignore_idx=None):
        '''
        returns idx of nearest points in the FAISS self.index
        if ignore_idx are >=0, will avoid returning these idx. (ignore_idx are all -1 for dev/test splits). used to avoid retrieving the same data points during training
        '''
        batch_size = input_ids.size(0)
        representations = self.get_representations(input_ids, attention_mask=attention_mask)
        representations = representations.cpu().numpy()
        if self.retrieval_metric == 'cosine_sim':
            faiss.normalize_L2(representations)
        extra_to_retrieve = 1 if not self.exclude_correct_explanations else self.num_relevant_points # plus one in case we need to the data point the e was given for. plus top_k for if args.exclude_correct_explanations=true
        distances_or_similarities, indices = self.index.search(representations, num_retrieve + extra_to_retrieve) 
        # will pull out the first num_retrieve column elements for each data point (row) in indices, while skipping elements when they match their respective ignore_idx in the train data
        if any(ignore_idx>=0): # any here is equivalent to all, given that ignore_idx will be entirely -1 for dev/test
            ignore_idx = ignore_idx.cpu().numpy()
            batch_idx_into_indices = np.zeros((batch_size, num_retrieve), dtype=np.int32)
            for i in range(batch_size):
                retrieved_self = ignore_idx[i] in indices[i]
                # exclude only the data point corresponding to the explanation
                if not self.exclude_correct_explanations:
                    if not retrieved_self:
                        data_point_idx_into_indices = list(range(num_retrieve))
                    else:
                        data_point_idx_into_indices = list(filter(lambda col_idx : indices[i,col_idx] != ignore_idx[i], range(num_retrieve+1)))
                    batch_idx_into_indices[i] = data_point_idx_into_indices
                # exclude all exactly correct explanations
                if self.exclude_correct_explanations:
                    current_index = self.get_index_from_seq(input_ids[i])
                    exclude_train_idx = self.index_to_train_idx[current_index]
                    data_point_idx_into_indices = []
                    counter = 0
                    while len(data_point_idx_into_indices) < num_retrieve:
                        if indices[i,counter] not in exclude_train_idx:
                            data_point_idx_into_indices.append(counter)
                        counter+=1
                    batch_idx_into_indices[i] = data_point_idx_into_indices
        else:
            batch_idx_into_indices = np.stack([range(num_retrieve) for i in range(batch_size)])
        return_idx = np.stack([indices[i, batch_idx_into_indices[i]] for i in range(batch_size)])       
        return return_idx

    def get_retrieval_scores(self, *args, **kwargs):
        if self.use_optimal_retrieval:
            return self.optimal_retrieval_scores(*args, **kwargs)
        else:
            return self.compute_retrieval_scores(*args, **kwargs)

    def compute_retrieval_scores(self, input_ids, attention_mask, retrieval_idx):
        '''
        return retrieval scores for input_ids against retrieval_idx points, of shape batch_size x k
        - input_ids : shape batch_size x seq_len
        - retrieval_idx : list of idx of shape batch_size x k x context_size
        '''
        # import pdb; pdb.set_trace()
        num_retrieve = self.top_k * self.context_size
        retrieval_idx = retrieval_idx.reshape(input_ids.size(0), num_retrieve)
        input_representations = self.get_representations(input_ids, attention_mask=attention_mask)
        # look up and embed neighoring data. will ALWAYS be individual sequences, even if x/y/e combined into larger context sets in p(y|x,{e})        
        # NOTE this would be more efficient with the next two lines for memory management, but something in torch backfires and its much less efficient
        # neighbor_grad_req = torch.enable_grad() if self.backprop_targets else torch.no_grad()
        # with torch.enable_grad():
        neighbor_ids = torch.tensor(
            [[self._pad_to_length(self.encode_idx(idx.item()), pad_token_id=self.retriever_tokenizer.pad_token_id, max_length=self.max_x_len) for idx in row] for row in retrieval_idx], 
        ) # shape: batch_size x k*context_size x seq len
        neighbor_ids = neighbor_ids.view(-1, neighbor_ids.size(-1)).to(self.device) # reshape for forward pass to model
        neighbor_mask = (neighbor_ids!=self.retriever_tokenizer.pad_token_id).float().to(self.device)
        neighbor_representations = self.get_representations(neighbor_ids, attention_mask=neighbor_mask)
        # DETACH neighbor representations. these are targets only
        if not self.backprop_targets:
            neighbor_representations = neighbor_representations.detach()
        # repeat input_representations to align with neighbor_reps
        input_representations = input_representations.repeat_interleave(num_retrieve, dim=0)
        # get similarity scores        
        retrieval_scores = self.compute_similarity(input_representations, neighbor_representations)
        # reshape to bs x k x context_size
        retrieval_scores = retrieval_scores.reshape(input_ids.size(0), self.top_k, self.context_size)
        retrieval_scores = retrieval_scores.sum(dim=-1)
        retrieval_scores = torch.exp(retrieval_scores)
        return retrieval_scores

    def optimal_retrieval_scores(self, input_ids, attention_mask, retrieval_idx):
        # for use with synthetic data, computes distance on idx in sequences
        # see compute_retrieval_scores for line by line documentation
        num_retrieve = self.context_size * self.top_k
        retrieval_idx = retrieval_idx.reshape(input_ids.size(0), num_retrieve)
        input_idx = torch.tensor([self.get_index_from_ids(seq) for seq in input_ids])
        neighbor_ids = torch.tensor(
            [[self._pad_to_length(self.make_single_sequence(idx.item(), add_special_tokens=True), pad_token_id=self.retriever_tokenizer.pad_token_id) for idx in row] for row in retrieval_idx], 
        ) # shape: batch_size x k*context_size x seq len
        neighbor_ids = neighbor_ids.view(-1, input_ids.size(-1)).to(self.device) # reshape for picking out seq idx
        neighbor_idx = torch.tensor([self.get_index_from_ids(seq) for seq in neighbor_ids])
        input_idx = input_idx.repeat_interleave(num_retrieve, dim=0)
        scores = self.similarity_function(input_idx, neighbor_idx)
        scores = scores.reshape(input_ids.size(0), self.top_k, self.context_size)
        scores = scores.sum(dim=-1)
        scores = torch.exp(scores)
        return scores.to(self.device)

    def optimal_retrieval(self, input_ids, num_retrieve):
        # returns list of lists of idx of data points that are CLOSE to input sequence on the index token (s[0])
        def _retrieve_single(input_ids):
            return_idx = []
            index = self.get_index_from_ids(input_ids)
            stripped_input = [_id for _id in input_ids[1:] if _id != self.retriever_tokenizer.pad_token_id and _id != self.retriever_tokenizer.sep_token_id] # ignore cls, pad, sep token ids
            # argsort indices by distance
            sort_idx = np.argsort([np.abs(candidate-index) for candidate in self.used_data_idx])
            idx_counter = 0
            while len(return_idx) < num_retrieve: 
                get_idx = self.used_data_idx[sort_idx[idx_counter]]
                data_at_idx = self.idx_to_data[get_idx]
                for data_point in data_at_idx:
                    data_idx = data_point['idx'] # index/location in self.*_ids
                    x = data_point['x']
                    not_same_data_point = not np.all(np.array(stripped_input) == np.array(x))
                    if not_same_data_point and len(return_idx) < num_retrieve:
                        return_idx.append(data_idx)
                idx_counter+=1
            return return_idx
        # retrieve for each input sequence
        return_idx = []
        for sequence_ids in input_ids.tolist():
            return_idx.append(_retrieve_single(sequence_ids))
        return return_idx

    def get_index_from_seq(self, input_ids) -> int:
        # get the special index number from the seq. index is the link between x and e
        str_ids = self.retriever_tokenizer.decode(input_ids[:6], skip_special_tokens=True) # assume first number not encoded by more than a few tokens
        index = int(str_ids.split()[0])
        return index

    def get_index_from_ids(self, input_ids) -> int:
        # get the special index number from the input. index is the link between x and e
        str_ids = self.retriever_tokenizer.decode(input_ids, skip_special_tokens=True) # assume first number not encoded by more than a few tokens
        strs = str_ids.split()
        # cut out tokens based on what the prefixes are
        if strs[0] == 'x:' or strs[0] == 'e:':
            strs = strs[1:]
        if strs[0] == 'y:':
            strs = strs[2:] # cut out y and label
        index = int(strs[0])
        return index

    def make_single_sequence(self, idx, add_special_tokens=False):
        # for making retriever model inputs given a data point index from the training data
        ids = []
        x, y, e = self.sequence_ids[idx], self.label_ids[idx], self.explanation_ids[idx]
        if 'X' in self.context_includes:
            ids.extend(self.x_prefix + x)
        if 'Y' in self.context_includes:
            ids.extend(self.y_prefix + y)
        if 'E' in  self.context_includes:
            ids.extend(self.e_prefix + e)
        if add_special_tokens:
            ids = [self.retriever_tokenizer.cls_token_id] + ids + [self.retriever_tokenizer.sep_token_id]
        return ids

    def make_task_input(self, idx, add_special_tokens=False):
        # for use in make_new_data. see encode_input for encoding for retriever forward pass
        ids = []
        x, y, e = self.task_sequence_ids[idx], self.task_label_ids[idx], self.task_explanation_ids[idx]
        if 'X' in self.context_includes:
            ids.extend(self.task_x_prefix + x)
        if 'Y' in self.context_includes:
            ids.extend(self.task_y_prefix + y)
        if 'E' in  self.context_includes:
            ids.extend(self.task_e_prefix + e)
        if add_special_tokens:
            ids = [self.task_tokenizer.cls_token_id] + ids + [self.task_tokenizer.sep_token_id]
        return ids        

    def make_new_data(self, orig_input_ids, orig_attention_mask, retrieve_idx) -> dict:
        '''
        orig_input_ids : shape batch_size x seq_len
        retrieve_idx : shape batch_size x k x context_size
        make dict of model kwargs

        and will have to split these up by what to condition on
        '''
        task_tokenizer=self.task_tokenizer
        pad_token_id = self.task_tokenizer.pad_token_id
        sep_token_id = self.task_tokenizer.sep_token_id
        seq_lens = torch.sum(orig_attention_mask, dim=-1)

        # make input_ids of shape batch_size x k x seq_len
        if self.use_textcat:
            orig_input_ids = orig_input_ids.unsqueeze(1).repeat(1, self.top_k, 1) 
            for i in range(orig_input_ids.size(0)):
                for k in range(self.top_k):
                    input_len = int(seq_lens[i].item())
                    context_ids_list = [self.make_task_input(idx) for idx in retrieve_idx[i, k]]
                    self._truncate_seq_set(context_ids_list, max_length=self.max_seq_len-input_len-len(context_ids_list)) # leave room for input_len, and len(context_ids_list) sep tokens
                    context_ids = self._combine_lists(context_ids_list)
                    orig_input_ids[i, k, input_len:(input_len+context_ids.size(-1))] = context_ids
            return_dict = {
                'input_ids' : orig_input_ids,
                'attention_mask': (orig_input_ids!=pad_token_id).float()
            }

        # make input_ids of shape batch_size x k x context_size x seq_len
        if self.use_ELV:
            extended_input = orig_input_ids.unsqueeze(1).unsqueeze(1).repeat(1, self.top_k, self.context_size, 1)
            for i in range(orig_input_ids.size(0)):
                input_len = int(seq_lens[i].item())
                for k in range(self.top_k):
                    context_ids_list = [self.make_task_input(idx) for idx in retrieve_idx[i,k]]
                    context_ids_list = [self._pad_to_length(seq + [sep_token_id], max_length=self.max_seq_len-input_len, pad_token_id=task_tokenizer.pad_token_id) for seq in context_ids_list]
                    for context_number, seq in enumerate(context_ids_list):
                        extended_input[i,k,context_number,input_len:] = torch.tensor(seq)
            attention_mask = (extended_input!=pad_token_id).float()
            return_dict = {
                'input_ids' : extended_input,
                'attention_mask': attention_mask,
            }

        return return_dict


    def _truncate_seq_set(self, tokens_list, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        assert max_length >= 1, "max length is problematic in _truncate_seq_set (below 1)"
        while True:
            lens = [len(tokens) for tokens in tokens_list]            
            total_length = sum(lens)
            if total_length <= max_length:
                break
            longest = np.argmax(lens)
            tokens_list[longest].pop()

    def _combine_lists(self, tokens_list) -> torch.Tensor:
        return_ids = []
        for tokens in tokens_list:
            return_ids.extend(tokens + [self.task_tokenizer.sep_token_id])
        return torch.tensor(return_ids)

    def _pad_seq(self, seq, length, pad_id):
        seq += [pad_id] * (length-len(seq))

    def _truncate_seq(self, tokens, max_length):
        while len(tokens) > max_length:
            tokens.pop()

    def _pad_to_length(self, tokens, max_length=None, pad_token_id=1):
        # either in-place or returned
        max_len = max_length if max_length is not None else self.max_seq_len
        self._pad_seq(tokens, max_len, pad_token_id)
        self._truncate_seq(tokens, max_len)
        return tokens










