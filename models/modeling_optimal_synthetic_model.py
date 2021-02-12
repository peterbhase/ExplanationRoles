import torch
from torch import nn
from transformers import AutoTokenizer
import numpy as np

class OptimalSyntheticModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
            
    def count_num_in_seq(self, seq, num):
        return len([n for n in seq if n==num])

    def forward(self, input_ids, **kwargs):
        '''
        this is the true model of the synthetic data
        - coded with primarily list comps
        - return fake loss, logits at the end
        '''
        input_strs = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in input_ids]
        input_lists = [input_str.split() for input_str in input_strs]
        mn_or_rd_list = [input_list[1] for input_list in input_lists]
        count_idx = [[-4,-3] if mn_or_rd=='1' else [-2,-1] for mn_or_rd in mn_or_rd_list]
        nums_to_count = [[input_list[idx] for idx in count_idx] for input_list, count_idx in zip(input_lists, count_idx)]
        num_counts = [[self.count_num_in_seq(seq, num) for num in nums] for seq, nums in zip(input_lists, nums_to_count)]
        preds = [1*(nums[0] > nums[1]) for nums in num_counts]
        # make fake logits
        fake_logits = torch.zeros(len(input_ids), 2)
        for i in range(len(fake_logits)):
            idx = int(preds[i])
            fake_logits[i,idx] = 1
        loss = torch.tensor(-1.)
        outputs = (loss, fake_logits)
        
        return outputs

