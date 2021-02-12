import numpy as np
import pandas as pd
import utils
import os

def write_synthetic_data(args):
    '''
    make data where data points are (y,s,e)
    - s : sequence of integers
    - idx : index which will be s[0]. depending on condition, there will exist a map f : s_idx --> e_idx that allows for retrieval-based approaches
    - m,n are ints
    - r,d are ints
    - y : labels given as 1[#m > #n] in the vanilla setup. in one condition, this holds if s[1]==1 else 1[#r> #d]
    - e : is an "explanation" of the data point, which gives information the features that cause the label

    NUMBERS START AT 1. no zeros appear in s
    '''
    print("Writing data...")
    if args.num_relevant_points < 0 and args.num_tasks < 0: # is neither parameter provided...
        args.num_relevant_points = args.context_size + 1
    # DETERMINE NUM_RELEVANT_POINTS HERE IF NOT PROVIDED
    if args.num_relevant_points < 0 and args.num_tasks > 0:
        assert args.num_train_synthetic % args.num_tasks == 0, "please make n_train divisible by num_tasks"
        args.num_relevant_points = args.num_train_synthetic // args.num_tasks
    n_train = args.num_train_synthetic
    n_dev = 10000 # NOTE these may be slightly modified to allow for even group sizes wrt k
    n_test = 50000 # NOTE these may be slightly modified to allow for even group sizes wrt k
    if args.small_data:
        n_train, n_dev, n_test = [int(args.small_size) for n in range(3)]
        if n_train % args.num_relevant_points != 0:
            n_train = ((n_train // args.num_relevant_points + 1) * args.num_relevant_points)            
    # slightly modify n_train if needed
    if n_train % args.num_relevant_points != 0:
        n_train = ((n_train // args.num_relevant_points + 1) * args.num_relevant_points)
    assert n_train % (args.num_relevant_points) == 0, "please make n_train divisible by num_relevant_points"
    if args.num_relevant_points > 1:
        if not args.num_relevant_points % 2 == 0:
            print("\n Note that num_relevant_points is odd! Hence balancing is not precisely 50/50 \n")
    
    # get train_use_idx. 
    num_per_train_idx = args.num_relevant_points
    n_train_idx = n_train // num_per_train_idx
    max_idx = args.max_int**2 if args.max_idx < 0 else args.max_idx
    assert n_train_idx <= max_idx, "need to decrease num_relevant_points to increase the numbers of tasks, or increase the num possible tasks by increasing args.max_int"
    train_use_idx = np.random.choice(np.arange(1,max_idx+1), size=n_train_idx, replace=False)
    # test time idx are seen in training by default, or can be flagged to make them new
    if not args.disjoint_test_idx:
        dev_use_idx = train_use_idx
        test_use_idx = train_use_idx
    elif args.disjoint_test_idx:
        eligible_idx = np.setdiff1d(np.arange(1, max_idx+1), train_use_idx)
        dev_use_idx = np.random.choice(eligible_idx, size=n_train_idx, replace=True) # will need to replace=True when n_test > max_idx*num_per_idx
        test_use_idx = dev_use_idx
    num_per_dev_idx = round(n_dev / len(dev_use_idx))
    num_per_test_idx = round(n_test / len(test_use_idx))

    # modify n_dev and n_test if needed
    n_dev += (len(dev_use_idx) * num_per_dev_idx - n_dev)
    n_test += (len(test_use_idx) * num_per_test_idx - n_test)

    # make labels in advance
    labels_list = [utils.balanced_array(size=n, prop=.5) for n in [n_train, n_dev, n_test]]
    utils.shuffle_lists(labels_list)
    train_labels, dev_labels, test_labels = labels_list    

    # make mn and rds. make idx to z dict
    max_mn = int(np.sqrt(max_idx))
    mn_and_rds, collected = [], []

    # here is the normal procedure: (See below for special case)
    order_counter = 123 # unique ints to start. will start at 0123 -> 1234 given +1
    unique_idx = set(np.concatenate([train_use_idx, dev_use_idx])) # by default this just turns into train_use_idx
    if not (args.use_mn_only and args.ordered_mnrd):
        while len(mn_and_rds) < len(unique_idx):
            # in this condition, randomly sample mnrd and simply avoid repeats
            if not args.ordered_mnrd:
                proposal = np.random.choice(np.arange(1, max_mn+1), size=4, replace=False)
                if str(proposal) not in collected: # weird truth value ambiguous just checking if proposal in mn_and_rds
                    mn_and_rds.append(proposal)
                    collected.append(str(proposal))

            # in this condition, gradually increment values of m/n/r/d so that the task information is dense in integer space
            if args.ordered_mnrd:
                assert max_idx <= 10000, "right now ordered_mnrd without use_mn_only has a max_idx of 10k"
                str_mnrd = '%04d' % order_counter
                while len(set(str_mnrd)) != len(list(str_mnrd)): # if list is not unique characters
                    order_counter += 1
                    str_mnrd = '%04d' % order_counter
                mnrd = np.array([int(_int) + 1 for _int in list(str_mnrd)])
                mn_and_rds.append(mnrd)
                order_counter += 1

    # this is a special condition where we need to order based on mn only, so we overwrite the above
    if args.use_mn_only and args.ordered_mnrd:
        mn_and_rds, collected = [], []
        integers = np.array([1,1])
        while len(mn_and_rds) < len(unique_idx):
            proposal = integers
            # increment if not all integers unique
            while len(set(proposal)) != len(proposal):
                last_idx_where_valid = [idx for idx in range(len(integers)) if integers[idx] < 100][-1]
                integers[last_idx_where_valid] += 1
            mn = np.array([int(_int) for _int in integers])
            distractors = np.random.choice(np.setdiff1d(np.arange(1,101), mn), size=2, replace=False)
            mnrd = np.concatenate([mn, distractors])
            mn_and_rds.append(mnrd)
            # increment
            last_idx_where_valid = [idx for idx in range(len(integers)) if integers[idx] < 100][-1]
            if last_idx_where_valid != 1:
                integers[-1] = 1
            integers[last_idx_where_valid] += 1

    # order things if doing smooth_idx_to_z
    if args.smooth_idx_to_z:
        train_use_idx = np.sort(train_use_idx)
        mn_and_rds = sorted(mn_and_rds, 
                        key = lambda x : x[0] + 1e-3 * x[1] + 1e-6 * x[2] + 1e-9 * x[3]) # this takes advantage of known scale of num_tasks to break ties by each next element of the mnrd array

    idx_to_z_dict = {idx : mn_and_rds[i] for i, idx in enumerate(unique_idx)}
    
    '''
    now want a few other properties, per idx per dataset
    - mn or rd balance: use_mn_or_rd within each idx
    - #counts balance: want mn/rd #-counts to swap half the time, so there is no bias in size
    - distractor feature: want the non-causal feature (mn or rd, depending on above indicator) to correlate with the causal one 50% of the time
    '''
    train_idx_to_info = {idx : 
        {'use_mn_or_rd' : utils.balanced_array(size=num_per_train_idx, prop=.5), # pick whether to use mn, or rd for 
         'swap_samples' : utils.balanced_array(size=num_per_train_idx, prop=.5), # set mnrd = (count1,2,3,4) or mnrd = (count3,4,1,2) based on this (whether to swap counts 1,2 and 3,4)
         'distractor_correlates' : utils.balanced_array(size=num_per_train_idx, prop=args.weak_feature_correlation), # whether to have the non-causal feature (mn or rd) correlate with the causal one
         'mnrd' : idx_to_z_dict[idx]
        } for idx in train_use_idx
    }
    dev_idx_to_info = {idx : 
        {'use_mn_or_rd' : utils.balanced_array(size=num_per_dev_idx, prop=.5), # pick whether to use mn, or rd for 
         'swap_samples' : utils.balanced_array(size=num_per_dev_idx, prop=.5), # set mnrd = (count1,2,3,4) or mnrd = (count3,4,1,2) based on this (whether to swap counts 1,2 and 3,4)
         'distractor_correlates' : utils.balanced_array(size=num_per_dev_idx, prop=.5), # whether to have the non-causal feature (mn or rd) correlate with the causal one
         'mnrd' : idx_to_z_dict[idx]
        } for idx in dev_use_idx
    }
    test_idx_to_info = {idx : 
        {'use_mn_or_rd' : utils.balanced_array(size=num_per_test_idx, prop=.5), # pick whether to use mn, or rd for 
         'swap_samples' : utils.balanced_array(size=num_per_test_idx, prop=.5), # set mnrd = (count1,2,3,4) or mnrd = (count3,4,1,2) based on this (whether to swap counts 1,2 and 3,4)
         'distractor_correlates' : utils.balanced_array(size=num_per_test_idx, prop=.5), # whether to have the non-causal feature (mn or rd) correlate with the causal one
         'mnrd' : idx_to_z_dict[idx]
        } for idx in test_use_idx
    }

    # make splits
    train_s_list, train_e_list = make_split(args, train_labels, train_use_idx, num_per_train_idx, train_idx_to_info, ignore_list = None)
    dev_s_list,   dev_e_list   = make_split(args, dev_labels,   dev_use_idx,   num_per_dev_idx,   dev_idx_to_info,   ignore_list = train_s_list)
    test_s_list,  test_e_list  = make_split(args, test_labels,  test_use_idx,  num_per_test_idx,  test_idx_to_info,  ignore_list = train_s_list)

    assert len(train_s_list) == n_train
    assert len(dev_s_list) == n_dev

    # make dfs and write splits
    train_df = pd.DataFrame({
            'unique_id' : i,
            's' : train_s_list[i],
            'e' : train_e_list[i],
            'label' : train_labels[i]
        } for i in range(n_train))
    dev_df = pd.DataFrame({
            'unique_id' : i+n_train,
            's' : dev_s_list[i],
            'e' : dev_e_list[i],
            'label' : dev_labels[i]
        } for i in range(n_dev))
    test_df = pd.DataFrame({
            'unique_id' : i+n_train+n_dev,
            's' : test_s_list[i],
            'e' : test_e_list[i],
            'label' : test_labels[i]
        } for i in range(n_test))
    folder = args.data_dir + '_' + args.experiment_name
    if not os.path.exists(folder): os.mkdir(folder)
    paths = [os.path.join(folder, split_name) + '.csv' for split_name in ['train','dev','test']]
    train_df.to_csv(paths[0], index=False)
    dev_df.to_csv(paths[1], index=False)
    test_df.to_csv(paths[2], index=False)

    print("\nData statistics:")
    print(f"\t Num train idx / tasks: {len(train_use_idx)} | Num per train idx: {num_per_train_idx}")
    print(f"\t Num dev idx / tasks:   {len(dev_use_idx)} | Num per dev idx:   {num_per_dev_idx}")
    print(f"\t Num test idx / tasks:  {len(test_use_idx)} | Num per test idx:  {num_per_test_idx}")

    return train_use_idx
    
def make_split(args, labels, use_idx, num_per_idx, idx_to_info, ignore_list=None):
    '''
    make data split. go through idx, num_per_idx points for each.
    - explanations get made for dev and test splits, but only training explanations are used in experiments
    '''
    s_list = []
    e_list = []
    s_len = 20
    skipped_samples = 0
    idx_function = get_idx_function(args.idx_function, max_idx=args.max_int**2) # map between x and e idx.
    for idx_num, idx in enumerate(use_idx):
        # print(f"at {idx_num}, have {idx}")
        added_for_idx = 0
        mnrd = idx_to_info[idx]['mnrd']
        # handle recomposable here, handle evidential inside of make_explanation
        if args.explanation_kind == 'recomposable': 
            # get piece id for each data point to be made for given idx
            num_piece_ids = args.recomposable_pieces
            # decompose mnrd across num_piece_ids
            off_elements_zero = (not args.recomposable_additive)
            decomposed_mnrd = utils.decompose_integers(mnrd, num_parts=num_piece_ids, off_elements_zero=off_elements_zero)
            # get piece ids for each data point to be made for this idx
            piece_ids = np.concatenate([np.arange(1,num_piece_ids+1), np.random.randint(1, num_piece_ids+1, size=num_per_idx-num_piece_ids)])
            # slice into decomposed mnrd to get mnrd for each data point
            recomposable_mnrd = np.stack([decomposed_mnrd[piece_id-1] for piece_id in piece_ids], axis=0)
            if not off_elements_zero:
                recomposable_mnrd = np.concatenate([recomposable_mnrd, piece_ids.reshape(-1,1)], axis=1)
        while added_for_idx < num_per_idx:
            point_idx = idx_num*num_per_idx + added_for_idx
            label = labels[point_idx]
            # fill in first two elements and get free idx
            s = np.zeros(s_len)
            s[0] = idx if args.use_index else 1            
            s[1] = idx_to_info[idx]['use_mn_or_rd'][added_for_idx]+1 if args.use_mn_indicator else 1 # +1 means 0->1 and 1->2 from balanced_array
            free_slot_idx = np.argwhere(s==0).reshape(-1)
            # get mn counts
            free_mn_slots = s_len - 2 - 3 # 2 for s[0],s[1], save 3 slots for r/d
            mn_count1 = np.random.randint(1,free_mn_slots) # don't take all free_mn_slots, save room for mn_count2
            mn_count2 = np.random.randint(0, min(mn_count1, free_mn_slots-mn_count1)) # the min is so you can't take more than min(#available,#count1)
            mn_idx1 = np.random.choice(free_slot_idx, size=mn_count1, replace=False)
            free_slot_idx = np.setdiff1d(free_slot_idx, mn_idx1)
            mn_idx2 = np.random.choice(free_slot_idx, size=mn_count2, replace=False)
            free_slot_idx = np.setdiff1d(free_slot_idx, mn_idx2)
            # get rd counts
            free_rd_slots = len(free_slot_idx)
            rd_count1 = np.random.randint(1,free_rd_slots) # don't take all free_rd_slots, save room for rd_count1
            rd_count2 = np.random.randint(0, min(rd_count1, free_rd_slots-rd_count1)) # the min is so you can't take than min(#available,#count1)
            rd_idx1 = np.random.choice(free_slot_idx, size=rd_count1, replace=False)
            free_slot_idx = np.setdiff1d(free_slot_idx, rd_idx1)
            rd_idx2 = np.random.choice(free_slot_idx, size=rd_count2, replace=False)
            free_slot_idx = np.setdiff1d(free_slot_idx, rd_idx2)

            if idx_to_info[idx]['swap_samples'][added_for_idx]:
                mn_idx1, mn_idx2, rd_idx1, rd_idx2 = rd_idx1, rd_idx2, mn_idx1, mn_idx2

            # assign mn and rd idx
            features_correlate = idx_to_info[idx]['distractor_correlates'][added_for_idx]
            mn_idx = np.array([mn_idx1, mn_idx2]).reshape(-1)
            rd_idx = np.array([rd_idx1, rd_idx2]).reshape(-1)
            use_mn = (s[1] == 1) or args.use_mn_only
            if label==1 and use_mn:
                mn_order = [0,1]
                rd_order = [0,1] if features_correlate else [1,0]
            if label==0 and use_mn:
                mn_order = [1,0]
                rd_order = [1,0] if features_correlate else [0,1]
            if label==1 and not use_mn:
                rd_order = [0,1]
                mn_order = [0,1] if features_correlate else [1,0]
            if label==0 and not use_mn:
                rd_order = [1,0]
                mn_order = [1,0] if features_correlate else [0,1]
            m_idx, n_idx = mn_idx[mn_order]
            r_idx, d_idx = rd_idx[rd_order]
            
            # get m,n,r,d and put into s.
            s[m_idx] = mnrd[0]
            s[n_idx] = mnrd[1]
            s[r_idx] = mnrd[2]
            s[d_idx] = mnrd[3]
            # fill empty spaces with random ints (not equal to mnrd)!
            where_empty = np.argwhere(s==0).reshape(-1)
            eligible_ints = np.setdiff1d(np.arange(1, args.max_int+1), mnrd)
            random_ints = np.random.choice(eligible_ints, len(where_empty), replace=True)
            s[where_empty] = random_ints
            # check that seq is full
            assert not any(s == 0)
            
            # add missing info to s in single-task scenario, where task solvable as y=f(x)
            if args.single_task:
                s = np.concatenate((s, mnrd))

            # make s into str and accumulate if not ignored
            s_str = ' '.join([str(int(num)) for num in s])
            if ignore_list is not None and s_str in ignore_list:
                skipped_samples += 1
                continue
            else:
                # make explanations
                use_mnrd = mnrd if not args.explanation_kind == 'recomposable' else recomposable_mnrd[added_for_idx]
                e_str = make_explanation(args, s, idx, use_mnrd, label, idx_function)
                # accumulate data
                s_list.append(s_str)
                e_list.append(e_str)
                added_for_idx += 1

    if skipped_samples > 0:
        print(f"skipped {skipped_samples} samples!")

    return s_list, e_list

def make_explanation(args, s, idx, mnrd, label, idx_function) -> str:
    '''
    make an explanation for a data point.
    - recomposable: (idx, m, n, r, d, p) is the pth piece of an explanation, which is recomposable by sum_pieces(e)
    - evidential: (idx, m, n, r, d) where mnrd all have unif +/- eps discrete noise added to them
    '''
    use_mn_or_rd = s[1]
    e_idx = idx_function(idx)
    if args.explanation_kind == 'missing_info' or args.explanation_kind == 'recomposable':
        explanation = [e_idx] + list(mnrd)
    if args.explanation_kind == 'evidential':
        eps = args.evidential_eps
        noise = np.random.randint(-eps,eps+1, size=4)
        mnrd = mnrd + noise
        explanation = [e_idx] + list(mnrd)
    if args.explanation_only_causal:
        keep_idx = [0] + ([1,2] if use_mn_or_rd==1 else [3,4])
        explanation=np.array(explanation)[keep_idx]
    if args.translate_explanation:
        explanation=translate_explanation(args, e=np.array(explanation))
    explanation = ' '.join([str(int(num)) for num in explanation])
    return explanation

def translate_explanation(args, e):
    if args.translate_explanation == '100minusx':
        e[1:] = 101 - e[1:]
    elif args.translate_explanation == 'xplus5':
        e[1:] = 5 + e[1:]
    return e

def get_idx_function(idx_function, max_idx):
    x = np.arange(1,max_idx+1)
    if idx_function == 'identity':
        y = x.copy()
    if idx_function == 'easy':
        y = max_idx-x
    if idx_function == 'hard':
        # interleave y=x and y=max_idx-x based on whether x is ever or odd
        y = [num if num%2==0 else max_idx-num for num in x]
    if idx_function == 'noise':
        y = np.arange(max_idx)
        np.random.shuffle(y)
    x_to_y = {x:y for x,y in zip(x,y)}
    def idx_function(x):
        return x_to_y[x]
    return idx_function

