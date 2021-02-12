import os
import argparse
import utils

# --- SemEval experiments --- #

def semeval_baseline(args):
    experiment = 'semeval_baseline' + ('_DEBUG' if args.small_data else '')
    utils.make_experiment_sheet(experiment=experiment, params=['seed','model','n'], num_seeds=len(seeds))
    for seed in seeds:
        for model in ['roberta-large', 'roberta-large']:
            for n in [5000, -1]:
                data_addin = f'--use_num_train {n}' if n > 0 else ''
                os.system(f"python main.py --gpu {args.gpu} --data_dir data/semeval --train_batch_size {tbs} --grad_accumulation_factor {gaf} --num_train_epochs 20  "
                          f" --use_retrieval false --model {model} --token_pooling ent_idx {data_addin} "
                          f"{small_data_addin} --seed {seed} --server {server} --experiment_name {experiment}_{model}_seed{seed} "
                )
                utils.write_experiment_result(experiment=experiment, params={'seed' : seed, 'model' : model})

def semeval_textcat(args):
    experiment = 'semeval_textcat' + ('_DEBUG' if args.small_data else '')
    utils.make_experiment_sheet(experiment=experiment, params=['seed','model','n','c','k'], num_seeds=len(seeds))
    for seed in seeds:
        for model in ['roberta-base']:
            for params in [(2,4), (1,8)]:
                c,k = params
                for n in [5000, -1]:
                    if n==5000:
                        epochs = 20
                    if n==-1: 
                        epochs=15
                    data_addin = f"--use_num_train {n}" if n>0 else ''
                    tbs = 2 if params==(2,4) else 1
                    gaf = 5 if params==(1,8) else 10
                    os.system(f"python main.py --gpu {args.gpu} --data_dir data/semeval --train_batch_size {tbs} --grad_accumulation_factor {gaf} --num_train_epochs {epochs} --model {model} "
                              f"--use_textcat true --use_retrieval true --train_retriever true --context_size {c} --top_k {k} --model {model} --rebuild_index_every_perc .2 --token_pooling ent_idx "
                              f"--max_seq_len 160 --max_e_len 60 --max_x_len 80 --warmup_proportion .01 --freeze_retriever_until 1 {data_addin} --context_includes YE "
                              f"{small_data_addin} --seed {seed} --server {server} --experiment_name {experiment}_{model}_n{n}_c{c}_k{k}_seed{seed} "
                    )
                    utils.write_experiment_result(experiment=experiment, params={'seed' : seed, 'model' : model, 'n' : n, 'c' : c, 'k' : k})

def semeval_textcat_by_context(args):
    experiment = 'semeval_textcat_by_context' + ('_DEBUG' if args.small_data else '')
    utils.make_experiment_sheet(experiment=experiment, params=['seed', 'context'], num_seeds=len(seeds))
    parameters = [('YE',1), ('YXE',1), ('YX',1)]
    for seed in seeds:
        for (context, context_size) in parameters:
            os.system(f"python main.py --gpu {args.gpu} --data_dir data/semeval --token_pooling ent_idx --train_batch_size {tbs} --grad_accumulation_factor {gaf} --num_train_epochs 15 --model roberta-base "
                      f" --max_seq_len 150 --max_x_len 80 --use_textcat true --use_retrieval true --rebuild_index_every_epoch true --train_retriever true --freeze_retriever_until 1 --context_includes {context} --context_size {context_size} --top_k 4 "
                      f"{small_data_addin} --seed {seed} --server {server} --experiment_name {experiment}_c{context}_seed{seed} --print true "
            )
            utils.write_experiment_result(experiment=experiment, params={'seed' : seed, 'context' : context})

# --- TACRED experiments --- #

def tacred_baseline(args):
    experiment = 'tacred_baseline' + ('_DEBUG' if args.small_data else '')
    utils.make_experiment_sheet(experiment=experiment, params=['seed','model','n'], num_seeds=len(seeds))
    for seed in seeds:
        for model in ['roberta-large', 'roberta-large']:
            for n in [5000,10000, -1]:
                epochs = 20 if n>0 else 10
                data_addin = f'--use_num_train {n}' if n > 0 else ''            
                os.system(f"python main.py --gpu {args.gpu} --data_dir data/tacred --use_retrieval false --train_batch_size {tbs} --grad_accumulation_factor {gaf} --num_train_epochs {epochs} --max_seq_len 160 --server 4  "
                          f" --use_retrieval false --model {model} --token_pooling ent_idx {data_addin} "
                          f"{small_data_addin} --seed {seed} --server {server} --experiment_name {experiment}_{model}_n{n}_seed{seed} "
                )
                utils.write_experiment_result(experiment=experiment, params={'seed' : seed, 'model' : model, 'n' : n})

def tacred_textcat(args):
    experiment = 'tacred_textcat' + ('_DEBUG' if args.small_data else '')
    utils.make_experiment_sheet(experiment=experiment, params=['seed','model','n','c','k'], num_seeds=len(seeds))
    for seed in seeds:
        for model in ['roberta-base']:
            for params in [(2,4), (1,8)]:
                c,k = params
                for n in [5000, 10000, -1]:
                    if n==5000:
                        epochs = 20
                    if n==10000:
                        epochs=15
                    if n==-1: 
                        epochs=10
                    data_addin = f" --use_num_train {n}" if n>0 else ''
                    os.system(f"python main.py --gpu {args.gpu} --data_dir data/tacred --train_batch_size {tbs} --grad_accumulation_factor {gaf} --num_train_epochs {epochs} --model {model} "
                              f"--use_textcat true --use_retrieval true --train_retriever true --context_size {c} --top_k {k} --model {model} --rebuild_index_every_perc .2 --token_pooling ent_idx "
                              f"--max_seq_len 160 --max_e_len 60 --max_x_len 80 --warmup_proportion .01 --freeze_retriever_until 1 {data_addin} --context_includes YE "
                              f"{small_data_addin} --seed {seed} --server {server} --experiment_name {experiment}_{model}_n{n}_c{c}_k{k}_seed{seed} "
                    )
                    utils.write_experiment_result(experiment=experiment, params={'seed' : seed, 'model' : model, 'n' : n, 'c' : c, 'k' : k})


def tacred_textcat_by_context(args):
    experiment = 'tacred_textcat_by_context' + ('_DEBUG' if args.small_data else '')
    utils.make_experiment_sheet(experiment=experiment, params=['seed', 'context'], num_seeds=len(seeds))
    parameters = [('YE',1), ('YXE',1), ('YX',1)]
    for seed in seeds:
        for (context, context_size) in parameters:
            os.system(f"python main.py --gpu {args.gpu} --data_dir data/tacred --token_pooling ent_idx --train_batch_size {tbs} --grad_accumulation_factor {gaf} --num_train_epochs 10 --model roberta-base "
                      f" --max_seq_len 150 --max_x_len 80 --use_textcat true --use_retrieval true --rebuild_index_every_perc .3 --train_retriever true --freeze_retriever_until 1 --context_includes {context} --context_size {context_size} --top_k 4 "
                      f"{small_data_addin} --seed {seed} --server {server} --experiment_name {experiment}_c{context}_seed{seed} --print true "
            )
            utils.write_experiment_result(experiment=experiment, params={'seed' : seed, 'context' : context})

# --- eSNLI experiments --- #

def esnli_baseline(args):
    experiment = 'esnli_baseline' + ('_DEBUG' if args.small_data else '')
    utils.make_experiment_sheet(experiment=experiment, params=['seed','model','n'], num_seeds=len(seeds))
    for seed in seeds:
        for model in ['roberta-base', 'roberta-large']:
            for n in [5000,10000,-1]:
                epochs = 20 if n>0 else 10
                data_addin = f'--use_num_train {n}' if n > 0 else ''            
                os.system(f"python main.py --gpu {args.gpu} --data_dir data/eSNLI --use_retrieval false --train_batch_size {tbs} --grad_accumulation_factor {gaf} --num_train_epochs {epochs} --max_seq_len 120  "
                          f" --use_retrieval false --model {model} {data_addin} "
                          f"{small_data_addin} --seed {seed} --server {server} --experiment_name {experiment}_{model}_n{n}_seed{seed} "
                )
                utils.write_experiment_result(experiment=experiment, params={'seed' : seed, 'model' : model, 'n' : n})


def esnli_textcat(args):
    experiment = 'esnli_textcat' + ('_DEBUG' if args.small_data else '')
    utils.make_experiment_sheet(experiment=experiment, params=['seed','model','n','c','k'], num_seeds=len(seeds))
    for seed in seeds:
        for model in ['roberta-base']:
            for params in [(4,4), (2,8)]:
                c,k = params
                for n in [5000, 10000, 50000, -1]:
                    if n==5000:
                        epochs = 20
                    if n==10000:
                        epochs=15
                    if n==50000:
                        epochs=10
                    if n==-1: 
                        epochs=5
                    if n==-1 and params == (4,4): 
                        continue
                    os.system(f"python main.py --gpu {args.gpu} --data_dir data/eSNLI --train_batch_size {tbs} --grad_accumulation_factor {gaf} --num_train_epochs {epochs} --model {model} "
                              f"--use_textcat true --use_retrieval true --train_retriever true --context_size {c} --top_k {k} --model {model} --rebuild_index_every_perc .2 "
                              f"--max_seq_len 120 --max_e_len 60 --max_x_len 90 --warmup_proportion .01 --freeze_retriever_until 1 --use_num_train {n} --context_includes E "
                              f"{small_data_addin} --seed {seed} --server {server} --experiment_name {experiment}_{model}_n{n}_c{c}_k{k}_seed{seed} "
                    )
                    utils.write_experiment_result(experiment=experiment, params={'seed' : seed, 'model' : model, 'n' : n, 'c' : c, 'k' : k})


def esnli_textcat_by_context(args):
    experiment = 'esnli_textcat_by_context_tune' + ('_DEBUG' if args.small_data else '')
    utils.make_experiment_sheet(experiment=experiment, params=['seed', 'context'], num_seeds=len(seeds))
    contexts_and_sizes = [('E',1), ('YXE',1), ('YX',1)]
    for seed in seeds:
        for (context, context_size) in contexts_and_sizes:
            train = False if context=='E' else True
            os.system(f"python main.py --gpu {args.gpu} --data_dir data/eSNLI --train_batch_size {tbs} --grad_accumulation_factor {gaf} --num_train_epochs 10 -u .2 --patience 4 --model roberta-base "
                      f" --max_seq_len 180 --use_textcat true --use_retrieval true --rebuild_index_every_epoch true --train_retriever true --freeze_retriever_until 1 --context_includes {context} --context_size {context_size} --top_k 4 "
                      f"{small_data_addin} --seed {seed} --server {server} --experiment_name {experiment}_c{context}_seed{seed} "
            )
            utils.write_experiment_result(experiment=experiment, params={'seed' : seed, 'context' : context})

# --- synthetic data --- #

def memorization_by_n(args):
    experiment = 'memorization_by_n' + ('_DEBUG' if args.small_data else '')
    utils.make_experiment_sheet(experiment=experiment, params=['seed', 'n'], num_seeds=len(seeds))
    for seed in seeds:
        for n in [5000, 10000, 20000, 50000]:
            num_epochs = 40 if n <= 20000 else 20
            os.system(f"python main.py --gpu {args.gpu} --data_dir data/synthetic --train_batch_size {tbs} --grad_accumulation_factor {gaf} --num_train_epochs {num_epochs} --model roberta-base "
                      f" -n {n} --use_retrieval false --context_includes E --num_tasks 500 "
                      f"{small_data_addin} --seed {seed} --server {server} --experiment_name {experiment}_n{n}_seed{seed} "
            )
            utils.write_experiment_result(experiment=experiment, params={'seed' : seed, 'n' : n})

def memorization_by_num_tasks(args): 
    experiment = f'memorization_by_num_tasks' + ('_DEBUG' if args.small_data else '')
    utils.make_experiment_sheet(experiment=experiment, params=['seed', 'model', 'signal_task', 'num_tasks', 'lr'], num_seeds=len(seeds))
    for seed in seeds:
        for model in ['roberta-base', 'roberta-large']:
            for signal_task in [0, 1]:
                use_index = False if signal_task==0 else True
                use_mn_indicator = False if signal_task==0 else True
                single_task=True if signal_task==2 else False
                for num_tasks in [2, 5, 10, 25, 50, 100, 250, 500]:
                    lr = 1e-6 if 'large' in model and num_tasks >= 250 else 1e-5
                    _epochs = 40 if lr==1e-5 else 60
                    os.system(f"python main.py --gpu {args.gpu} --data_dir data/synthetic --train_batch_size {tbs} --grad_accumulation_factor {gaf} --num_train_epochs {_epochs} --model {model} "
                              f" -n 5000 --num_tasks {num_tasks} --use_index {use_index} --use_mn_indicator true --use_retrieval false --context_size 1 --top_k 1 --use_textcat true --lr {lr} "
                              f"{small_data_addin} --seed {seed} --server {server} --experiment_name {experiment}_signal{signal_task}_num_tasks{num_tasks}_seed{seed} --single_task {single_task} "
                    )
                    utils.write_experiment_result(experiment=experiment, params={'seed' : seed, 'model' : model, 'signal_task' : signal_task, 'num_tasks' : num_tasks, 'lr' : lr})

def memorization_by_r_smoothness(args):
    disjoint_test_idx = True
    experiment = 'memorization_by_r_smoothness' + ('_DEBUG' if args.small_data else '')
    utils.make_experiment_sheet(experiment=experiment, params=['seed', 'r', 'smooth', 'disjoint_test_idx'], num_seeds=len(seeds))
    for seed in seeds:
        for r in [1]:
            for smooth in [0,1]:
                max_idx = 10000
                os.system(f"python main.py --gpu {args.gpu} --data_dir data/synthetic --train_batch_size {tbs} --grad_accumulation_factor {gaf} --num_train_epochs 40 --model roberta-base "
                          f" -n 5000 --max_idx {max_idx} --smooth_idx_to_z {smooth} --ordered_mnrd true --use_retrieval false --context_includes E --num_relevant_points {r} --context_size 1 --top_k 1 "
                          f"{small_data_addin} --seed {seed} --server {server} --experiment_name {experiment}_r{r}_smooth{smooth}_seed{seed} --disjoint_test_idx {disjoint_test_idx} "
                )
                utils.write_experiment_result(experiment=experiment, params={'seed' : seed, 'r' : r, 'smooth' : smooth, 'disjoint_test_idx' : disjoint_test_idx})

def memorization_by_seed_test(args):
    experiment = 'memorization_by_seed_test' + ('_DEBUG' if args.small_data else '')
    utils.make_experiment_sheet(experiment=experiment, params=['seed'], num_seeds=len(seeds))
    for seed in seeds:
        os.system(f"python main.py --gpu {args.gpu} --data_dir data/synthetic --train_batch_size {tbs} --grad_accumulation_factor {gaf} --model roberta-base --max_seq_len 120  "
                  f" --use_retrieval false --context_includes E --num_relevant_points 10 --use_textcat true -n 5000 "
                  f" --num_train_epochs 40 "
                  f" {small_data_addin} --seed {seed} --server {server} --experiment_name {experiment}_seed{seed} "
        )
        utils.write_experiment_result(experiment=experiment, params={'seed' : seed})


# --- missing info experiments --- #

def missing_by_learning(args):
    experiment = 'missing_by_learning' + ('_DEBUG' if args.small_data else '')
    utils.make_experiment_sheet(experiment=experiment, params=['seed', 'method', 'condition'], num_seeds=len(seeds))
    for seed in seeds:
        for method in ['textcat','ELV']:
            use_textcat = use_ELV = False
            if method == 'textcat':
                use_textcat = True
                c = 6; k = 6
            if method == 'ELV':
                use_ELV = True
                c = 4; k = 4
            for condition in ['learning','fixed','optimal']:
                use_retrieval = train_retriever = True
                use_optimal_retrieval = False
                if condition == 'fixed':
                    train_retriever=False
                if condition == 'optimal':
                    train_retriever=False
                    use_optimal_retrieval=True
                rb = .2 if train_retriever else -1
                os.system(f"python main.py --gpu {args.gpu} --data_dir data/synthetic --train_batch_size {tbs} --grad_accumulation_factor {gaf} --model roberta-base --max_seq_len 120 "
                          f" --use_retrieval {use_retrieval} --use_optimal_retrieval {use_optimal_retrieval} --train_retriever {train_retriever} --context_includes E --num_relevant_points 10 --context_size {c} --top_k {k} --use_textcat true -n 5000 --rebuild_index_every_perc {rb} "
                          f" --use_textcat {use_textcat} --use_ELV {use_ELV} --freeze_retriever_until 2 --num_train_epochs 20 "
                          f" {small_data_addin} --seed {seed} --server {server} --experiment_name {experiment}_{method}_{condition} "
                )
                utils.write_experiment_result(experiment=experiment, params={'seed' : seed, 'method': method, 'condition' : condition})

def missing_by_rb(args):
    experiment = 'missing_by_rb' + ('_DEBUG' if args.small_data else '')
    utils.make_experiment_sheet(experiment=experiment, params=['seed', 'perc'], num_seeds=len(seeds))
    for seed in seeds:
        for perc in [.1, .2, .333, .5, 1]:
            perc = perc if perc<1 else -1
            every = True if perc==1 else False
            os.system(f"python main.py --gpu {args.gpu} --data_dir data/synthetic --train_batch_size {tbs} --grad_accumulation_factor {gaf} --num_train_epochs 20 --model roberta-base "
                      f" --use_retrieval true --train_retriever true --freeze_retriever_until 1 --context_includes E --num_relevant_points 10 --context_size 4 --top_k 4 --use_textcat true -n 5000 "
                      f" --rebuild_index_every_perc {perc} --rebuild_index_every_epoch {every} "
                      f" {small_data_addin} --seed {seed} --server {server} --experiment_name missing_by_rb_rb{perc} "
            )
            utils.write_experiment_result(experiment=experiment, params={'seed' : seed, 'perc' : perc})

def missing_by_r_smoothness(args):
    disjoint_test_idx = True
    experiment = f'missing_by_r_smoothness' + ('_DEBUG' if args.small_data else '')
    utils.make_experiment_sheet(experiment=experiment, params=['seed', 'r', 'smooth', 'train', 'disjoint'], num_seeds=len(seeds))
    for seed in seeds:
        for r in [1]:
            for smooth in [0, 1]:
                for train in [0, 1]:
                    rb = .2 if train else -1
                    os.system(f"python main.py --gpu {args.gpu} --data_dir data/synthetic --train_batch_size {tbs} --grad_accumulation_factor {gaf} --num_train_epochs 20 --model roberta-base "
                              f" -n 5000 --smooth_idx_to_z {smooth} --ordered_mnrd true --use_retrieval true --train_retriever {train} --context_includes E --num_relevant_points {r} --context_size 1 --top_k 12 "
                              f"  --use_textcat true --freeze_retriever_until 2 --rebuild_index_every_perc {rb} --disjoint_test_idx {disjoint_test_idx} "
                              f"{small_data_addin} --seed {seed} --server {server} --experiment_name {experiment}_r{r}_smooth{smooth}_train{train}_seed{seed} "
                    )
                    utils.write_experiment_result(experiment=experiment, params={'seed' : seed, 'r': r, 'smooth' : smooth, 'train' : train, 'disjoint' : disjoint_test_idx})

def missing_by_feature_correlation(args):
    experiment = 'missing_by_feature_correlation' + ('_DEBUG' if args.small_data else '')
    utils.make_experiment_sheet(experiment=experiment, params=['seed', 'condition'], num_seeds=len(seeds))
    for seed in seeds:
        for condition in ['optR', 'memorization']:
            for corr in [.5, .6, .7, .8, .9, 1]:
                for causal in [0, 1]:
                    epochs=20
                    use_retrieval = train_retriever = True
                    use_optimal_retrieval = False
                    if condition=='memorization':
                        epochs=40
                        train_retriever=use_retrieval=False
                    if condition == 'fixed':
                        train_retriever=False
                    if condition == 'optR':
                        train_retriever=False
                        use_optimal_retrieval=True
                    if causal==1 and condition=='memorization':
                        continue
                    rb = .2 if train_retriever else -1
                    os.system(f"python main.py --gpu {args.gpu} --data_dir data/synthetic --train_batch_size {tbs} --grad_accumulation_factor {gaf} --num_train_epochs {epochs} --model roberta-base "
                              f" --use_retrieval {use_retrieval} --use_optimal_retrieval {use_optimal_retrieval} --train_retriever {train_retriever} --context_includes E --num_relevant_points 10 --context_size 1 --top_k 1 --use_textcat true -n 5000 --rebuild_index_every_perc {rb} "
                              f" --use_textcat true --freeze_retriever_until 2 --explanation_only_causal {causal} --weak_feature_correlation {corr} --print true "
                              f" {small_data_addin} --seed {seed} --server {server} --experiment_name {experiment}_{condition}_causal{causal}_corr{corr} "
                    )
                    utils.write_experiment_result(experiment=experiment, params={'seed' : seed, 'condition' : condition, 'causal' : causal, 'corr' : corr})

def missing_opt_by_translate_model_n(args):
    experiment = f'missing_opt_by_translate_model_n' + ('_DEBUG' if args.small_data else '')
    utils.make_experiment_sheet(experiment=experiment, params=['seed', 'model', 'n', 'translate'], num_seeds=len(seeds))
    for seed in seeds:
        for translate in [1,0]:
            for model in ['roberta-base']:
                for n, epochs in [(5000, 40), (10000, 30), (20000, 20)]:
                    _translate = '--translate xplus5 ' if translate else ''
                    os.system(f"python main.py --gpu {args.gpu} --data_dir data/synthetic --train_batch_size {tbs} --grad_accumulation_factor {gaf} --num_train_epochs {epochs} --model {model} "
                              f" -n {n} --use_retrieval true --use_optimal_retrieval true --context_includes E --num_tasks 500 --top_k 1 --lr {lr} "
                              f" --use_textcat true --context_size 1 --print true {_translate} "
                              f"{small_data_addin} --seed {seed} --server {server} --experiment_name {experiment}_{model}_n{n}_translate{translate}_seed{seed} "
                    )
                    utils.write_experiment_result(experiment=experiment, params={'seed' : seed, 'model' : model, 'n' : n, 'translate' : translate})

def missing_by_seed_test(args):
    experiment = 'missing_seed_test' + ('_DEBUG' if args.small_data else '')
    utils.make_experiment_sheet(experiment=experiment, params=['seed', 'method', 'condition'], num_seeds=len(seeds))
    for seed in seeds:
        for method in ['ELV']:
            use_textcat = use_ELV = False
            if method == 'textcat':
                use_textcat = True
                c = k = 4
            if method == 'ELV':
                use_textcat = True
                c = k = 4
            for condition in ['learning']:
                use_retrieval = train_retriever = True
                use_optimal_retrieval = False
                rb = .2 if train_retriever else -1
                os.system(f"python main.py --gpu {args.gpu} --data_dir data/synthetic --train_batch_size {tbs} --grad_accumulation_factor {gaf} --model roberta-base --max_seq_len 120  "
                          f" --use_retrieval {use_retrieval} --use_optimal_retrieval {use_optimal_retrieval} --train_retriever {train_retriever} --context_includes E --num_relevant_points 10 --context_size {c} --top_k {k} --use_textcat true -n 5000 --rebuild_index_every_perc {rb} "
                          f" --use_textcat {use_textcat} --use_ELV {use_ELV} --freeze_retriever_until 5 --num_train_epochs 25 "
                          f" {small_data_addin} --seed {seed} --server {server} --experiment_name {experiment}_{method}_{condition}_seed{seed} "
                )
                utils.write_experiment_result(experiment=experiment, params={'seed' : seed, 'method': method, 'condition' : condition})

def missing_by_k(args):
    experiment = 'missing_textcat_by_k' + ('_DEBUG' if args.small_data else '')
    utils.make_experiment_sheet(experiment=experiment, params=['seed', 'k'], num_seeds=len(seeds))
    for seed in seeds:
        for k in [10, 8, 6, 4, 2, 1]:
            train_retriever=(k!=1)
            rb = .2 if train_retriever else -1
            os.system(f"python main.py --gpu {args.gpu} "
                      f" --data_dir data/synthetic --use_retrieval true --train_retriever {train_retriever} --context_includes E --num_relevant 10 -n 5000 --train_batch_size {tbs} --grad_accumulation_factor {gaf} --num_print 2 --rebuild_index_every_perc {rb} "
                      f" --use_textcat true --context_size 1 --top_k {k} --freeze_retriever_until 5 --num_train_epochs 25 "
                      f" {small_data_addin} --seed {seed} --server {server} --experiment_name missing_by_k_c1_k{k}_seed{seed} "
            )
            utils.write_experiment_result(experiment=experiment, params={'seed' : seed, 'k' : k})

# --- evidential experiments --- #

def evidential_by_init(args):
    experiment = f'evidential_by_init' + ('_DEBUG' if args.small_data else '')
    utils.make_experiment_sheet(experiment=experiment, params=['seed', 'epochs', 'degrade'], num_seeds=len(seeds))
    for seed in seeds:
        for epochs in [-1, 0, 4, 8]:
            for degrade in [0, 5e-3, 1e-2, 5e-2]:
                train_retriever=False if epochs==-1 else True
                rb = .2 if train_retriever else -1
                train_epochs = 20
                os.system(f"python main.py --gpu {args.gpu} --data_dir data/synthetic --train_batch_size {tbs} --grad_accumulation_factor {gaf} --model roberta-base "
                          f" --use_retrieval true --train_retriever {train_retriever} --context_includes E --num_relevant_points 10 --context_size 4 --top_k 4 --use_textcat true -n 5000 --rebuild_index_every_perc {rb} "
                          f" --freeze_retriever_until {epochs} --num_train_epochs {train_epochs} --explanation_kind evidential --evidential_eps 2 --degrade_retriever {degrade} "
                          f" {small_data_addin} --seed {seed} --server {server} --experiment_name {experiment}_e{epochs}_d{degrade}_seed{seed} "
                )
                utils.write_experiment_result(experiment=experiment, params={'seed' : seed, 'epochs' : epochs, 'degrade' : degrade})

def evidential_opt_by_method_c(args):
    experiment = f'evidential_optR_by_method_c' + ('_DEBUG' if args.small_data else '')
    utils.make_experiment_sheet(experiment=experiment, params=['seed', 'method', 'c'], num_seeds=len(seeds))
    for seed in seeds:
        for method in ['textcat', 'ELV']:
            use_ELV = (method == 'ELV')
            use_textcat = (method == 'textcat')
            for c in reversed([1,2,4,6,8,10,12,14]):
                os.system(f"python main.py --gpu {args.gpu} --data_dir data/synthetic --train_batch_size {tbs} --grad_accumulation_factor {gaf} --num_train_epochs 20 --model roberta-base "
                          f" -n 5000 --explanation_kind evidential --evidential_eps 4 --use_retrieval true --use_optimal_retrieval true --context_includes E --num_tasks 250 --top_k 1 "
                          f" --use_textcat {use_textcat} --use_ELV {use_ELV} --context_size {c} --max_seq_len 300 --print true "
                          f"{small_data_addin} --seed {seed} --server {server} --experiment_name {experiment}_{method}_c{c}_seed{seed} "
                )
                utils.write_experiment_result(experiment=experiment, params={'seed' : seed, 'method' : method, 'c' : c})

def evidential_opt_by_method_n(args):
    experiment = 'evidential_optR_by_method_n' + ('_DEBUG' if args.small_data else '')
    utils.make_experiment_sheet(experiment=experiment, params=['seed', 'method', 'n'], num_seeds=len(seeds))
    for seed in seeds:
        for method in ['ELV', 'textcat']:
            use_ELV = (method == 'ELV')
            use_textcat = (method == 'textcat')
            for n in [1000, 1500, 2500, 5000, 7500, 10000]:
                os.system(f"python main.py --gpu {args.gpu} --data_dir data/synthetic --train_batch_size {tbs} --grad_accumulation_factor {gaf} --num_train_epochs 20 --model roberta-base "
                          f" -n {n} --explanation_kind evidential --evidential_eps 2 --use_retrieval true --use_optimal_retrieval true --context_includes E --num_tasks 250 --context_size 5 --top_k 1 "
                          f" --use_textcat {use_textcat} --use_ELV {use_ELV} "
                          f"{small_data_addin} --seed {seed} --server {server} --experiment_name {experiment}_{method}_n{n}_seed{seed} "
                )
                utils.write_experiment_result(experiment=experiment, params={'seed' : seed, 'method' : method, 'n' : n})

def evidential_by_k(args):
    experiment = 'evidential_textcat_by_k' + ('_DEBUG' if args.small_data else '')
    utils.make_experiment_sheet(experiment=experiment, params=['seed', 'k'], num_seeds=len(seeds))
    for seed in seeds:
        for k in [10, 8, 6, 4, 2, 1]:
            train_retriever=(k!=1)
            rb = .2 if train_retriever else -1
            os.system(f"python main.py --gpu {args.gpu} "
                      f" --data_dir data/synthetic --use_retrieval true --train_retriever {train_retriever} --context_includes E --num_relevant 10 -n 5000 --train_batch_size {tbs} --grad_accumulation_factor {gaf} --num_print 2 --explanation_kind evidential --evidential_eps 2  --rebuild_index_every_perc {rb} "
                      f" --use_textcat true --context_size 1 --top_k {k} --freeze_retriever_until 5 --num_train_epochs 25 "
                      f" {small_data_addin} --seed {seed} --server {server} --experiment_name evidential_by_k_c1_k{k}_seed{seed} "
            )
            utils.write_experiment_result(experiment=experiment, params={'seed' : seed, 'k' : k})


def evidential_by_retriever(args):
    experiment = 'evidential_by_retriever' + ('_DEBUG' if args.small_data else '')
    utils.make_experiment_sheet(experiment=experiment, params=['seed', 'train', 'retriever'], num_seeds=len(seeds))
    for seed in seeds:
        for train in [0, 1]:
            for condition in ['random','roberta-base','sentencebert']:
                if condition == 'random': 
                    reinitialize=True
                    retrieval_model = 'roberta-base'
                if condition == 'roberta-base':
                    reinitialize=False
                    retrieval_model = 'roberta-base'
                if condition == 'sentencebert':
                    reinitialize=False
                    retrieval_model = 'sentence-transformers/roberta-base-nli-stsb-mean-tokens'
                rb = .2 if train else -1
                os.system(f"python main.py --gpu {args.gpu} --data_dir data/synthetic --train_batch_size {tbs} --grad_accumulation_factor {gaf} --num_train_epochs 20 --model roberta-base "
                          f" --use_retrieval true --train_retriever {train} --context_includes E --num_relevant_points 10 --context_size 4 --top_k 4 --use_textcat true -n 5000 --rebuild_index_every_perc {rb} "
                          f" --retrieval_model {retrieval_model} --reinitialize_retriever {reinitialize} --freeze_retriever_until 4 --explanation_kind evidential --evidential_eps 2 "
                          f" {small_data_addin} --seed {seed} --server {server} --experiment_name {experiment}_train{train}_Retriever{condition}_seed{seed} "
                )
                utils.write_experiment_result(experiment=experiment, params={'seed' : seed, 'train': train, 'retriever' : condition})

def evidential_by_learning(args):
    experiment = 'evidential_by_learning' + ('_DEBUG' if args.small_data else '')
    utils.make_experiment_sheet(experiment=experiment, params=['seed', 'method', 'condition'], num_seeds=len(seeds))
    for seed in seeds:
        for method in ['textcat','ELV']:
            use_textcat = use_ELV = False
            if method == 'textcat':
                use_textcat = True
                c = 6; k = 6
            if method == 'ELV':
                use_ELV = True
                c = 4; k = 4
            for condition in ['learning','fixed','optimal']:
                use_retrieval = train_retriever = True
                use_optimal_retrieval = False
                if condition == 'fixed':
                    train_retriever=False
                if condition == 'optimal':
                    train_retriever=False
                    use_optimal_retrieval=True
                rb = .2 if train_retriever else -1
                os.system(f"python main.py --gpu {args.gpu} --data_dir data/synthetic --train_batch_size {tbs} --grad_accumulation_factor {gaf} --model roberta-base --max_seq_len 120  "
                          f" --use_retrieval {use_retrieval} --use_optimal_retrieval {use_optimal_retrieval} --train_retriever {train_retriever} --context_includes E --num_relevant_points 10 --context_size {c} --top_k {k} --use_textcat true -n 5000 --rebuild_index_every_perc {rb} "
                          f" --use_textcat {use_textcat} --use_ELV {use_ELV} --freeze_retriever_until 5 --num_train_epochs 25 --explanation_kind evidential --evidential_eps 2 "
                          f" {small_data_addin} --seed {seed} --server {server} --experiment_name {experiment}_{method}_{condition}_seed{seed} "
                )
                utils.write_experiment_result(experiment=experiment, params={'seed' : seed, 'method': method, 'condition' : condition})

def evidential_by_seed_test(args):
    experiment = f'evidential_seed_test' + ('_DEBUG' if args.small_data else '')
    utils.make_experiment_sheet(experiment=experiment, params=['seed', 'method', 'condition'], num_seeds=len(seeds))
    for seed in seeds:
        for method in ['textcat']:
            use_textcat = use_ELV = False
            if method == 'textcat':
                use_textcat = True
                c = k = 6
            if method == 'ELV':
                use_textcat = True
                c = k = 4
            for condition in ['learning']:
                use_retrieval = train_retriever = True
                use_optimal_retrieval = False
                rb = .2 if train_retriever else -1
                os.system(f"python main.py --gpu {args.gpu} --data_dir data/synthetic --train_batch_size {tbs} --grad_accumulation_factor {gaf} --model roberta-base --max_seq_len 120  "
                          f" --use_retrieval {use_retrieval} --use_optimal_retrieval {use_optimal_retrieval} --train_retriever {train_retriever} --context_includes E --num_relevant_points 10 --context_size {c} --top_k {k} --use_textcat true -n 5000 --rebuild_index_every_perc {rb} "
                          f" --use_textcat {use_textcat} --use_ELV {use_ELV} --freeze_retriever_until 5 --num_train_epochs 25 --explanation_kind evidential --evidential_eps 2 "
                          f" {small_data_addin} --seed {seed} --server {server} --experiment_name {experiment}_{method}_{condition}_seed{seed} "
                )
                utils.write_experiment_result(experiment=experiment, params={'seed' : seed, 'method': method, 'condition' : condition})

# --- recomposable experiments --- #

def recomposable_by_learning(args):
    experiment = 'recomposable_by_learning' + ('_DEBUG' if args.small_data else '')
    utils.make_experiment_sheet(experiment=experiment, params=['seed', 'method', 'condition'], num_seeds=len(seeds))
    for seed in seeds:
        for method in ['textcat','ELV']:
            use_textcat = use_ELV = False
            if method == 'textcat':
                use_textcat = True
                c = 6; k = 6
            if method == 'ELV':
                use_ELV = True
                c = 4; k = 4
            for condition in ['optimal','learning','fixed']:
                use_retrieval = train_retriever = True
                use_optimal_retrieval = False
                if condition == 'fixed':
                    train_retriever=False
                if condition == 'optimal':
                    train_retriever=False
                    use_optimal_retrieval=True
                rb = .2 if train_retriever else -1
                os.system(f"python main.py --gpu {args.gpu} --data_dir data/synthetic --train_batch_size {tbs} --grad_accumulation_factor {gaf} --model roberta-base --max_seq_len 120 "
                          f" --use_retrieval {use_retrieval} --use_optimal_retrieval {use_optimal_retrieval} --train_retriever {train_retriever} --context_includes E --num_relevant_points 10 --context_size {c} --top_k {k} --use_textcat true -n 5000 --rebuild_index_every_perc {rb} "
                          f" --use_textcat {use_textcat} --use_ELV {use_ELV} --freeze_retriever_until 5 --num_train_epochs 25 --explanation_kind recomposable --recomposable_pieces 2 "
                          f" {small_data_addin} --seed {seed} --server {server} --experiment_name {experiment}_{method}_{condition} "
                )
                utils.write_experiment_result(experiment=experiment, params={'seed' : seed, 'method': method, 'condition' : condition})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", '-e', type=str) 
    parser.add_argument("--gpu", default='0', type=str) 
    parser.add_argument("--server", default='', type=str) 
    parser.add_argument("--seeds", default=1, type=int) 
    parser.add_argument("--start", default=0, type=int) 
    parser.add_argument("--train_batch_size", default=2, type=int, help='')
    parser.add_argument("--grad_accumulation_factor", default=5, type=int, help='')
    parser.add_argument("--small_data", '-s', action='store_true')
    args = parser.parse_args()
 
    # globals
    server = args.server
    small_data_addin = f'-s -ss 11 --num_train_epochs 2 ' if args.small_data else ''
    seeds = list(range(args.start, args.seeds))
    tbs = args.train_batch_size
    gaf = args.grad_accumulation_factor

    # experiments
    if args.experiment == 'semeval_baseline': semeval_baseline(args)
    if args.experiment == 'semeval_textcat': semeval_textcat(args)
    if args.experiment == 'semeval_textcat_by_context': semeval_textcat_by_context(args)

    if args.experiment == 'tacred_baseline': tacred_baseline(args)
    if args.experiment == 'tacred_textcat': tacred_textcat(args)
    if args.experiment == 'tacred_textcat_by_context': tacred_textcat_by_context(args) 

    if args.experiment == 'esnli_baseline': esnli_baseline(args)
    if args.experiment == 'esnli_textcat': esnli_textcat(args)
    if args.experiment == 'esnli_textcat_by_context': esnli_textcat_by_context(args)

    if args.experiment == 'memorization_by_n': memorization_by_n(args)
    if args.experiment == 'memorization_by_num_tasks': memorization_by_num_tasks(args)
    if args.experiment == 'memorization_by_r_smoothness': memorization_by_r_smoothness(args)
    if args.experiment == 'memorization_by_seed_test': memorization_by_seed_test(args)

    if args.experiment == 'missing_by_learning': missing_by_learning(args)
    if args.experiment == 'missing_by_k': missing_by_k(args)
    if args.experiment == 'missing_by_rb': missing_by_rb(args)
    if args.experiment == 'missing_by_r_smoothness': missing_by_r_smoothness(args)  
    if args.experiment == 'missing_by_feature_correlation': missing_by_feature_correlation(args)     
    if args.experiment == 'missing_opt_by_translate_model_n': missing_opt_by_translate_model_n(args)
    if args.experiment == 'missing_by_seed_test': missing_by_seed_test(args)

    if args.experiment == 'evidential_by_learning': evidential_by_learning(args)
    if args.experiment == 'evidential_opt_by_method_c': evidential_opt_by_method_c(args)
    if args.experiment == 'evidential_opt_by_method_n': evidential_opt_by_method_n(args)
    if args.experiment == 'evidential_by_c': evidential_by_c(args)
    if args.experiment == 'evidential_by_k': evidential_by_k(args)
    if args.experiment == 'evidential_by_init': evidential_by_init(args)
    if args.experiment == 'evidential_by_retriever': evidential_by_retriever(args)
    if args.experiment == 'evidential_by_seed_test': evidential_by_seed_test(args)
    
    if args.experiment == 'recomposable_by_learning': recomposable_by_learning(args)
    
