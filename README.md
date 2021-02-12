# When Can Models Learn From Explanations?

This is the codebase for the paper: "[When Can Models Learn From Explanations? A Formal Framework for Understanding the Roles of Explanation Data](https://arxiv.org/abs/2102.02201)"

Here's the directory structure:

```
data/ --> data folder (files too large to upload here but are publicly available)
models/ --> contains special model classes for use with retrieval model
training_reports/ --> folder to be populated with individual training run reports
result_sheets/ --> folder to be populated with .csv's of results from experiments 
figures/ --> contains plots generated by plots.Rmd
main.py --> main script for all individual experiments in the paper
make_SNLI_data.py --> convert e-SNLI .txt files to .csv's
plots.Rmd --> R markdown file that makes plots using .csv's in result_sheets
report.py --> experiment logging class, reports appear in training_reports
retriever.py --> class for retrieval model
run_tasks.py --> script for running several experiments for each RQ in the paper
utils.py --> data loading and miscellaneous utilities
write_synthetic_data.py --> script for writing synthetic datasets
```

The code is written in python 3.6. `plots.Rmd` is an R markdown file that makes the plots for each experiment. The package requirements are:

```
torch==1.4
transformers==3.3.1
faiss-cpu==1.6.3
pandas==1.0.5
numpy==1.18.5
scipy==1.4.1
sklearn==0.23.1
argparse==1.1
json==2.0.9
```

Experimental results in the paper are replicated by running `run_tasks.py` with a special experiment command. Below, we give commands organized by the corresponding research question in the paper. For synthetic data experiments, all that is required is that you first set `save_dir` and `cache_dir` in `main.py`. We later give instructions for downloading and formatting data for experiments with existing datasets.

The `run_tasks.py` script can take a few additional arguments when desired: `--seeds` gives the number of seeds to run for each session (defaults to 1), `--gpu` controls the GPU, and `--train_batch_size` and `--grad_accumulation_factor` can be used to control the effective train batch size and memory usage. 

*RQ1*

`python run_tasks.py --experiment memorization_by_num_tasks`

*RQ2*

`python run_tasks.py --experiment memorization_by_n`

`python run_tasks.py --experiment missing_by_learning`

*RQ3*

`python run_tasks.py --experiment evidential_by_learning`

`python run_tasks.py --experiment recomposable_by_learning`

*RQ4*

`python run_tasks.py --experiment evidential_opt_by_method_n`

*RQ5*

`python run_tasks.py --experiment memorization_by_r_smoothness`

*RQ6*

`python run_tasks.py --experiment missing_by_feature_correlation`

`python run_tasks.py --experiment missing_opt_by_translate_model_n`

*RQ7*

`python run_tasks.py --experiment evidential_by_retriever`

`python run_tasks.py --experiment evidential_by_init`

*RQ8*

The dataused used in the paper can be obtained here: [TACRED](https://catalog.ldc.upenn.edu/LDC2018T24) and [e-SNLI](https://github.com/OanaMariaCamburu/e-SNLI) (SemEval included here), and should be placed into folders in `data/` titled `semeval`, `tacred`, and `eSNLI`. Running `make_SNLI_data.py` will format the SNLI data into .csv files as expected by data utilities in `utils.py`. 

Experiments with existing datasets:

*SemEval*

`python run_tasks.py --experiment semeval_baseline`

`python run_tasks.py --experiment semeval_textcat`

`python run_tasks.py --experiment semeval_textcat_by_context`

*TACRED*

`python run_tasks.py --experiment tacred_baseline`

`python run_tasks.py --experiment tacred_textcat`

`python run_tasks.py --experiment tacred_textcat_by_context`

*e-SNLI*

`python run_tasks.py --experiment esnli_baseline`

`python run_tasks.py --experiment esnli_textcat`

`python run_tasks.py --experiment esnli_textcat_by_context`

Additional tuning experiments:

`python run_tasks.py --experiment missing_by_k`

`python run_tasks.py --experiment missing_by_rb`

`python run_tasks.py --experiment evidential_opt_by_method_c`

`python run_tasks.py --experiment evidential_by_k`

Seed tests:

`python run_tasks.py --experiment memorization_by_seed_test --seeds 10`

`python run_tasks.py --experiment missing_by_seed_test --seeds 5`

`python run_tasks.py --experiment evidential_by_seed_test --seeds 5`
