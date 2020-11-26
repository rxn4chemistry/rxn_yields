import typer
import torch
import pkg_resources
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import pandas as pd
from typing import List, Tuple

from rxn_yields.core import SmilesClassificationModel
from rxn_yields.data import generate_buchwald_hartwig_rxns
from rxn_yields.augmentation import do_random_permutations_on_df, do_randomizations_on_df
import sklearn
try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False

app = typer.Typer()

RANDOM_SPLIT = [

    ('FullCV_01', 2768), ('FullCV_02', 2768), ('FullCV_03', 2768), ('FullCV_04', 2768), ('FullCV_05', 2768),
    ('FullCV_06', 2768), ('FullCV_07', 2768), ('FullCV_08', 2768), ('FullCV_09', 2768), ('FullCV_10', 2768),
]

TEST_SPLIT = [('Test1', 3058), ('Test2', 3056), ('Test3', 3059), ('Test4', 3056)]


DISCOVERY_SPLIT_NEW = [
   ('FullCV_01', 99), ('FullCV_01', 198), ('FullCV_01', 396), ('FullCV_01', 792), ('FullCV_01', 1187), ('FullCV_01', 1384),  ('FullCV_01', 1978),
   ('FullCV_02', 99), ('FullCV_02', 198), ('FullCV_02', 396), ('FullCV_02', 792), ('FullCV_02', 1187), ('FullCV_02', 1384), 
   ('FullCV_02', 1978),
   ('FullCV_03', 99), ('FullCV_03', 198), ('FullCV_03', 396), ('FullCV_03', 792), ('FullCV_03', 1187), ('FullCV_03', 1384), 
   ('FullCV_03', 1978),
   ('FullCV_04', 99), ('FullCV_04', 198), ('FullCV_04', 396), ('FullCV_04', 792), ('FullCV_04', 1187), ('FullCV_04', 1384), 
   ('FullCV_04', 1978),
   ('FullCV_05', 99), ('FullCV_05', 198), ('FullCV_05', 396), ('FullCV_05', 792), ('FullCV_05', 1187), ('FullCV_05', 1384), 
   ('FullCV_05', 1978),
    ('FullCV_06', 99), ('FullCV_06', 198), ('FullCV_06', 396), ('FullCV_06', 792), ('FullCV_06', 1187), ('FullCV_06', 1384),  ('FullCV_06', 1978), 
    ('FullCV_07', 99), ('FullCV_07', 198), ('FullCV_07', 396), ('FullCV_07', 792), ('FullCV_07', 1187), ('FullCV_07', 1384),  ('FullCV_07', 1978),
    ('FullCV_08', 99), ('FullCV_08', 198), ('FullCV_08', 396), ('FullCV_08', 792), ('FullCV_08', 1187), ('FullCV_08', 1384),  ('FullCV_08', 1978),
    ('FullCV_09', 99), ('FullCV_09', 198), ('FullCV_09', 396), ('FullCV_09', 792), ('FullCV_09', 1187), ('FullCV_09', 1384),  ('FullCV_09', 1978),
    ('FullCV_10', 99), ('FullCV_10', 198), ('FullCV_10', 396), ('FullCV_10', 792), ('FullCV_10', 1187), ('FullCV_10', 1384),  ('FullCV_10', 1978),
]

DISCOVERY_SPLIT_HALF1 = [
   ('FullCV_01', 99), ('FullCV_01', 198), ('FullCV_01', 396), ('FullCV_01', 792), ('FullCV_01', 1187), ('FullCV_01', 1384),  ('FullCV_01', 1978),
   ('FullCV_02', 99), ('FullCV_02', 198), ('FullCV_02', 396), ('FullCV_02', 792), ('FullCV_02', 1187), ('FullCV_02', 1384), 
   ('FullCV_02', 1978),
   ('FullCV_03', 99), ('FullCV_03', 198), ('FullCV_03', 396), ('FullCV_03', 792), ('FullCV_03', 1187), ('FullCV_03', 1384), 
   ('FullCV_03', 1978),
   ('FullCV_04', 99), ('FullCV_04', 198), ('FullCV_04', 396), ('FullCV_04', 792), ('FullCV_04', 1187), ('FullCV_04', 1384), 
   ('FullCV_04', 1978),
   ('FullCV_05', 99), ('FullCV_05', 198), ('FullCV_05', 396), ('FullCV_05', 792), ('FullCV_05', 1187), ('FullCV_05', 1384), 
   ('FullCV_05', 1978)
]

DISCOVERY_SPLIT_HALF2 = [
    ('FullCV_06', 99), ('FullCV_06', 198), ('FullCV_06', 396), ('FullCV_06', 792), ('FullCV_06', 1187), ('FullCV_06', 1384),  ('FullCV_06', 1978), 
    ('FullCV_07', 99), ('FullCV_07', 198), ('FullCV_07', 396), ('FullCV_07', 792), ('FullCV_07', 1187), ('FullCV_07', 1384),  ('FullCV_07', 1978),
    ('FullCV_08', 99), ('FullCV_08', 198), ('FullCV_08', 396), ('FullCV_08', 792), ('FullCV_08', 1187), ('FullCV_08', 1384),  ('FullCV_08', 1978),
    ('FullCV_09', 99), ('FullCV_09', 198), ('FullCV_09', 396), ('FullCV_09', 792), ('FullCV_09', 1187), ('FullCV_09', 1384),  ('FullCV_09', 1978),
    ('FullCV_10', 99), ('FullCV_10', 198), ('FullCV_10', 396), ('FullCV_10', 792), ('FullCV_10', 1187), ('FullCV_10', 1384),  ('FullCV_10', 1978),
]


@app.command()
def randomsplit_canonical():
    """
    Launch training and evaluation with a pretrained base model from rxnfp.
    """
    launch_training_on_all_splits(experiment='repro_randomsplit', splits=RANDOM_SPLIT, base_model='pretrained', dropout=0.7987, learning_rate=0.00009659, epochs=10)

@app.command()
def randomsplit_rotated_15():
    """
    Launch training and evaluation with a pretrained base model from rxnfp.
    """
    launch_training_on_all_splits(experiment='repro_randomsplit', splits=RANDOM_SPLIT,  base_model='pretrained', dropout=0.7987, learning_rate=0.00009659, n_randomizations=15, random_type='rotated', epochs=10)

@app.command()
def randomsplit_permuted_non_fixed_5():
    """
    Launch training and evaluation with a pretrained base model from rxnfp.
    """
    launch_training_on_all_splits(experiment='repro_randomsplit', splits=RANDOM_SPLIT,  base_model='pretrained', dropout=0.7987, learning_rate=0.00009659, n_permutations=5, fixed_perm=False, epochs=10)

@app.command()
def randomsplit_permuted_non_fixed_rotated_15():
    """
    Launch training and evaluation with a pretrained base model from rxnfp.
    """
    launch_training_on_all_splits(experiment='repro_randomsplit', splits=RANDOM_SPLIT,  base_model='pretrained', dropout=0.7987, learning_rate=0.00009659, n_permutations=15, random_type='rotated', fixed_perm=False, epochs=10)


@app.command()
def testsplit_canonical():
    """
    Launch training and evaluation with a pretrained base model from rxnfp.
    """
    launch_training_on_all_splits(experiment='repro_testsplit_multi', splits=TEST_SPLIT, base_model='pretrained', dropout=0.7987, learning_rate=0.00009659, epochs=10, manual_seeds=[1,2,3,4,5])

@app.command()
def testsplit_rotated_15():
    """
    Launch training and evaluation with a pretrained base model from rxnfp.
    """
    launch_training_on_all_splits(experiment='repro_testsplit_multi', splits=TEST_SPLIT,  base_model='pretrained', dropout=0.7987, learning_rate=0.00009659, n_randomizations=15, random_type='rotated', epochs=10, manual_seeds=[1,2,3,4,5])


@app.command()
def testsplit_permuted_non_fixed_5():
    """
    Launch training and evaluation with a pretrained base model from rxnfp.
    """
    launch_training_on_all_splits(experiment='repro_testsplit_multi', splits=TEST_SPLIT,  base_model='pretrained', dropout=0.7987, learning_rate=0.00009659, n_permutations=5, fixed_perm=False, epochs=10, manual_seeds=[1,2,3,4,5])


@app.command()
def testsplit_permuted_non_fixed_rotated_15():
    """
    Launch training and evaluation with a pretrained base model from rxnfp.
    """
    launch_training_on_all_splits(experiment='repro_testsplit_multi', splits=TEST_SPLIT,  base_model='pretrained', dropout=0.7987, learning_rate=0.00009659, n_permutations=15, random_type='rotated', fixed_perm=False, epochs=10, manual_seeds=[1,2,3,4,5])


@app.command()
def discoverysplit_canonical():
    """
    Launch training and evaluation with a pretrained base model from rxnfp.
    """
    launch_training_on_all_splits(experiment='repro_discoverysplit', splits=DISCOVERY_SPLIT_NEW, base_model='pretrained', dropout=0.7987, learning_rate=0.00009659, epochs=10)


@app.command()
def discoverysplit_permuted_non_fixed_5():
    """
    Launch training and evaluation with a pretrained base model from rxnfp.
    """
    launch_training_on_all_splits(experiment='repro_discoverysplit', splits=DISCOVERY_SPLIT_NEW,  base_model='pretrained', dropout=0.7987, learning_rate=0.00009659, n_permutations=5, fixed_perm=False, epochs=10)


@app.command()
def discoverysplit1_rotated_15():
    """
    Launch training and evaluation with a pretrained base model from rxnfp.
    """
    launch_training_on_all_splits(experiment='repro_discoverysplit', splits=DISCOVERY_SPLIT_HALF1,  base_model='pretrained', dropout=0.7987, learning_rate=0.00009659, n_randomizations=15, random_type='rotated', epochs=10)


@app.command()
def discoverysplit1_permuted_non_fixed_rotated_15():
    """
    Launch training and evaluation with a pretrained base model from rxnfp.
    """
    launch_training_on_all_splits(experiment='repro_discoverysplit', splits=DISCOVERY_SPLIT_HALF1,  base_model='pretrained', dropout=0.7987, learning_rate=0.00009659, n_permutations=15, random_type='rotated', fixed_perm=False, epochs=10)


@app.command()
def discoverysplit2_rotated_15():
    """
    Launch training and evaluation with a pretrained base model from rxnfp.
    """
    launch_training_on_all_splits(experiment='repro_discoverysplit', splits=DISCOVERY_SPLIT_HALF2,  base_model='pretrained', dropout=0.7987, learning_rate=0.00009659, n_randomizations=15, random_type='rotated', epochs=10)


@app.command()
def discoverysplit2_permuted_non_fixed_rotated_15():
    """
    Launch training and evaluation with a pretrained base model from rxnfp.
    """
    launch_training_on_all_splits(experiment='repro_discoverysplit', splits=DISCOVERY_SPLIT_HALF2,  base_model='pretrained', dropout=0.7987, learning_rate=0.00009659, n_permutations=15, random_type='rotated', fixed_perm=False, epochs=10)



def launch_training_on_all_splits(experiment: str, splits: List, base_model: str, dropout: float, learning_rate: float, n_permutations: int=0, n_randomizations:int =0, epochs: int=15, fixed_perm: bool=False, random_type: str='', manual_seeds=[42]):
    project = f'buchwald_hartwig_{experiment}'
    
    for manual_seed in manual_seeds:
        model_args = {
        'wandb_project': project, 'num_train_epochs': epochs, 'overwrite_output_dir': True,
        'learning_rate': learning_rate, 'gradient_accumulation_steps': 1,
        'regression': True, "num_labels":1, "fp16": False,
        "evaluate_during_training": True, 'manual_seed': manual_seed,
        "max_seq_length": 300, "train_batch_size": 16,"warmup_ratio": 0.00,
        "config" : { 'hidden_dropout_prob': dropout } }
            
        for (name, split) in splits:
            if wandb_available: 
                wandb.init(name=f"{name}_perm_{n_permutations}_rand_{n_randomizations}_{random_type}_epochs_{epochs}_split_{str(split).replace('-','_')}", project=project, reinit=True)
                wandb.config.update({
                        "base_model": base_model,
                        "n_permutations": n_permutations,
                        "n_randomizations": n_randomizations,
                        "epochs": epochs,
                        "fixed_perm": fixed_perm,
                        "random_type": random_type,
                        "seed": manual_seed,
                        "name": name,
                        "dropout": dropout,
                        "learning_rate": learning_rate,
                        "split": split,
                    })

            df_doyle = pd.read_excel('../data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx', sheet_name=name)
            df_doyle['rxn'] = generate_buchwald_hartwig_rxns(df_doyle)

            train_df = df_doyle.iloc[:split-1][['rxn', 'Output']] # paper has starting index 1 not 0
            test_df = df_doyle.iloc[split-1:][['rxn', 'Output']] # paper has starting index 1 not 0

            train_df.columns = ['text', 'labels']
            test_df.columns = ['text', 'labels']
            mean = train_df.labels.mean()
            std = train_df.labels.std()
            train_df['labels'] = (train_df['labels'] - mean) / std
            test_df['labels'] = (test_df['labels'] - mean) / std

            if n_permutations > 0:
                augmented_train_df = do_random_permutations_on_df(train_df,
                    n_permutations=n_permutations, fixed=fixed_perm, random_type=random_type, seed=model_args['manual_seed'])
                train_df = augmented_train_df.sample(frac=1., random_state=model_args['manual_seed'])
            elif n_randomizations > 0:
                augmented_train_df = do_randomizations_on_df(train_df, n_randomizations=n_randomizations, random_type=random_type, seed=model_args['manual_seed'])
                train_df = augmented_train_df.sample(frac=1., random_state=model_args['manual_seed'])

            model_path =  pkg_resources.resource_filename("rxnfp", f"models/transformers/bert_{base_model}")
            pretrained_bert = SmilesClassificationModel("bert", model_path, num_labels=1, args=model_args, use_cuda=torch.cuda.is_available())
            pretrained_bert.train_model(train_df, output_dir=f"../repro/{experiment}/{experiment}_buchwald_hartwig_{name}_perm_{n_permutations}_rand_{n_randomizations}_{random_type if not random_type=='' else 'canonical'}_epochs_{epochs}_split_{str(split).replace('-','_')}_seed_{manual_seed}", eval_df=test_df, r2=sklearn.metrics.r2_score)
            if wandb_available: wandb.join() # multiple runs in same script

if __name__ == '__main__':
    app()
