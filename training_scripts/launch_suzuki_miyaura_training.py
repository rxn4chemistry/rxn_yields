import typer
import torch
import pkg_resources
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from rdkit import Chem
from rdkit.Chem import rdChemReactions
import pandas as pd
from tqdm import tqdm
from typing import List

from rxn_yields.core import SmilesTokenizer, SmilesClassificationModel
import sklearn
try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False

app = typer.Typer()

NAME_SPLIT = [
    ('random_split_0', 4032), ('random_split_1', 4032), ('random_split_2', 4032), ('random_split_3', 4032),
    ('random_split_4', 4032), ('random_split_5', 4032), ('random_split_6', 4032), ('random_split_7', 4032),
    ('random_split_8', 4032), ('random_split_9', 4032),
]

DISCOVERY_SPLIT = [
    ('random_split_0', 288), ('random_split_0', 576), ('random_split_0', 1152),
    ('random_split_1', 288), ('random_split_1', 576), ('random_split_1', 1152),
    ('random_split_2', 288), ('random_split_2', 576), ('random_split_2', 1152),
    ('random_split_3', 288), ('random_split_3', 576), ('random_split_3', 1152),
    ('random_split_4', 288), ('random_split_4', 576), ('random_split_4', 1152),
    ('random_split_5', 288), ('random_split_5', 576), ('random_split_5', 1152),
    ('random_split_6', 288), ('random_split_6', 576), ('random_split_6', 1152),
    ('random_split_7', 288), ('random_split_7', 576), ('random_split_7', 1152),
    ('random_split_8', 288), ('random_split_8', 576), ('random_split_8', 1152),
    ('random_split_9', 288), ('random_split_9', 576), ('random_split_9', 1152),
]



@app.command()
def pretrained():
    """
    Launch training and evaluation with a pretrained base model from rxnfp with same hyperparameters as for Buchwald Hartwig experiment.
    """
    launch_training_on_all_splits(experiment='full', splits=NAME_SPLIT, base_model='pretrained', dropout=0.7987, learning_rate=0.00009659)

@app.command()
def finetuned():
    """
    Launch training and evaluation with a finetuned base model from rxnfp with same hyperparameters as for Buchwald Hartwig experiment.
    """
    launch_training_on_all_splits(experiment='full', splits=NAME_SPLIT, base_model='ft', dropout=0.7304, learning_rate=0.0000976)

@app.command()
def discovery():
    """
    Launch training with reduced data set for reaction discovery experiments. 
    """
    launch_training_on_all_splits(experiment='discovery', splits=DISCOVERY_SPLIT, base_model='ft', dropout=0.7542, learning_rate=0.00009116)

@app.command()
def pretrained_tuned():
    """
    Launch training and evaluation with a pretrained base model from rxnfp with tuned hyperparameters.
    """
    launch_training_on_all_splits(experiment='full_tuned', splits=NAME_SPLIT, base_model='pretrained', dropout=0.5848, learning_rate=0.00005812)

@app.command()
def finetuned_tuned():
    """
    Launch training and evaluation with a finetuned base model from rxnfp with tuned hyperparameters.
    """
    launch_training_on_all_splits(experiment='full_tuned', splits=NAME_SPLIT, base_model='ft', dropout=0.7542, learning_rate=0.00009116)



def launch_training_on_all_splits(experiment: str, splits: List, base_model: str, dropout: float, learning_rate: float):
    project = f'suzuki_miyaura_training_{experiment}_{base_model}'
    model_args = {
    'wandb_project': project, 'num_train_epochs': 15, 'overwrite_output_dir': True,
    'learning_rate': learning_rate, 'gradient_accumulation_steps': 1,
    'regression': True, "num_labels":1, "fp16": False,
    "evaluate_during_training": False, 'manual_seed': 42,
    "max_seq_length": 300, "train_batch_size": 16,"warmup_ratio": 0.00,
    "config" : { 'hidden_dropout_prob': dropout } }    


    for (name, split) in splits:
        if wandb_available: wandb.init(name=name, project=project, reinit=True)
        df = pd.read_csv(f'../data/Suzuki-Miyaura/random_splits/{name}.tsv', sep='\t')

        train_df = df.iloc[:split][['rxn', 'y']] 
        test_df = df.iloc[split:][['rxn', 'y']] 

        train_df.columns = ['text', 'labels']
        test_df.columns = ['text', 'labels']

        mean = train_df.labels.mean()
        std = train_df.labels.std()

        train_df['labels'] = (train_df['labels'] - mean) / std
        test_df['labels'] = (test_df['labels'] - mean) / std

        model_path =  pkg_resources.resource_filename("rxnfp", f"models/transformers/bert_{base_model}")
        pretrained_bert = SmilesClassificationModel("bert", model_path, num_labels=1, args=model_args, use_cuda=torch.cuda.is_available())
        pretrained_bert.train_model(train_df, output_dir=f"outputs_suzuki_miyaura_{experiment}_{base_model}_{name}_split_{str(split).replace('-','_')}", eval_df=test_df, r2=sklearn.metrics.r2_score)
        if wandb_available: wandb.join()


if __name__ == '__main__':
    app()


