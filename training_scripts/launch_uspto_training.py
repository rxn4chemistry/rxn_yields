import typer
import torch
import pkg_resources
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import pandas as pd
from typing import List, Tuple

from rxn_yields.core import SmilesClassificationModel
import sklearn
try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False

app = typer.Typer()


@app.command()
def gram():
    """
    Launch training and evaluation with a pretrained base model from rxnfp on USPTO gram scale data.
    """
    launch_training(experiment='gram', split='random_split', base_model='pretrained', dropout=0.5237, learning_rate=0.00001562)

@app.command()
def gram_randomized():
    """
    Launch training and evaluation with a pretrained base model from rxnfp on USPTO gram scale data. Sanity check randomized yields.
    """
    launch_training(experiment='gram', split='random_split', base_model='pretrained', dropout=0.5237, learning_rate=0.00001562, sanity_check=True)

@app.command()
def gram_smoothed():
    """
    Launch training and evaluation with a pretrained base model from rxnfp on USPTO gram scale data. Smoothed yields.
    """
    launch_training(experiment='gram_smooth', split='random_split', base_model='pretrained', dropout=0.5237, learning_rate=0.00001562)

@app.command()
def gram_time_split():
    """
    Launch training and evaluation with a pretrained base model from rxnfp on USPTO gram scale data.
    """
    launch_training(experiment='gram', split='random_split', base_model='pretrained', dropout=0.5237, learning_rate=0.00001562)


@app.command()
def milligram():
    """
    Launch training and evaluation with a pretrained base model from rxnfp on USPTO milligram scale data.
    """
    launch_training(experiment='milligram', split='random_split', base_model='pretrained', dropout=0.5826, learning_rate=0.00002958)

@app.command()
def milligram_randomized():
    """
    Launch training and evaluation with a pretrained base model from rxnfp on USPTO milligram scale data. Sanity check randomized yields.
    """
    launch_training(experiment='milligram', split='random_split', base_model='pretrained', dropout=0.5826, learning_rate=0.00002958, sanity_check=True)

@app.command()
def milligram_smoothed():
    """
    Launch training and evaluation with a pretrained base model from rxnfp on USPTO milligram scale data. Smoothed yields.
    """
    launch_training(experiment='milligram_smooth', split='random_split', base_model='pretrained', dropout=0.5826, learning_rate=0.00002958)

@app.command()
def milligram_time_split():
    """
    Launch training and evaluation with a pretrained base model from rxnfp on USPTO milligram scale data. Time split.
    """
    launch_training(experiment='milligram', split='random_split', base_model='pretrained', dropout=0.5826, learning_rate=0.00002958)


def launch_training(experiment: str, split: str, base_model: str, dropout: float, learning_rate: float, sanity_check: bool = False):
    project = f'uspto_{experiment}_{split}_{base_model}'
    if sanity_check: project += '_randomized_yields'
    model_args = {
    'wandb_project': project, 'num_train_epochs': 15, 'overwrite_output_dir': True,
    'learning_rate': learning_rate, 'gradient_accumulation_steps': 1,
    'regression': True, "num_labels":1, "fp16": False,
    "evaluate_during_training": False, 'manual_seed': 42,
    "max_seq_length": 300, "train_batch_size": 16,"warmup_ratio": 0.00,
    "config" : { 'hidden_dropout_prob': dropout } }
        
    train_df = pd.read_csv(f'../data/uspto/{experiment}_train_{split}.tsv', sep='\t', index_col=0)
    train_df.columns = ["text", "labels"]
    test_df = pd.read_csv(f'../data/uspto/{experiment}_test_{split}.tsv', sep='\t', index_col=0)
    test_df.columns = ["text", "labels"]
    mean = train_df.labels.mean()
    std = train_df.labels.std()
    print(mean, std)

    train_df['labels'] = (train_df['labels'] - mean) / std
    test_df['labels'] = (test_df['labels'] - mean) / std


    train_df = train_df.sample(frac=1., random_state=42)

    if sanity_check: train_df['labels'] = train_df['labels'].sample(frac=1.0, random_state=21).values

    model_path =  pkg_resources.resource_filename("rxnfp", f"models/transformers/bert_{base_model}")
    pretrained_bert = SmilesClassificationModel("bert", model_path, num_labels=1, args=model_args, use_cuda=torch.cuda.is_available())
    pretrained_bert.train_model(train_df, output_dir=f"uspto_{experiment}_{split}_{base_model}", eval_df=test_df, r2=sklearn.metrics.r2_score)


if __name__ == '__main__':
    app()

