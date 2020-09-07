import typer
import torch
import pkg_resources
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import pandas as pd
from typing import List, Tuple

from rxn_yields.core import SmilesClassificationModel
from rxn_yields.data import generate_buchwald_hartwig_rxns
import sklearn
try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False

app = typer.Typer()

NAME_SPLIT = [
    ('FullCV_01', 2768), ('FullCV_02', 2768), ('FullCV_03', 2768), ('FullCV_04', 2768), ('FullCV_05', 2768),
    ('FullCV_06', 2768), ('FullCV_07', 2768), ('FullCV_08', 2768), ('FullCV_09', 2768), ('FullCV_10', 2768),
    ('Test1', 3058), ('Test2', 3056), ('Test3', 3059), ('Test4', 3056),
    ('Plates1-3', '1-1075'), ('Plates1-3', '1076-2515'), ('Plates1-3', '2516-3955'), ('Plate2_new', '1076-2515')
]

DISCOVERY_SPLIT = [
    ('FullCV_01', 208), ('FullCV_01', 415), ('FullCV_01', 829),
    ('FullCV_02', 208), ('FullCV_02', 415), ('FullCV_02', 829),
    ('FullCV_03', 208), ('FullCV_03', 415), ('FullCV_03', 829),
    ('FullCV_04', 208), ('FullCV_04', 415), ('FullCV_04', 829),
    ('FullCV_05', 208), ('FullCV_05', 415), ('FullCV_05', 829),
    ('FullCV_06', 208), ('FullCV_06', 415), ('FullCV_06', 829),
    ('FullCV_07', 208), ('FullCV_07', 415), ('FullCV_07', 829),
    ('FullCV_08', 208), ('FullCV_08', 415), ('FullCV_08', 829),
    ('FullCV_09', 208), ('FullCV_09', 415), ('FullCV_09', 829),
    ('FullCV_10', 208), ('FullCV_10', 415), ('FullCV_10', 829),
]

@app.command()
def pretrained():
    """
    Launch training and evaluation with a pretrained base model from rxnfp.
    """
    launch_training_on_all_splits(experiment='full', splits=NAME_SPLIT, base_model='pretrained', dropout=0.7987, learning_rate=0.00009659)

@app.command()
def finetuned():
    """
    Launch training and evaluation with a finetuned base model from rxnfp.
    """
    launch_training_on_all_splits(experiment='full', splits=NAME_SPLIT, base_model='ft', dropout=0.7304, learning_rate=0.0000976)

@app.command()
def discovery():
    """
    Launch training with reduced data set for reaction discovery experiments. 
    """
    launch_training_on_all_splits(experiment='discovery', splits=DISCOVERY_SPLIT, base_model='pretrained', dropout=0.7987, learning_rate=0.00009659)

def launch_training_on_all_splits(experiment: str, splits: List, base_model: str, dropout: float, learning_rate: float):
    project = f'buchwald_hartwig_training_{experiment}_{base_model}'
    model_args = {
    'wandb_project': project, 'num_train_epochs': 15, 'overwrite_output_dir': True,
    'learning_rate': learning_rate, 'gradient_accumulation_steps': 1,
    'regression': True, "num_labels":1, "fp16": False,
    "evaluate_during_training": False, 'manual_seed': 42,
    "max_seq_length": 300, "train_batch_size": 16,"warmup_ratio": 0.00,
    "config" : { 'hidden_dropout_prob': dropout } }
        
    for (name, split) in splits:
        if wandb_available: wandb.init(name=name, project=project, reinit=True)

        df_doyle = pd.read_excel('../data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx', sheet_name=name)
        df_doyle['rxn'] = generate_buchwald_hartwig_rxns(df_doyle)

        if name.startswith("Plate"):
            start, end = split.split('-')
            start = int(start)
            end = int(end)
            test_df = df_doyle.iloc[start-1:end-1][['rxn', 'Output']] # paper has starting index 1 not 0

            train_df = df_doyle[~df_doyle.index.isin(test_df.index)][['rxn', 'Output']]

        else:
            train_df = df_doyle.iloc[:split-1][['rxn', 'Output']] # paper has starting index 1 not 0
            test_df = df_doyle.iloc[split-1:][['rxn', 'Output']] # paper has starting index 1 not 0

        train_df.columns = ['text', 'labels']
        test_df.columns = ['text', 'labels']
        mean = train_df.labels.mean()
        std = train_df.labels.std()
        train_df['labels'] = (train_df['labels'] - mean) / std
        test_df['labels'] = (test_df['labels'] - mean) / std

        model_path =  pkg_resources.resource_filename("rxnfp", f"models/transformers/bert_{base_model}")
        pretrained_bert = SmilesClassificationModel("bert", model_path, num_labels=1, args=model_args, use_cuda=torch.cuda.is_available())
        pretrained_bert.train_model(train_df, output_dir=f"outputs_buchwald_hartwig_{experiment}_{base_model}_{name}_split_{str(split).replace('-','_')}", eval_df=test_df, r2=sklearn.metrics.r2_score)
        if wandb_available: wandb.join() # multiple runs in same script

if __name__ == '__main__':
    app()
