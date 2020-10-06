import typer
import torch
import pkg_resources
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import pandas as pd
from typing import List, Tuple

from rxnfp.models import SmilesClassificationModel
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
    # ('Plates1-3', '1-1075'), ('Plates1-3', '1076-2515'), ('Plates1-3', '2516-3955'), ('Plate2_new', '1076-2515')
]

DISCOVERY_SPLIT = [
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
    'wandb_project': project, 'num_train_epochs': 10, 'overwrite_output_dir': True,
    'learning_rate': learning_rate, 'gradient_accumulation_steps': 1,
    'regression': True, "num_labels":1, "fp16": False,
    "evaluate_during_training": True, 'manual_seed': 42,
    "max_seq_length": 300, "train_batch_size": 16,"warmup_ratio": 0.00,
    "config" : { 'hidden_dropout_prob': dropout } }
        
    for (name, split) in splits:
        if wandb_available: wandb.init(name=name, project=project, reinit=True)

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

        model_path =  pkg_resources.resource_filename("rxnfp", f"models/transformers/bert_{base_model}")
        pretrained_bert = SmilesClassificationModel("bert", model_path, num_labels=1, args=model_args, use_cuda=torch.cuda.is_available())
        pretrained_bert.train_model(train_df, output_dir=f"outputs_buchwald_hartwig_{experiment}_{base_model}_{name}_split_{str(split).replace('-','_')}", eval_df=test_df, r2=sklearn.metrics.r2_score)
        if wandb_available: wandb.join() # multiple runs in same script

if __name__ == '__main__':
    app()
