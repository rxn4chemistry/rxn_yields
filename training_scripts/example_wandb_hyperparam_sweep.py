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
    raise ValueError('Wandb is not available')

def main():
    def train():
        wandb.init()
        print("HyperParams=>>", wandb.config)
        model_args = {
        'wandb_project': "doyle_random_01",
        'num_train_epochs': 10,'overwrite_output_dir': True,
        'gradient_accumulation_steps': 1, "warmup_ratio": 0.00,
        "train_batch_size": 16, 'regression': True, "num_labels":1,
        "fp16": False, "evaluate_during_training": True,
        "max_seq_length": 300,  
        "config" : {
        'hidden_dropout_prob': wandb.config.dropout_rate, 
        },
        'learning_rate': wandb.config.learning_rate,
        }
        model_path =  pkg_resources.resource_filename("rxnfp", f"models/transformers/bert_pretrained")
        pretrained_bert = SmilesClassificationModel("bert", model_path, num_labels=1, args=model_args, use_cuda=torch.cuda.is_available())
        pretrained_bert.train_model(train_df, output_dir='outputs_pretrained_bert_doyle_random_01', 
            eval_df=test_df, r2=sklearn.metrics.r2_score)


    df_doyle = pd.read_excel('../data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx', sheet_name='FullCV_01')
    df_doyle['rxn'] = generate_buchwald_hartwig_rxns(df_doyle)

    train_df = df_doyle.iloc[:2373][['rxn', 'Output']]


    test_df = df_doyle.iloc[2373:2768][['rxn', 'Output']]
    train_df.columns = ['text', 'labels']
    test_df.columns = ['text', 'labels']

    mean = train_df.labels.mean()
    std = train_df.labels.std()

    train_df['labels'] = (train_df['labels'] - mean) / std
    test_df['labels'] = (test_df['labels'] - mean) / std

    sweep_config = {
        'method': 'bayes', #grid, random, bayes
        'metric': {
          'name': 'r2',
          'goal': 'maximize'   
        },
        'parameters': {  
            'learning_rate': {
                'min': 1e-6,
                'max': 1e-4
                
            },
            'dropout_rate': {
                'min': 0.05,
                'max': 0.8
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="doyle_random_01_hyperparams_sweep")
    wandb.agent(sweep_id,function=train)

if __name__ == '__main__':
    main()
