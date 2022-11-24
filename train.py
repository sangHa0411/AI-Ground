
import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
import copy 
from model.bert import Bert
from model.config import BertConfig
from torch.utils.data import DataLoader
from utils.preprocessor import Spliter, preprocess, parse
from utils.collator import DataCollatorWithMasking, DataCollatorWithPadding
from trainer import Trainer

import warnings

def train(args) :

    # -- Ignore Warnings
    warnings.filterwarnings('ignore')

    cuda_flag = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_flag else "cpu")
    print("\nDevice:", device)

    # -- Seed
    seed_everything(args.seed)

    # -- Load dataset
    history_data_df = pd.read_csv(os.path.join(args.data_dir, args.history_data_file), encoding='utf-8')
    profile_data_df = pd.read_csv(os.path.join(args.data_dir, args.profile_data_file), encoding='utf-8')
    meta_data_df = pd.read_csv(os.path.join(args.data_dir, args.meta_data_file), encoding='utf-8')

    # -- Preprocess dataset
    df = preprocess(history_data_df, meta_data_df)
    dataset = parse(df)
    print(dataset)
    
    # -- Model Arguments
    max_album_value = max(df['album_id'].unique())
    max_genre_value = max(df['genre'].unique())
    max_country_value = max(df['country'].unique())

    # -- Token dictionary
    special_token_dict = {
        'country_pad_token_id' : max_country_value+1,
        'country_mask_token_id' : max_country_value+2,
        'genre_pad_token_id' : max_genre_value+1,
        'genre_mask_token_id' : max_genre_value+2,
        'album_pad_token_id' : max_album_value+1,
        'album_mask_token_id' : max_album_value+2,
    }
    
    album_size = max_album_value + 3
    genre_size = max_genre_value + 3
    country_size = max_country_value + 3

    model_config = BertConfig(
        album_size=album_size,
        genre_size=genre_size,
        country_size=country_size,
        age_size=len(profile_data_df['age'].unique()),
        gender_size=len(profile_data_df['sex'].unique()),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        max_length=args.max_length,
        num_attention_heads=args.num_head,
        intermediate_size=args.intermediate_size,
        hidden_dropout_prob=args.dropout_prob,
        attention_probs_dropout_prob=args.dropout_prob,
        max_position_embeddings=args.max_length,
    )

    # -- Model
    num_labels = max_album_value + 1
    model_config.vocab_size = num_labels
    model = Bert(model_config)

    if args.do_eval :
    
        spliter = Spliter(leave_probability=args.leave_probability)
        dataset = dataset.map(spliter, batched=True, num_proc=args.num_workers)

        train_dataset = dataset 
        print(train_dataset)

        eval_dataset = copy.deepcopy(dataset)
        eval_dataset = eval_dataset.filter(lambda x : len(x['labels']) > 0)
        print(eval_dataset)

        # -- Train Data Collator
        train_data_collator = DataCollatorWithMasking(
            special_token_dict=special_token_dict,
            max_length=args.max_length,
            mlm=True,
            mlm_probability=args.mlm_probability,
        )

        # -- Data Loader 
        train_data_loader = DataLoader(
            train_dataset, 
            batch_size=args.train_batch_size, 
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=train_data_collator
        )

        # -- Eval Data Collator
        eval_data_collator = DataCollatorWithPadding(
            special_token_dict=special_token_dict,
            max_length=args.max_length,
        )

        # -- Data Loader 
        eval_data_loader = DataLoader(
            eval_dataset, 
            batch_size=args.eval_batch_size, 
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=eval_data_collator
        )

        trainer = Trainer(args, model, device, train_data_loader, eval_data_loader)
        trainer.train()

    else :

        print(dataset)

        # -- Train Data Collator
        train_data_collator = DataCollatorWithMasking(
            special_token_dict=special_token_dict,
            max_length=args.max_length,
            mlm=True,
            mlm_probability=args.mlm_probability,
        )

        # -- Data Loader 
        train_data_loader = DataLoader(
            dataset, 
            batch_size=args.train_batch_size, 
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=train_data_collator
        )
            
        trainer = Trainer(args, model, device, train_data_loader)
        trainer.train()

 

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':

    # input options
    parser = argparse.ArgumentParser(description='Upsage - (Ai Ground)')
    parser.add_argument('--seed', type=int, 
        default=42, 
        help='random seed'
    )
    parser.add_argument('--data_dir', type=str,
        default='../data',
        help='data directory'
    )
    parser.add_argument('--do_eval', type=bool,
        default=False,
        help='validation falg'
    )
    parser.add_argument('--meta_data_file', type=str,
        default='meta_data.csv',
        help='metadata csv file'
    )
    parser.add_argument('--profile_data_file', type=str,
        default='profile_data.csv',
        help='profile data csv file'
    )
    parser.add_argument('--history_data_file', type=str,
        default='history_data.csv',
        help='history data csv file'
    )
    parser.add_argument('--max_length', type=int,
        default=256,
        help='max length of albums'
    )
    parser.add_argument('--learning_rate', type=float,
        default=5e-5,
        help='learning rate'
    )
    parser.add_argument('--weight_decay', type=float,
        default=1e-3,
        help='weight decay'
    )
    parser.add_argument('--train_batch_size', type=int,
        default=32,
        help='batch size for training'
    )
    parser.add_argument('--eval_batch_size', type=int,
        default=16,
        help='batch size for training'
    )
    parser.add_argument('--epochs', type=int,
        default=10,
        help='epochs to training'
    )
    parser.add_argument('--max_steps', type=int,
        default=-1,
        help='max steps to training'
    )
    parser.add_argument('--leave_probability', type=float,
        default=0.2,
        help='validation_ratio'
    )
    parser.add_argument('--mlm_probability', type=float,
        default=0.15,
        help='masking ratio during training'
    )
    parser.add_argument('--num_workers', type=int,
        default=4,
        help='the number of workers for data loader'
    )
    parser.add_argument('--logging_steps', type=int,
        default=100,
        help='logging steps'
    )
    parser.add_argument('--eval_steps', type=int,
        default=500,
        help='eval steps'
    )
    parser.add_argument('--save_steps', type=int,
        default=500,
        help='save steps'
    )
    parser.add_argument('--hidden_size', type=int,
        default=64,
        help='d_model'
    )
    parser.add_argument('--intermediate_size', type=int,
        default=256,
        help='hidden_size'
    )
    parser.add_argument('--num_head', type=int,
        default=8,
        help='num_head'
    )
    parser.add_argument('--num_layers', type=int,
        default=6,
        help='num_layers'
    )
    parser.add_argument('--dropout_prob', type=float,
        default=0.1,
        help='dropout_prob'
    )
    parser.add_argument('--warmup_ratio', type=float,
        default=0.05,
        help='warmup_ratio'
    )
    parser.add_argument('--eps', type=float,
        default=1e-6,
        help='norm_prob'
    )
    parser.add_argument('--save_dir', type=str,
        default='./exps',
        help='save_directory'
    )
    args = parser.parse_args()
    train(args)