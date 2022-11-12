
import os
import wandb
import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from model.model import Bert
from model.config import BertConfig
from torch.utils.data import DataLoader
from utils.metrics import compute_metrics
from utils.preprocessor import preprocess, parse
from utils.collator import DataCollatorWithMasking, DataCollatorWithPadding
from utils.scheduler import LinearWarmupScheduler
from dotenv import load_dotenv
import warnings

def train(args) :

    # -- Ignore Warnings
    warnings.filterwarnings('ignore')

    # -- Device
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

    model_arguments = BertConfig(
        album_size=album_size,
        genre_size=genre_size,
        country_size=country_size,
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
    model = Bert(model_arguments).to(device)


    if args.do_eval :

        # -- Train Data Collator
        train_data_collator = DataCollatorWithMasking(
            profile_data=profile_data_df, 
            special_token_dict=special_token_dict,
            max_length=args.max_length,
            mlm=True,
            mlm_probability=args.mlm_probability,
            eval_flag=True,
        )

        # -- Eval Data Collator
        eval_data_collator = DataCollatorWithPadding(
            profile_data=profile_data_df, 
            special_token_dict=special_token_dict,
            max_length=args.max_length,
            eval_flag=True,
        )

        # -- Data Loader 
        train_data_loader = DataLoader(
            dataset, 
            batch_size=args.train_batch_size, 
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=train_data_collator
        )

        # -- Data Loader 
        eval_data_loader = DataLoader(
            dataset, 
            batch_size=args.eval_batch_size, 
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=eval_data_collator
        )

        # -- Training
        train_data_iterator = iter(train_data_loader)
        total_steps = len(train_data_loader) * args.epochs
        warmup_steps = int(total_steps * args.warmup_ratio)

        loss_fn = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = LinearWarmupScheduler(optimizer, total_steps, warmup_steps)

        load_dotenv(dotenv_path="wandb.env")
        WANDB_AUTH_KEY = os.getenv("WANDB_AUTH_KEY")
        wandb.login(key=WANDB_AUTH_KEY)

        name = f"EP:{args.epochs}_BS:{args.train_batch_size}_LR:{args.learning_rate}_WR:{args.warmup_ratio}_WD:{args.weight_decay}"
        wandb.init(
            entity="sangha0411",
            project="bert4rec",
            group=f"ai-ground",
            name=name
        )

        training_args = {
            "epochs": args.epochs, 
            "total_steps" : total_steps,
            "warmup_steps" : warmup_steps,
            "batch_size": args.train_batch_size, 
            "learning_rate": args.learning_rate, 
            "weight_decay": args.weight_decay, 
        }
        wandb.config.update(training_args)


        print('\nTraining')
        for step in tqdm(range(total_steps)) :

            try :
                data = next(train_data_iterator)
            except StopIteration :
                train_data_iterator = iter(train_data_loader)
                data = next(train_data_iterator)

            optimizer.zero_grad()

            album_input, genre_input, country_input = data['album_input'], data['genre_input'], data['country_input']
            album_input = album_input.long().to(device)
            genre_input = genre_input.long().to(device)
            country_input = country_input.long().to(device)

            logits = model(
                album_input=album_input, 
                genre_input=genre_input,
                country_input=country_input
            )

            labels = data['labels'].long().to(device)
            loss = loss_fn(logits.view(-1, album_size), labels.view(-1,))
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % args.logging_steps == 0 and step > 0 :
                current_lr = scheduler.get_last_lr()[0]
                log = {'train/step' : step, 'train/loss' : loss.item(), 'train/lr' : current_lr}
                wandb.log(log)
                print(log)

            if step % args.eval_steps == 0 and step > 0 :

                model.eval()

                with torch.no_grad() :
                    print('\nValidation at %d step' %step)
                    eval_predictions, eval_labels = [], []
                    for eval_data in tqdm(eval_data_loader) :

                        album_input, genre_input, country_input = eval_data['album_input'], eval_data['genre_input'], eval_data['country_input']
                        album_input = album_input.long().to(device)
                        genre_input = genre_input.long().to(device)
                        country_input = country_input.long().to(device)

                        logits = model(
                            album_input=album_input, 
                            genre_input=genre_input,
                            country_input=country_input
                        )

                        logits = logits[album_input==special_token_dict['album_mask_token_id']]

                        eval_predictions.extend(logits.detach().cpu().numpy().tolist())
                        eval_labels.extend(eval_data['labels'].detach().cpu().numpy().tolist())
                    
                    eval_log = compute_metrics(eval_predictions, eval_labels)
                    eval_log = {'eval/' + k : v for k, v in eval_log.items()}
                    wandb.log(eval_log)
                    print(eval_log)

                model.train()

        # Evaluation
        model.eval()
        with torch.no_grad() :
            eval_predictions, eval_labels = [], []
            for eval_data in tqdm(eval_data_loader) :

                album_input, genre_input, country_input = eval_data['album_input'], eval_data['genre_input'], eval_data['country_input']
                album_input = album_input.long().to(device)
                genre_input = genre_input.long().to(device)
                country_input = country_input.long().to(device)

                logits = model(
                    album_input=album_input, 
                    genre_input=genre_input,
                    country_input=country_input
                )

                logits = logits[album_input==special_token_dict['album_mask_token_id']]

                eval_predictions.extend(logits.detach().cpu().numpy().tolist())
                eval_labels.extend(eval_data['labels'].detach().cpu().numpy().tolist())
            
            eval_log = compute_metrics(eval_predictions, eval_labels)
            eval_log = {'eval/' + k : v for k, v in eval_log.items()}
            wandb.log(eval_log)
            print(eval_log)
    
    else :

        # -- Train Data Collator
        train_data_collator = DataCollatorWithMasking(
            profile_data=profile_data_df, 
            special_token_dict=special_token_dict,
            max_length=args.max_length,
            mlm=True,
            mlm_probability=args.mlm_probability,
            eval_flag=False,
        )

        # -- Data Loader 
        train_data_loader = DataLoader(
            dataset, 
            batch_size=args.train_batch_size, 
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=train_data_collator
        )

        # -- Training
        train_data_iterator = iter(train_data_loader)
        total_steps = len(train_data_loader) * args.epochs
        warmup_steps = int(total_steps * args.warmup_ratio)

        loss_fn = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = LinearWarmupScheduler(optimizer, total_steps, warmup_steps)

        load_dotenv(dotenv_path="wandb.env")
        WANDB_AUTH_KEY = os.getenv("WANDB_AUTH_KEY")
        wandb.login(key=WANDB_AUTH_KEY)

        name = f"EP:{args.epochs}_BS:{args.train_batch_size}_LR:{args.learning_rate}_WR:{args.warmup_ratio}_WD:{args.weight_decay}"
        wandb.init(
            entity="sangha0411",
            project="bert4rec",
            group=f"ai-ground",
            name=name
        )

        training_args = {
            "epochs": args.epochs, 
            "total_steps" : total_steps,
            "warmup_steps" : warmup_steps,
            "batch_size": args.train_batch_size, 
            "learning_rate": args.learning_rate, 
            "weight_decay": args.weight_decay, 
        }
        wandb.config.update(training_args)

        print('\nTraining')
        for step in tqdm(range(total_steps)) :

            try :
                data = next(train_data_iterator)
            except StopIteration :
                train_data_iterator = iter(train_data_loader)
                data = next(train_data_iterator)

            optimizer.zero_grad()

            album_input, genre_input, country_input = data['album_input'], data['genre_input'], data['country_input']
            album_input = album_input.long().to(device)
            genre_input = genre_input.long().to(device)
            country_input = country_input.long().to(device)

            logits = model(
                album_input=album_input, 
                genre_input=genre_input,
                country_input=country_input
            )

            labels = data['labels'].long().to(device)
            loss = loss_fn(logits.view(-1, album_size), labels.view(-1,))
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % args.logging_steps == 0 and step > 0 :
                current_lr = scheduler.get_last_lr()[0]
                log = {'step' : step, 'loss' : loss.item(), 'lr' : current_lr}
                wandb.log(log)
                print(log)
            
            if step % args.save_steps == 0 and step > 0 :
                model_path = os.path.join(args.save_dir, f'checkpoint-{step}.pt')        
                torch.save(model.state_dict(), model_path)
  
        model_path = os.path.join(args.save_dir, f'checkpoint-{total_steps}.pt')        
        torch.save(model.state_dict(), model_path)

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