
import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from model.model import Bert
from model.config import BertConfig
from utils.preprocessor import preprocess, parse
from utils.collator import DataCollatorWithMasking
from torch.utils.data import DataLoader
from utils.scheduler import LinearWarmupScheduler
from sklearn.metrics import accuracy_score
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

    # -- Token dictionary
    special_token_dict = {
        'country_pad_token_id' : 1,
        'country_mask_token_id' : 21,
        'genre_pad_token_id' : 9,
        'genre_mask_token_id' : 26,
        'album_pad_token_id' : 0,
        'album_mask_token_id' : 1,
    }

    # -- Data Collator
    data_collator = DataCollatorWithMasking(
        profile_data=profile_data_df, 
        special_token_dict=special_token_dict,
        max_length=args.max_length,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    # -- Model Arguments
    album_size = max(df['album_id'].unique()) + 1
    genre_size = max(df['genre'].unique()) + 1
    country_size = max(df['country'].unique()) + 2

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

        # -- Split
        datasets = dataset.train_test_split(test_size=0.1)

        # -- Data Loader 
        train_data_loader = DataLoader(
            datasets['train'], 
            batch_size=args.train_batch_size, 
            shuffle=True,
            collate_fn=data_collator
        )

        # -- Data Loader 
        eval_data_loader = DataLoader(
            datasets['test'], 
            batch_size=args.eval_batch_size, 
            shuffle=False,
            collate_fn=data_collator
        )

        # -- Training
        train_data_iterator = iter(train_data_loader)
        total_steps = len(train_data_loader) * args.epochs
        warmup_steps = int(total_steps * args.warmup_ratio)

        loss_fn = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = LinearWarmupScheduler(optimizer, total_steps, warmup_steps)

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
                print('Step : %d \t Loss : %.5f, Learning Rate : %f' %(step, loss, current_lr))

            if step % args.eval_steps == 0 and step > 0 :
                print('Validation at %dstep' %step)
                eval_loss, eval_acc = 0.0, 0.0
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

                    predictions = torch.argmax(logits, -1)
                    labels = data['labels'].long().to(device)

                    candidates = torch.where(labels==-100, False, True)
                    rights = torch.where(labels==predictions, True, False)

                    acc = torch.sum(rights & candidates) / torch.sum(candidates)
                    eval_acc += acc.item()

                    loss = loss_fn(logits.view(-1, album_size), labels.view(-1,))
                    eval_loss += loss.item()

                eval_acc /= len(eval_data_loader)
                eval_loss /= len(eval_data_loader)
                print('Step : %d \t Eval Loss : %.5f, Eval Accuracy : %.5f' %(step, eval_loss, eval_acc))

                # model_path = os.path.join(args.save_path, f'checkpoint-{step}.pt')        
                # torch.save(model.state_dict(), model_path)
    else :
        # -- Data Loader 
        train_data_loader = DataLoader(
            dataset, 
            batch_size=args.train_batch_size, 
            shuffle=True,
            collate_fn=data_collator
        )

        # -- Training
        train_data_iterator = iter(train_data_loader)
        total_steps = len(train_data_loader) * args.epochs
        warmup_steps = int(total_steps * args.warmup_ratio)

        loss_fn = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = LinearWarmupScheduler(optimizer, total_steps, warmup_steps)

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
                print('Step : %d \t Loss : %.5f, Learning Rate : %f' %(step, loss, current_lr))

            if step % args.save_steps == 0 and step > 0 :
                model_path = os.path.join(args.save_path, f'checkpoint-{step}.pt')        
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
    parser = argparse.ArgumentParser(description='AI Fashion Coordinator.')
    parser.add_argument('--seed', type=int, 
        default=42, 
        help='random seed'
    )
    parser.add_argument('--data_dir', type=str,
        default='../data',
        help='data directory'
    )
    parser.add_argument('--do_eval', type=bool,
        default=True,
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