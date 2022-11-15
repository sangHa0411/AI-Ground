
import os
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from model.model import Bert
from model.config import BertConfig
from torch.utils.data import DataLoader
from utils.preprocessor import preprocess, parse
from utils.collator import DataCollatorWithPadding
import warnings

TOPK = 25
ENSEMBLE_SIZE = 3
MODEL_PATHS = [
    'exps/seed42/checkpoint-3250.pt',
    'exps/seed1234/checkpoint-2600.pt',
    'exps/seed95/checkpoint-4000.pt',
    ]

def train(args) :

    # -- Ignore Warnings
    warnings.filterwarnings('ignore')

    # -- Device
    cuda_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(cuda_str)
    print("\nDevice:", device)

    # -- Load dataset
    history_data_df = pd.read_csv(os.path.join(args.data_dir, args.history_data_file), encoding='utf-8')
    profile_data_df = pd.read_csv(os.path.join(args.data_dir, args.profile_data_file), encoding='utf-8')
    meta_data_df = pd.read_csv(os.path.join(args.data_dir, args.meta_data_file), encoding='utf-8')

    ids = list(profile_data_df['profile_id'])

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

    # -- Data Collator
    data_collator = DataCollatorWithPadding(
        profile_data=profile_data_df, 
        special_token_dict=special_token_dict,
        max_length=args.max_length,
    )

    # -- Model
    num_labels = max_album_value + 1
    model_config.num_labels = num_labels

    prediction_logits = []
    for i in range(ENSEMBLE_SIZE) :    
        model_path = MODEL_PATHS[i]
        print('\nLoading Model : %s' %model_path)

        model = Bert(model_config).to(device)
        model.load_state_dict(torch.load(model_path, map_location=cuda_str))

        # -- Loader 
        data_loader = DataLoader(
            dataset, 
            batch_size=args.eval_batch_size, 
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=data_collator
        )

        model.eval()
        sub_predictions = []
        with torch.no_grad() :
            for data in tqdm(data_loader) :
                
                age_input, gender_input = data['age'], data['gender']
                age_input = age_input.long().to(device)
                gender_input = gender_input.long().to(device)

                album_input, genre_input, country_input = data['album_input'], data['genre_input'], data['country_input']
                album_input = album_input.long().to(device)
                genre_input = genre_input.long().to(device)
                country_input = country_input.long().to(device)

                logits = model(
                    album_input=album_input, 
                    genre_input=genre_input,
                    country_input=country_input,
                    age_input=age_input,
                    gender_input=gender_input,
                )

                logits = F.softmax(logits[:, -1, :], dim=-1).detach().cpu().numpy().tolist()
                sub_predictions.extend(logits)
        
        sub_predictions = np.array(sub_predictions)
        prediction_logits.append(sub_predictions)

    prediction_logit = np.mean(prediction_logits, axis=0)
    pred_args = np.argsort(-prediction_logit, axis=-1)

    predictions = {}
    for i, p in zip(ids, pred_args) :
        predictions[i] = p[:TOPK]

    submission_df = pd.read_csv(os.path.join(args.data_dir, args.submission_file))
    submission_predictions = []
    for p_id in tqdm(submission_df['profile_id']) :
        submission_predictions.append(predictions[p_id].tolist())

    submission_df['predicted_list'] = submission_predictions
    submission_df.to_csv(args.output_file, index=False)


if __name__ == '__main__':

    # input options
    parser = argparse.ArgumentParser(description='Upsage - (Ai Ground)')
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
    parser.add_argument('--data_dir', type=str,
        default='../data',
        help='data directory'
    )
    parser.add_argument('--submission_file', type=str,
        default='sample_submission.csv',
        help='submission csv file'
    )
    parser.add_argument('--max_length', type=int,
        default=256,
        help='max length of albums'
    )
    parser.add_argument('--eval_batch_size', type=int,
        default=16,
        help='batch size for training'
    )
    parser.add_argument('--num_workers', type=int,
        default=4,
        help='the number of workers for data loader'
    )
    parser.add_argument('--model_path', type=str,
        default='./exps/checkpoint-3500.pt',
        help='saved model path'
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
    parser.add_argument('--output_file', type=str,
        default='./results/submission.csv',
        help='prediction directory'
    )
    args = parser.parse_args()
    train(args)