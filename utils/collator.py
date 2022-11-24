
import torch
import random
import numpy as np
from tqdm import tqdm


class DataCollatorWithMasking :
    def __init__(
        self, 
        special_token_dict,
        max_length,
        mlm=True,
        mlm_probability=0.15,
        label_pad_token_id=-100,
    ) : 
        self.special_token_dict = special_token_dict
        self.max_length = max_length
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.label_pad_token_id = label_pad_token_id


    def __call__(self, dataset) :
        albums, genres, countries = [], [], []

        # max_length = min(self.max_length, max([len(d['album']) for d in dataset]))
        max_length = self.max_length

        for data in dataset :
            album, genre, country = data['album'], data['genre'], data['country']

            if len(album) < max_length :
                pad_length = max_length - len(album)
                album = [self.special_token_dict['album_pad_token_id']] * pad_length + album
                genre = [self.special_token_dict['genre_pad_token_id']] * pad_length + genre
                country = [self.special_token_dict['country_pad_token_id']] * pad_length + country  
            else :
                album = album[-max_length:]
                genre = genre[-max_length:]
                country = country[-max_length:]


            albums.append(album)
            genres.append(genre)
            countries.append(country)

        album_tensor = torch.tensor(albums, dtype=torch.int32)
        genre_tensor = torch.tensor(genres, dtype=torch.int32)
        country_tensor = torch.tensor(countries, dtype=torch.int32)
        
        album_tensor, genre_tensor, country_tensor, label_tensor = self.torch_mask(
            album_tensor,
            genre_tensor,
            country_tensor,
            self.special_token_dict
        )

        batch = {
            'album_input' : album_tensor, 
            'labels' : label_tensor, 
            'genre_input' : genre_tensor,
            'country_input' : country_tensor,
        }
        return batch


    def torch_mask(self, 
        album_tensor,
        genre_tensor,
        country_tensor,
        speical_token_dict, 
    ) :

        batch_size, seq_size = album_tensor.shape
        
        last_mask_tensor = torch.zeros(seq_size)
        last_mask_tensor[-1] = 1.0

        last_mask_indices = random.sample(range(batch_size), int(batch_size * 0.5))

        label_tensor = album_tensor.clone()
        pad_token_id = speical_token_dict['album_pad_token_id']
        
        probability_matrix = torch.full(label_tensor.shape, self.mlm_probability)
        probability_matrix[last_mask_indices] = last_mask_tensor

        probability_matrix = torch.where(
            label_tensor == pad_token_id, 0.0, self.mlm_probability
        )

        masked_indices = torch.bernoulli(probability_matrix).bool()

        label_tensor[~masked_indices] = -100
        album_tensor[masked_indices] = speical_token_dict['album_mask_token_id']
        genre_tensor[masked_indices] = speical_token_dict['genre_mask_token_id']
        country_tensor[masked_indices] = speical_token_dict['country_mask_token_id']

        return album_tensor, genre_tensor, country_tensor, label_tensor 


class DataCollatorWithPadding :
    def __init__(
        self, 
        special_token_dict,
        max_length,
    ) : 
        self.special_token_dict = special_token_dict
        self.max_length = max_length

    def __call__(self, dataset) :
        ids, albums, genres, countries = [], [], [], []

        labels = []
        # max_length = min(self.max_length, max([len(d['album']) for d in dataset]))
        max_length = self.max_length

        for data in dataset :
            d_id = data['id']
            ids.append(d_id)
            
            album = data['album'] + [self.special_token_dict['album_mask_token_id']]
            genre = data['genre'] + [self.special_token_dict['genre_mask_token_id']]
            country = data['country'] + [self.special_token_dict['country_mask_token_id']]

            if len(album) < max_length :
                pad_length = max_length - len(album)
                album = [self.special_token_dict['album_pad_token_id']] * pad_length + album
                genre = [self.special_token_dict['genre_pad_token_id']] * pad_length + genre
                country = [self.special_token_dict['country_pad_token_id']] * pad_length + country
            else :
                album = album[-max_length:]
                genre = genre[-max_length:]
                country = country[-max_length:]

            if 'labels' in data :
                labels.append(data['labels'])

            albums.append(album)
            genres.append(genre)
            countries.append(country)

        id_tensor = torch.tensor(ids, dtype=torch.int32)
        album_tensor = torch.tensor(albums, dtype=torch.int32)
        genre_tensor = torch.tensor(genres, dtype=torch.int32)
        country_tensor = torch.tensor(countries, dtype=torch.int32)

        batch = {
            'id' : id_tensor, 
            'album_input' : album_tensor, 
            'genre_input' : genre_tensor,
            'country_input' : country_tensor,
        }

        if len(labels) > 0 :
            batch['labels'] = labels
        return batch