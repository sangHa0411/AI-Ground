
import torch
import random
import numpy as np


class DataCollatorWithMasking :
    def __init__(
        self, 
        profile_data,
        special_token_dict,
        max_length,
        keyword_max_length,
        mlm=True,
        mlm_probability=0.15,
        label_pad_token_id=-100,
    ) : 
        self.build(profile_data)
        self.special_token_dict = special_token_dict
        self.max_length = max_length
        self.keyword_max_length = keyword_max_length
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.label_pad_token_id = label_pad_token_id


    def build(self, profile_data) :

        profile_gender, profile_age = {}, {}
        
        for i in range(len(profile_data)) :
            p_id, gender, age = profile_data.iloc[i][['profile_id', 'sex', 'age']]
            profile_gender[p_id] = gender
            profile_age[p_id] = age

        gender_dict = {s:i for i,s in enumerate(sorted(profile_data['sex'].unique()))}
        age_dict = {s:i for i,s in enumerate(sorted(profile_data['age'].unique()))}

        self.profile_gender = profile_gender
        self.profile_age = profile_age
        
        self.gender_dict = gender_dict
        self.age_dict = age_dict


    def __call__(self, dataset) :
        albums, genres, countries, keywords = [], [], [], []
        genders, ages = [], []

        max_length = min(self.max_length, max([len(d['album']) for d in dataset]))

        for data in dataset :
            d_id = data['id']
            genders.append(self.gender_dict[self.profile_gender[d_id]])
            ages.append(self.age_dict[self.profile_age[d_id]])

            album, genre, country, keyword = data['album'], data['genre'], data['country'], data['keyword']

            if len(album) < max_length :
                pad_length = max_length - len(album)
                album = [self.special_token_dict['album_pad_token_id']] * pad_length + album
                genre = [self.special_token_dict['genre_pad_token_id']] * pad_length + genre
                country = [self.special_token_dict['country_pad_token_id']] * pad_length + country
                keyword = [
                    [self.special_token_dict['keyword_pad_token_id']] * self.keyword_max_length
                ] * pad_length + keyword 
            else :
                album = album[-max_length:]
                genre = genre[-max_length:]
                country = country[-max_length:]
                keyword = keyword[-max_length:]


            albums.append(album)
            genres.append(genre)
            countries.append(country)
            keywords.append(keyword)

        album_tensor = torch.tensor(albums, dtype=torch.int32)
        genre_tensor = torch.tensor(genres, dtype=torch.int32)
        country_tensor = torch.tensor(countries, dtype=torch.int32)
        keyword_tensor = torch.tensor(keywords, dtype=torch.int32)
        
        genders = torch.tensor(genders, dtype=torch.int32)
        ages = torch.tensor(ages, dtype=torch.int32)

        album_tensor, genre_tensor, country_tensor, keyword_tensor, label_tensor = self.torch_mask(
            album_tensor,
            genre_tensor,
            country_tensor,
            keyword_tensor,
            self.special_token_dict
        )

    
        batch = {
            'album_input' : album_tensor, 
            'labels' : label_tensor, 
            'genre_input' : genre_tensor,
            'country_input' : country_tensor,
            'keyword_input' : keyword_tensor,
            'gender' : genders,
            'age' : ages
        }
        return batch


    def torch_mask(self, 
        album_tensor,
        genre_tensor,
        country_tensor,
        keyword_tensor,
        speical_token_dict, 
    ) :

        batch_size, seq_size = album_tensor.shape
        
        last_mask_tensor = torch.zeros(seq_size, dtype=torch.double)
        last_mask_tensor[-1] = 1.0

        last_mask_indices = random.sample(range(batch_size), int(batch_size * 0.1))

        label_tensor = album_tensor.clone()
        pad_token_id = speical_token_dict['album_pad_token_id']
        
        probability_matrix = torch.full(label_tensor.shape, self.mlm_probability, dtype=torch.double)
        probability_matrix[last_mask_indices] = last_mask_tensor

        probability_matrix = torch.where(
            label_tensor == pad_token_id, 0.0, probability_matrix
        )

        masked_indices = torch.bernoulli(probability_matrix).bool()

        label_tensor[~masked_indices] = -100
        album_tensor[masked_indices] = speical_token_dict['album_mask_token_id']
        genre_tensor[masked_indices] = speical_token_dict['genre_mask_token_id']
        country_tensor[masked_indices] = speical_token_dict['country_mask_token_id']
        keyword_tensor[masked_indices] = speical_token_dict['keyword_mask_token_id']

        return album_tensor, genre_tensor, country_tensor, keyword_tensor, label_tensor 



class DataCollatorWithPadding :
    def __init__(
        self, 
        profile_data,
        special_token_dict,
        max_length,
        keyword_max_length,
    ) : 
        self.build(profile_data)
        self.special_token_dict = special_token_dict
        self.max_length = max_length
        self.keyword_max_length = keyword_max_length

    def build(self, profile_data) :

        profile_gender, profile_age = {}, {}
        
        for i in range(len(profile_data)) :
            p_id, gender, age = profile_data.iloc[i][['profile_id', 'sex', 'age', ]]
            profile_gender[p_id] = gender
            profile_age[p_id] = age

        gender_dict = {s:i for i,s in enumerate(sorted(profile_data['sex'].unique()))}
        age_dict = {s:i for i,s in enumerate(sorted(profile_data['age'].unique()))}

        self.profile_gender = profile_gender
        self.profile_age = profile_age
        
        self.gender_dict = gender_dict
        self.age_dict = age_dict


    def __call__(self, dataset) :
        ids, albums, genres, countries, keywords = [], [], [], [], []
        genders, ages = [], []

        labels = []
        max_length = min(self.max_length, max([len(d['album']) for d in dataset]))

        for data in dataset :
            d_id = data['id']
            ids.append(d_id)
            genders.append(self.gender_dict[self.profile_gender[d_id]])
            ages.append(self.age_dict[self.profile_age[d_id]])
            
            album = data['album'] + [self.special_token_dict['album_mask_token_id']]
            genre = data['genre'] + [self.special_token_dict['genre_mask_token_id']]
            country = data['country'] + [self.special_token_dict['country_mask_token_id']]
            keyword = data['keyword'] + [
                [self.special_token_dict['keyword_mask_token_id']] * self.keyword_max_length
            ]

            if len(album) < max_length :
                pad_length = max_length - len(album)
                album = [self.special_token_dict['album_pad_token_id']] * pad_length + album
                genre = [self.special_token_dict['genre_pad_token_id']] * pad_length + genre
                country = [self.special_token_dict['country_pad_token_id']] * pad_length + country
                keyword = [
                    [self.special_token_dict['keyword_pad_token_id']] * self.keyword_max_length
                ] * pad_length + keyword 
            else :
                album = album[-max_length:]
                genre = genre[-max_length:]
                country = country[-max_length:]
                keyword = keyword[-max_length:]

            if 'labels' in data :
                labels.append(data['labels'])

            albums.append(album)
            genres.append(genre)
            countries.append(country)
            keywords.append(keyword)

        id_tensor = torch.tensor(ids, dtype=torch.int32)
        album_tensor = torch.tensor(albums, dtype=torch.int32)
        genre_tensor = torch.tensor(genres, dtype=torch.int32)
        country_tensor = torch.tensor(countries, dtype=torch.int32)
        keyword_tensor = torch.tensor(keywords, dtype=torch.int32)

        genders = torch.tensor(genders, dtype=torch.int32)
        ages = torch.tensor(ages, dtype=torch.int32)

        batch = {
            'id' : id_tensor, 
            'album_input' : album_tensor, 
            'genre_input' : genre_tensor,
            'country_input' : country_tensor,
            'keyword_input' : keyword_tensor,
            'gender' : genders,
            'age' : ages,
        }

        if len(labels) > 0 :
            batch['labels'] = labels
        return batch