
import torch
from tqdm import tqdm


class DataCollatorWithMasking :
    def __init__(
        self, 
        profile_data,
        special_token_dict,
        max_length,
        mlm=True,
        mlm_probability=0.15,
        label_pad_token_id=-100,
    ) : 
        self.build(profile_data)
        self.special_token_dict = special_token_dict
        self.max_length = max_length
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.label_pad_token_id = label_pad_token_id

    
    def build(self, profile_data) :

        profile_gender, profile_age, profile_pr_interest, profile_ch_interest = {}, {}, {}, {}
        
        for i in tqdm(range(len(profile_data))) :
            p_id, gender, age, pr_interest, ch_interest = profile_data.iloc[i][
                ['profile_id', 'sex', 'age', 'pr_interest_keyword_cd_1', 'ch_interest_keyword_cd_1']
            ]
            profile_gender[p_id] = gender
            profile_age[p_id] = age
            profile_pr_interest[p_id] = pr_interest
            profile_ch_interest[p_id] = ch_interest

        gender_dict = {s:i for i,s in enumerate(sorted(profile_data['sex'].unique()))}
        age_dict = {s:i for i,s in enumerate(sorted(profile_data['age'].unique()))}
        pr_interest_dict = {s:i for i,s in enumerate(sorted(profile_data['pr_interest_keyword_cd_1'].unique()))}
        ch_interest_dict = {s:i for i,s in enumerate(sorted(profile_data['ch_interest_keyword_cd_1'].unique()))}

        self.profile_gender = profile_gender
        self.profile_age = profile_age
        self.profile_pr_interest = profile_pr_interest
        self.profile_ch_interest = profile_ch_interest
        
        self.gender_dict = gender_dict
        self.age_dict = age_dict
        self.pr_interest_dict = pr_interest_dict
        self.ch_interest_dict = ch_interest_dict



    def __call__(self, dataset) :
        albums, genres, countries = [], [], []
        genders, ages, pr_interests, ch_interests = [], [], [], []

        max_length = min(self.max_length, max([len(d['album']) for d in dataset]))

        for data in dataset :
            d_id = data['id']
            genders.append(self.gender_dict[self.profile_gender[d_id]])
            ages.append(self.age_dict[self.profile_age[d_id]])
            pr_interests.append(self.pr_interest_dict[self.profile_pr_interest[d_id]])
            ch_interests.append(self.ch_interest_dict[self.profile_ch_interest[d_id]])

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
        
        genders = torch.tensor(genders, dtype=torch.int32)
        ages = torch.tensor(ages, dtype=torch.int32)
        pr_interests = torch.tensor(pr_interests, dtype=torch.int32)
        ch_interests = torch.tensor(ch_interests, dtype=torch.int32)

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
            'gender' : genders,
            'age' : ages,
            'pr_interest' : pr_interests,
            'ch_interest' : ch_interests
        }
        return batch


    def torch_mask(self, 
        album_tensor,
        genre_tensor,
        country_tensor,
        speical_token_dict, 
    ) :

        label_tensor = album_tensor.clone()
        pad_token_id = speical_token_dict['album_pad_token_id']
        
        probability_matrix = torch.full(label_tensor.shape, self.mlm_probability)
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
        profile_data,
        special_token_dict,
        max_length,
    ) : 
        self.build(profile_data)
        self.special_token_dict = special_token_dict
        self.max_length = max_length


    def build(self, profile_data) :

        profile_gender, profile_age, profile_pr_interest, profile_ch_interest = {}, {}, {}, {}
        
        for i in tqdm(range(len(profile_data))) :
            p_id, gender, age, pr_interest, ch_interest = profile_data.iloc[i][
                ['profile_id', 'sex', 'age', 'pr_interest_keyword_cd_1', 'ch_interest_keyword_cd_1']
            ]
            profile_gender[p_id] = gender
            profile_age[p_id] = age
            profile_pr_interest[p_id] = pr_interest
            profile_ch_interest[p_id] = ch_interest

        gender_dict = {s:i for i,s in enumerate(sorted(profile_data['sex'].unique()))}
        age_dict = {s:i for i,s in enumerate(sorted(profile_data['age'].unique()))}
        pr_interest_dict = {s:i for i,s in enumerate(sorted(profile_data['pr_interest_keyword_cd_1'].unique()))}
        ch_interest_dict = {s:i for i,s in enumerate(sorted(profile_data['ch_interest_keyword_cd_1'].unique()))}

        self.profile_gender = profile_gender
        self.profile_age = profile_age
        self.profile_pr_interest = profile_pr_interest
        self.profile_ch_interest = profile_ch_interest
        
        self.gender_dict = gender_dict
        self.age_dict = age_dict
        self.pr_interest_dict = pr_interest_dict
        self.ch_interest_dict = ch_interest_dict



    def __call__(self, dataset) :
        ids, albums, genres, countries = [], [], [], []
        genders, ages, pr_interests, ch_interests = [], [], [], []

        labels = []
        max_length = min(self.max_length, max([len(d['album']) for d in dataset]))

        for data in dataset :
            d_id = data['id']
            ids.append(d_id)
            genders.append(self.gender_dict[self.profile_gender[d_id]])
            ages.append(self.age_dict[self.profile_age[d_id]])
            pr_interests.append(self.pr_interest_dict[self.profile_pr_interest[d_id]])
            ch_interests.append(self.ch_interest_dict[self.profile_ch_interest[d_id]])

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

        genders = torch.tensor(genders, dtype=torch.int32)
        ages = torch.tensor(ages, dtype=torch.int32)
        pr_interests = torch.tensor(pr_interests, dtype=torch.int32)
        ch_interests = torch.tensor(ch_interests, dtype=torch.int32)

        batch = {
            'id' : id_tensor, 
            'album_input' : album_tensor, 
            'genre_input' : genre_tensor,
            'country_input' : country_tensor,
            'gender' : genders,
            'age' : ages,
            'pr_interest' : pr_interests,
            'ch_interest' : ch_interests
        }

        if len(labels) > 0 :
            batch['labels'] = labels
        return batch

