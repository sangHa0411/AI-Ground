
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
        eval_flag=True,
    ) : 
        self.build(profile_data)
        self.special_token_dict = special_token_dict
        self.max_length = max_length
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.label_pad_token_id = label_pad_token_id
        self.eval_flag = eval_flag


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

        if self.eval_flag :
            max_length = min(self.max_length, max([len(d['album'])-1 for d in dataset]))    
        else :
            max_length = min(self.max_length, max([len(d['album']) for d in dataset]))

        for data in dataset :
            d_id = data['id']
            genders.append(self.gender_dict[self.profile_gender[d_id]])
            ages.append(self.age_dict[self.profile_age[d_id]])
            pr_interests.append(self.pr_interest_dict[self.profile_pr_interest[d_id]])
            ch_interests.append(self.ch_interest_dict[self.profile_ch_interest[d_id]])

            album = data['album'][:-1] if self.eval_flag else data['album']
            if len(album) < max_length :
                pad_length = max_length - len(album)
                album = album + [self.special_token_dict['album_pad_token_id']] * pad_length
            else :
                album = album[-max_length:]
            albums.append(album)

            genre = data['genre'][:-1] if self.eval_flag else data['genre']
            if len(genre) < max_length :
                pad_length = max_length - len(genre)
                genre = genre + [self.special_token_dict['genre_pad_token_id']] * pad_length
            else :
                genre = genre[-max_length:]
            genres.append(genre)

            country = data['country'][:-1] if self.eval_flag else data['country']
            if len(country) < max_length :
                pad_length = max_length - len(country)
                country = country + [self.special_token_dict['country_pad_token_id']] * pad_length
            else :
                country = country[-max_length:]
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
        eval_flag,
    ) : 
        self.build(profile_data)
        self.special_token_dict = special_token_dict
        self.max_length = max_length
        self.eval_flag=eval_flag


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

        if self.eval_flag :
            
            labels = []
            for data in dataset :
                d_id = data['id']
                genders.append(self.gender_dict[self.profile_gender[d_id]])
                ages.append(self.age_dict[self.profile_age[d_id]])
                pr_interests.append(self.pr_interest_dict[self.profile_pr_interest[d_id]])
                ch_interests.append(self.ch_interest_dict[self.profile_ch_interest[d_id]])

                album, label = data['album'][:-1] + [self.special_token_dict['album_mask_token_id']], data['album'][-1]
                if len(album) < max_length :
                    pad_length = max_length - len(album)
                    album = album + [self.special_token_dict['album_pad_token_id']] * pad_length
                else :
                    album = album[-max_length:]
                albums.append(album)
                labels.append(label)

                genre = data['genre'][:-1] + [self.special_token_dict['genre_mask_token_id']]
                if len(genre) < max_length :
                    pad_length = max_length - len(genre)
                    genre = genre + [self.special_token_dict['genre_pad_token_id']] * pad_length
                else :
                    genre = genre[-max_length:]
                genres.append(genre)

                country = data['country'][:-1] + [self.special_token_dict['country_mask_token_id']]
                if len(country) < max_length :
                    pad_length = max_length - len(country)
                    country = country + [self.special_token_dict['country_pad_token_id']] * pad_length
                else :
                    country = country[-max_length:]
                countries.append(country)


            album_tensor = torch.tensor(albums, dtype=torch.int32)
            genre_tensor = torch.tensor(genres, dtype=torch.int32)
            country_tensor = torch.tensor(countries, dtype=torch.int32)
            label_tensor = torch.tensor(labels, dtype=torch.int32)

            genders = torch.tensor(genders, dtype=torch.int32)
            ages = torch.tensor(ages, dtype=torch.int32)
            pr_interests = torch.tensor(pr_interests, dtype=torch.int32)
            ch_interests = torch.tensor(ch_interests, dtype=torch.int32)

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

        else :
                
            for data in dataset :
                d_id = data['id']
                genders.append(self.gender_dict[self.profile_gender[d_id]])
                ages.append(self.age_dict[self.profile_age[d_id]])
                pr_interests.append(self.pr_interest_dict[self.profile_pr_interest[d_id]])
                ch_interests.append(self.ch_interest_dict[self.profile_ch_interest[d_id]])
                
                album = data['album']
                if len(album) < max_length :
                    pad_length = max_length - len(album)
                    album = album + [self.special_token_dict['album_pad_token_id']] * pad_length
                else :
                    album = album[-max_length:]
                albums.append(album)

                genre = data['genre']
                if len(genre) < max_length :
                    pad_length = max_length - len(genre)
                    genre = genre + [self.special_token_dict['genre_pad_token_id']] * pad_length
                else :
                    genre = genre[-max_length:]
                genres.append(genre)

                country = data['country']
                if len(country) < max_length :
                    pad_length = max_length - len(country)
                    country = country + [self.special_token_dict['country_pad_token_id']] * pad_length
                else :
                    country = country[-max_length:]
                countries.append(country)

            album_tensor = torch.tensor(albums, dtype=torch.int32)
            genre_tensor = torch.tensor(genres, dtype=torch.int32)
            country_tensor = torch.tensor(countries, dtype=torch.int32)

            genders = torch.tensor(genders, dtype=torch.int32)
            ages = torch.tensor(ages, dtype=torch.int32)
            pr_interests = torch.tensor(pr_interests, dtype=torch.int32)
            ch_interests = torch.tensor(ch_interests, dtype=torch.int32)

            batch = {
                'album_input' : album_tensor, 
                'genre_input' : genre_tensor,
                'country_input' : country_tensor,
                'gender' : genders,
                'age' : ages,
                'pr_interest' : pr_interests,
                'ch_interest' : ch_interests
            }
            return batch
