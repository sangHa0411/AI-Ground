
import pandas as pd
from tqdm import tqdm
from datasets import Dataset

def preprocess(history_data, meta_data) :

    df = history_data[['profile_id', 'log_time', 'album_id']].drop_duplicates(
        subset=['profile_id', 'album_id', 'log_time']).sort_values(
            by = ['profile_id', 'log_time']
        ).reset_index(drop = True)

    meta_countries = []
    for c in list(meta_data['country']) :
        if isinstance(c, float) :
            meta_countries.append('Unknown')
        else :
            meta_countries.append(c)
            
    meta_data['country'] = meta_countries
    
    meta_genre = meta_data['genre_mid'].to_dict()
    meta_country = meta_data['country'].to_dict()
    genre_dict = {s:i for i,s in enumerate(sorted(meta_data['genre_mid'].unique()))}
    country_dict = {s:i for i,s in enumerate(sorted(meta_data['country'].unique()))}

    genres, countries = [], []

    for i in tqdm(range(len(df))) :
        album = df.iloc[i]['album_id']
        genres.append(genre_dict[meta_genre[album]])
        countries.append(country_dict[meta_country[album]])

    df['genre'] = genres
    df['country'] = countries

    return df


def parse(df) :
    profile_ids = list(df['profile_id'].unique())

    d_ids, d_albums, d_genres, d_countries = [], [], [], []
    for p in profile_ids :
        sub_df = df[df['profile_id'] == p]

        d_ids.append(p)
        d_albums.append(list(sub_df['album_id']))
        d_genres.append(list(sub_df['genre']))
        d_countries.append(list(sub_df['country']))

    parsed_df = pd.DataFrame(
        {
            'id' : d_ids, 
            'genre' : d_genres,
            'album' : d_albums,
            'country' : d_countries
        }
    )
    parsed_dataset = Dataset.from_pandas(parsed_df)
    return parsed_dataset


class Spliter :

    def __init__(self, leave_probability) :
        self.leave_probability = leave_probability

    def __call__(self, dataset) :

        labels = []
        albums, genres, countries = [], [], []
        org_albums, org_genres, org_countries = dataset['album'], dataset['genre'], dataset['country']

        for i in range(len(org_albums)) :
            album, genre, country = org_albums[i], org_genres[i], org_countries[i]
        
            if len(album) == 1 :
                albums.append(album)
                genres.append(genre)
                countries.append(country)
                labels.append([])
            else :
                remain_size = int(len(album) * (1-self.leave_probability))

                label = album[remain_size:]
                album, genre, country = album[:remain_size], genre[:remain_size], country[:remain_size]

                albums.append(album)
                genres.append(genre)
                countries.append(country)
                labels.append(label)

        dataset['album'] = albums
        dataset['genre'] = genres
        dataset['country'] = countries
        dataset['labels'] = labels

        return dataset

