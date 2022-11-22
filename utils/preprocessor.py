
import pandas as pd
from tqdm import tqdm
from datasets import Dataset


def parse(df, keywords) :
    profile_ids = list(df['profile_id'].unique())

    d_ids, d_albums, d_keywords, d_genres, d_countries = [], [], [], [], []
    for p in profile_ids :
        sub_df = df[df['profile_id'] == p]

        album_sequence = list(sub_df['album_id'])
        keyword_sequence = [keywords[a] for a in album_sequence]

        d_ids.append(p)

        d_albums.append(album_sequence)
        d_keywords.append(keyword_sequence)
        d_genres.append(list(sub_df['genre']))
        d_countries.append(list(sub_df['country']))

    parsed_df = pd.DataFrame(
        {
            'id' : d_ids, 
            'genre' : d_genres,
            'album' : d_albums,
            'keyword' : d_keywords,
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
        albums, genres, countries, keywords = [], [], [], []
        org_albums, org_genres, org_countries = dataset['album'], dataset['genre'], dataset['country']
        org_keywords = dataset['keyword']

        for i in range(len(org_albums)) :
            album, genre, country = org_albums[i], org_genres[i], org_countries[i]
            keyword = org_keywords[i]

            if len(album) == 1 :
                albums.append(album)
                genres.append(genre)
                countries.append(country)
                keywords.append(keyword)
                labels.append([])
            else :
                remain_size = int(len(album) * (1-self.leave_probability))

                label = album[remain_size:]
                album, genre, country = album[:remain_size], genre[:remain_size], country[:remain_size]
                keyword = keyword[:remain_size]

                albums.append(album)
                genres.append(genre)
                countries.append(country)
                keywords.append(keyword)
                labels.append(label)

        dataset['album'] = albums
        dataset['genre'] = genres
        dataset['country'] = countries
        dataset['keyword'] = keywords
        dataset['labels'] = labels

        return dataset

