
import pandas as pd
from tqdm import tqdm
from datasets import Dataset


def parse(df, keywords) :
    profile_ids = list(df['profile_id'].unique())

    d_ids, d_albums, d_logs, d_keywords, d_genres, d_countries = [], [], [], [], [], []
    for p in profile_ids :
        sub_df = df[df['profile_id'] == p]

        album_sequence = list(sub_df['album_id'])
        album_logs = list(sub_df['log_time'])
        keyword_sequence = [keywords[a] for a in album_sequence]

        d_ids.append(p)

        d_albums.append(album_sequence)
        d_logs.append(album_logs)
        d_keywords.append(keyword_sequence)
        d_genres.append(list(sub_df['genre']))
        d_countries.append(list(sub_df['country']))

    parsed_df = pd.DataFrame(
        {
            'id' : d_ids, 
            'genre' : d_genres,
            'log_time' : d_logs,
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
        org_logs = dataset['log_time']
        org_keywords = dataset['keyword']

        for i in range(len(org_albums)) :
            logs = org_logs[i]
            album, genre, country = org_albums[i], org_genres[i], org_countries[i]
            keyword = org_keywords[i]

            row_size = len(album)

            sub_labels = []
            sub_albums, sub_genres, sub_countries, sub_keywords = [], [], [], []

            for j in range(row_size) :
                log = str(logs[j])
                if log[:6] == '202203' :
                    sub_albums.append(album[j])
                    sub_genres.append(genre[j])
                    sub_countries.append(country[j])
                    sub_keywords.append(keyword[j])
                else :
                    sub_labels.append(album[j])

            albums.append(sub_albums)
            genres.append(sub_genres)
            countries.append(sub_countries)
            keywords.append(sub_keywords)
            labels.append(sub_labels)

        dataset['album'] = albums
        dataset['genre'] = genres
        dataset['country'] = countries
        dataset['keyword'] = keywords
        dataset['labels'] = labels

        return dataset

