from tqdm import tqdm

def load_history(history_data, meta_data) :

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

def load_meta(meta_data, meta_data_plus, keyword_max_length=10) :

    album_ids = sorted(meta_data['album_id'].unique())
    keyword_size = len(meta_data_plus['keyword_name'].unique())
    keywords_map = {k:i for i, k in enumerate(meta_data_plus['keyword_name'].unique())}

    pad_keyword_value = keyword_size + 1

    album_keywords = {}

    for i in tqdm(album_ids) :
        sub_meta_plus_df = meta_data_plus[meta_data_plus['album_id'] == i]   
        sub_meta_plus_df = sub_meta_plus_df[[True if v >= 3 else False for v in sub_meta_plus_df['keyword_value']]]

        sub_keywords = list(sub_meta_plus_df['keyword_name'])
        sub_keywords = [keywords_map[k] for k in sub_keywords]

        if len(sub_keywords) >= keyword_max_length :
            sub_keywords = sub_keywords[:keyword_max_length]
        else :
            pad_length = keyword_max_length - len(sub_keywords)
            sub_keywords = [pad_keyword_value] * pad_length + sub_keywords

        album_keywords[i] = sub_keywords

    return album_keywords, keyword_size

