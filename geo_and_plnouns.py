import pandas as pd
import re
import requests
import time
from collections import Counter
import ast 
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

API_KEY = 'pk.eyJ1IjoibXZkdmxkbiIsImEiOiJjbHphNzR0MDEwMXR5MnBweHN3M3IxbnY2In0.QWEatCWyi0fec2lWw5CRMw'

def get_coordinates(place_name, api_key):
    url = f'https://api.mapbox.com/geocoding/v5/mapbox.places/{place_name},+Lake+District,+England.json'
    params = {
        'access_token': api_key,
        'limit': 1
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError:
            print("Error decoding JSON response")
            return None
        
        if data['features']:
            location = data['features'][0]['center']
            return {
                'latitude': location[1],
                'longitude': location[0]
            }
        else:
            print("No data found for the given place.")
            return None
    else:
        print(f"Request failed with status code {response.status_code}")
        return None

def extract_nouns(s):
    try:
        arr = re.findall(r'[a-zA-Z\s]+', s)
        arr = [phrase.strip() for phrase in arr if phrase.strip()]
        return arr
    except:
        return []

def extract_punct(s):
    punctuation_chars = r"[.,?!:;'\-()\"]"
    try:
        arr = re.findall(punctuation_chars, s)
        return arr
    except:
        return []

def count_values(arr, possible_values):
    # Initialize a count dictionary for each possible value
    print(possible_values)
    counts = {val: 0 for val in possible_values}
    for sublist in arr:
        for item in sublist:
            if item in counts:
                counts[item] += 1
    count = [counts[val] for val in possible_values]
    
    '''
    if np.std(count) == 0:
        norm_count = [0] * len(count)
    else:
        norm_count = (count - np.mean(count)) / np.std(count)
    '''
    return count

def get_possible(arrs, cutoff=None):
    pos_dict = dict(Counter(item for sublist in arrs for item in sublist))
    keys = sorted(pos_dict.items(), key=lambda item: item[1], reverse=True)
    if not cutoff:
        cutoff = len(keys)
    return dict(sorted(pos_dict.items(), key=lambda item: item[1], reverse=True)), [k for k, v in keys[:cutoff]]

def plot_cdf(freq_dict, title):
    values = np.array(list(freq_dict.values()))

    cumulative_sum = np.cumsum(values)
    total_sum = cumulative_sum[-1]
    cumulative_normalized = cumulative_sum / total_sum

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(values) + 1), cumulative_normalized, marker='o', linestyle='-', color='b')

    # Set ticks for each key
    plt.xticks(range(round(len(values) / 200) * 10, len(values) + 1, round(len(values) / 200) * 10))

    # Set axis labels and title
    plt.xlabel('Nouns (Sorted by Frequency)')
    plt.ylabel('Cumulative Distribution')
    plt.title(f'Cumulative Distribution Plot of Frequency Distribution for {title}')

    # Add grid for clarity
    plt.grid(True)

    # Show the plot
    plt.show()

df = pd.read_csv('final.csv')
df['geonouns'] = df['geonouns'].apply(extract_nouns)
df['plnames'] = df['plnames'].apply(extract_nouns)
df['punctuation'] = df['sentence'].apply(extract_punct)
df['gpt_class'] = df['gpt_class'].apply(lambda x: [x])
df['distl_class'] = df['distl_class'].apply(lambda x: ast.literal_eval(x))

distl_dict, distl_pos = get_possible(df['distl_class'])
gpt_dict, gpt_pos = get_possible(df['gpt_class'])
geo_dict, geo_pos = get_possible(df['geonouns'], 50)
pl_dict, pl_pos = get_possible(df['plnames'], 200)
punct_dict, punct_pos = get_possible(df['punctuation'])

plot_cdf(geo_dict, 'Geo-Nouns')
plot_cdf(pl_dict, 'Place Names')

result = df.groupby(['author', 'date', 'X1', 'X2']).agg(
    distl_count=('distl_class', lambda x: count_values(x, distl_pos)),
    gpt_count=('gpt_class', lambda x: count_values(x, gpt_pos)),
    geo_count=('geonouns', lambda x: count_values(x, geo_pos)),
    pl_count=('plnames', lambda x: count_values(x, pl_pos)),
    punct_count=('punctuation', lambda x: count_values(x, punct_pos))
).reset_index()
print(result)

distl_df = pd.DataFrame(result['distl_count'].tolist(), index=result.index)
distl_df.columns = [f'distl_{list(distl_dict.keys())[i]}' for i in range(distl_df.shape[1])]

gpt_df = pd.DataFrame(result['gpt_count'].tolist(), index=result.index)
gpt_df.columns = [f'gpt_{list(gpt_dict.keys())[i]}' for i in range(gpt_df.shape[1])]

geo_df = pd.DataFrame(result['geo_count'].tolist(), index=result.index)
geo_df.columns = [f'geo_{list(geo_dict.keys())[i]}' for i in range(geo_df.shape[1])]

pl_df = pd.DataFrame(result['pl_count'].tolist(), index=result.index)
pl_df.columns = [f'pl_{list(pl_dict.keys())[i]}' for i in range(pl_df.shape[1])]

punct_df = pd.DataFrame(result['punct_count'].tolist(), index=result.index)
punct_df.columns = [f'punct_{list(punct_dict.keys())[i]}' for i in range(punct_df.shape[1])]
print(distl_df)

df = pd.concat([result.drop(columns=['distl_count', 'gpt_count', 'geo_count', 'pl_count', 'punct_count']), 
                         distl_df, gpt_df, geo_df, pl_df, punct_df], axis=1)
df_meta = pd.read_csv('../data/lake_district/LD_metadata.csv')
df.insert(4, 'word_count', df_meta['Word_count'])
df.insert(4, 'dec_written', df_meta['Decade_comp'])
print(df)

df.to_csv('raw_counts_50_200.csv')

'''
dfs = []
for key in geo_dict:
    mask = df['geonouns'].apply(lambda x: key in x)

    filtered_df = df[mask]
    filtered_df['geonouns_tuple'] = filtered_df['geonouns'].apply(tuple)

    filtered_df['distl_class'] = filtered_df['distl_class'].apply(lambda x: ast.literal_eval(x))
    distl_class_expanded = filtered_df['distl_class'].explode()

    classes = {1, 2, 3, 4, 5}
    gpt_count = [filtered_df['gpt_class'].eq(value).sum() for value in classes]
    dst_count = [distl_class_expanded.eq(value).sum() for value in classes]

    print(gpt_count)
    print(dst_count)

    result_df = pd.DataFrame({
        'geonoun': f'{key} ({geo_dict[key]})',
        'count': [geo_dict[key]] * 5,
        'class': [1, 2, 3, 4, 5],
        'distl_class_count': dst_count,
        'gpt_class_count': gpt_count,
        'distl_class_prop': [round(x, 3) for x in dst_count / sum(dst_count)],
        'gpt_class_prop': [round(x, 3) for x in gpt_count / sum(gpt_count)],
    })
    result_df = result_df.rename_axis('class')
    dfs.append(result_df)

df = pd.concat(dfs, ignore_index=True)
print(df)
df.to_csv('geonoun_freq.csv')

geo_loc = {}
for key in pl_dict:
    print(f'{key}: {get_coordinates(key, API_KEY)}')
    geo_loc[key] = get_coordinates(key, API_KEY)
    time.sleep(0.5)
'''