import pandas as pd
import re
import time
import matplotlib.pyplot as plt
import numpy as np

def parse_lines(text):
    # Define a regular expression pattern to match lines in a numbered or dashed line format
    pattern_numbered = r'\d+\.\s*(.+?)\s*(?=\d+\.|$)'
    pattern_dashed = r'-\s*(.+?)\s*(?=-|$)'
    
    numbered_matches = re.findall(pattern_numbered, text, re.DOTALL)
    dashed_matches = re.findall(pattern_dashed, text, re.DOTALL)
    
    # Return the lists of parsed strings
    if numbered_matches:
        return numbered_matches
    else:
        return dashed_matches

def process_set(strings):
    result = []
    for s in strings:
        match = re.search(r'[^a-zA-Z\s]', s)
        if match:
            elem = s[:match.start()]
        else:
            elem = s

        if "\n" in elem:
            elem = elem.split("\n")[0]
        elem = elem.strip()

        if len(elem.split(" ")) < 4:
            result.append(elem)
    return sorted(list(set(result)))

def generate_freq_df(strings, answers):
    df = pd.DataFrame(0, index=answers, columns=[s.title() for s in strings])
    
    # Iterate over each answer
    for answer in answers:
        for string in strings:
            if string in answer:
                df.at[answer, string.title()] = 1
    df = df.drop(df.columns[:1], axis=1)

    return df.reset_index(drop=True)

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

df = pd.read_csv('paragraph_gpt_emotions.csv')
df = df.dropna(subset=['gpt_class'])

parsed_arr = []
for answer in df['gpt_class']:
    print(answer)
    parsed = parse_lines(answer)
    print(parsed)
    parsed_arr.append(parsed)

emotions = set([item for sublist in parsed_arr for item in sublist])
emotion_list = process_set(emotions)

df_emotions = generate_freq_df(emotion_list, df['gpt_class'])
counts_dict = df_emotions.sum().to_dict()
print(df_emotions)
time.sleep(10)
#plot_cdf(dict(sorted(counts_dict.items(), key=lambda item: item[1], reverse=True)), "Emotions")

# Initialize variables
coverage_threshold = 0.99
selected_columns = []
remaining_rows = set(df_emotions.index)

while True:
    best_column = None
    best_covered_rows = set()

    for column in df_emotions.columns:
        if column not in selected_columns:
            print(column)
            covered_rows = {i for i in remaining_rows if df_emotions.loc[i, selected_columns + [column]].sum() >= 1}
            if len(covered_rows) > len(best_covered_rows):
                best_column = column
                best_covered_rows = covered_rows

    if best_column is None:
        break

    selected_columns.append(best_column)
    remaining_rows -= best_covered_rows

    # Check if 99% of rows are covered
    covered_percentage = (len(df_emotions) - len(remaining_rows)) / len(df_emotions)
    if covered_percentage >= coverage_threshold:
        break

# Print selected columns
print("Selected Columns:", selected_columns)

filtered_df = df_emotions[selected_columns]
print(filtered_df)

'''
df = pd.concat([df, df_emotions.reset_index(drop=True)], axis=1)
print(df.columns)
df.to_csv('paragraph_gpt_emotions_classified.csv')
'''