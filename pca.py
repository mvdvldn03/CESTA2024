import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np

df = pd.read_csv('counts_50_200.csv')

pca_cols = df.columns[df.columns.get_loc('punct_0'):df.columns.get_loc('punct_10') + 1]

punct_matrix = np.eye(11)
norms = np.linalg.norm(punct_matrix, axis=1, keepdims=True)
norm_matrix = punct_matrix / norms
identity_df = pd.DataFrame(norm_matrix, columns=[f'Feature_{i}' for i in range(11)])

df_T = df[pca_cols].T
pca = PCA(n_components=2) 
pca_result = pca.fit_transform(df[pca_cols])
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
pca_df['author'] = df['author']
pca_df['date'] = df['dec_written']
pca_df['words'] = df['word_count']
print(pca_df)

scaler_1 = MinMaxScaler()
scaler_2 = MinMaxScaler()
pca_df['date_normalized'] = scaler_1.fit_transform(pca_df[['date']])
pca_df['words_normalized'] = scaler_2.fit_transform(pca_df[['words']])

#Date Scaling
plt.figure(figsize=(10, 6))
sc = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['date_normalized'], cmap='viridis', s=50)
cbar = plt.colorbar(sc, label='Date Written')
cbar.set_label('Date Written')

# Set the color bar ticks to reflect the original date values
date_min, date_max = df['dec_written'].min(), df['dec_written'].max()
date_ticks = scaler_1.transform(np.array([date_min, date_max]).reshape(-1, 1)).flatten()
cbar.set_ticks(date_ticks)
cbar.set_ticklabels([date_min, date_max])

# Annotate the points with the 'author' labels
for i, txt in enumerate(pca_df['author']):
    plt.annotate(txt, (pca_df['PC1'][i], pca_df['PC2'][i]), fontsize=6)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Emotion Tagging (GPT, xtremedistl), Geo-Noun (Top 50), Place Name (Top 200), and Punctuation Frequency for Lake District Authors')
plt.grid(True)
plt.show()  

#Word Count Normalized
plt.figure(figsize=(10, 6))
sc = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['words_normalized'], cmap='viridis', s=50)
cbar = plt.colorbar(sc, label='Word Count')
cbar.set_label('Word Count')

word_min, word_max = df['word_count'].min(), df['word_count'].max()
word_ticks = scaler_2.transform(np.array([word_min, word_max]).reshape(-1, 1)).flatten()
cbar.set_ticks(word_ticks)
cbar.set_ticklabels([word_min, word_max])

for i, txt in enumerate(pca_df['author']):
    plt.annotate(txt, (pca_df['PC1'][i], pca_df['PC2'][i]), fontsize=6)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Emotion Tagging (GPT, xtremedistl), Geo-Noun (Top 50), Place Name (Top 200), and Punctuation Frequency for Lake District Authors')
plt.grid(True)
plt.show()  