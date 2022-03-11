import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA, _pca
from sklearn.manifold import TSNE
from tensorflow import keras

from consts import TRAIN_DIR

np.random.seed(42)

batch_size = 8

train_ds = keras.image_dataset_from_directory(TRAIN_DIR,
                                              image_size=(200, 200),
                                              batch_size=batch_size,
                                              seed=42)

train_ds = train_ds.unbatch()

inputs = list(train_ds.map(lambda x, y: x))

inputs = np.array(inputs)

labels = list(train_ds.map(lambda x, y: y))
labels = np.array(labels)

feat_cols = ['pixel'+str(i) for i in range(inputs.shape[1])]

df = pd.DataFrame(inputs, columns=feat_cols)
df['y'] = labels
df['label'] = df['y'].apply(lambda i: str(i))

inputs, labels = None, None

random_perm = np.random.permutation(df.shape[0])

fig = plt.figure(figsize=(15, 15))
for i in range(0, 15):
    ax = fig.add_subplot(3, 5, i+1, title='Digit: {}'.format(str(df.loc[random_perm[i], 'label'])))

ax.matshow(df.loc[random_perm[i], feat_cols].values.reshape((28, 28)).astype(float))
plt.show()

pca = PCA(n_components=3)

pca_result = pca.fit_transform(df[feat_cols].values)

df['pca-one'] = pca_result[:, 0]
df['pca-two'] = pca_result[:, 1]
df['pca-three'] = pca_result[:, 2]

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
ax.scatter(
    xs=df.loc[random_perm, :]["pca-one"],
    ys=df.loc[random_perm, :]["pca-two"],
    zs=df.loc[random_perm, :]["pca-three"],
    c=df.loc[random_perm, :]["y"],
    cmap='tab10'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.show()


N = 10000

df_subset = df.loc[random_perm[:N], :].copy()

data_subset = df_subset[feat_cols].values

pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_subset)

df['pca-one'] = pca_result[:, 0]
df['pca-two'] = pca_result[:, 1]
df['pca-three'] = pca_result[:, 2]

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

time_start = time.time()

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

tsne_results = tsne.fit_transform(data_subset)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

