import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', None)
seed_df = pd.read_csv("seeds_dataset.csv", names=['area', 'perimeter', 'compactness', 'kernel_length', 'kernel_width',
                                             'asymm_coeff', 'kernel_groove_length', 'class'])

print(seed_df.head())
# sns.pairplot(df, hue='class', height=3, aspect=1)
features = ['area', 'perimeter', 'compactness', 'kernel_length', 'kernel_width', 'asymm_coeff',
            'kernel_groove_length']
# Get only the feature columns
x = seed_df.loc[:, features].values
print(np.mean(x))
# Get only the class column
y = seed_df.loc[:, ['class']].values
# Apply StandardScaler to the features
x = StandardScaler().fit_transform(x)
print(np.mean(x), np.std(x))

# Run PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principal_comp_df = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
seed_principal_df = pd.concat([principal_comp_df, seed_df[['class']]], axis=1)

print(seed_principal_df.head())
# colors = ['r', 'g', 'b']  # TODO Change colors!!
# plt.scatter(finalDf['principal component 1'], finalDf['principal component 2'], s=50, c=colors)

fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('PC1', fontsize=12)
ax.set_ylabel('PC2', fontsize=12)
ax.set_title('Seeds Dataset PCA (2 components)', fontsize=18)
classes = [1, 2, 3]
colors = ['c', 'm', 'y']
for s_class, color in zip(classes, colors):
    indicesToKeep = seed_principal_df['class'] == s_class
    ax.scatter(seed_principal_df.loc[indicesToKeep, 'PC1']
               , seed_principal_df.loc[indicesToKeep, 'PC2']
               , c=color
               , s=50)
ax.legend(classes)
ax.grid()
plt.show()
