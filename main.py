import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans

with open('data.txt', 'r', encoding='utf-8') as file:
    data = file.read()


data = data.lower()
data = re.sub('\n', ' ', data)
data = re.sub('\W', ' ', data)
data = re.sub('\d', ' ', data)


word_list = list(filter(None, re.split(' ', data)))

for count, str in enumerate(word_list):
    word_list[count] = f'#{str}#'

character_list = []
env_list = []  # the environment list contains all environments that occur in the data including duplicates

for x in word_list:
    List = [x[i:i + 2]
            for i in range(len(x) - 1)]
    env_list.extend(List)

for x in word_list:
    List = [x[i:i + 1]
            for i in range(len(x))]
    character_list.extend(List)
character_list = list(set(character_list))
character_list.remove('#')

env_dict = {}
for b in character_list:
    lst = []
    for a in env_list:
        x = re.search(b, a)
        if x is not None:
            a = re.sub(b, '_', a, 1)
            lst.append(a)
        else:
            continue
        env_dict.update({b: lst})

print(env_dict)

total_env = []  # the environment set is all possible environments given the character list
# total_env = [[f"_{x}", f"_{x}"] for x in character_list]
for a in env_dict:
    total_env = total_env + env_dict[a]
total_env = list(set(total_env))

print(total_env)

count_list = []
for a in env_dict:
    lst = []
    for b in total_env:
        lst.append(env_dict[a].count(b))
    count_list.append(lst)
count_matrix = np.array(count_list)

normal_matrix = ((count_matrix / len(env_list))
                 / (np.matmul(np.expand_dims([sum([row[i] for row in count_matrix / len(env_list)])
                                              for i in range(0, len((count_matrix / len(env_list))[0]))], axis=1),
                              np.expand_dims([sum(row) for row in count_matrix / len(env_list)], axis=0)))
                 .T)

df = pd.DataFrame(normal_matrix, columns = total_env, index = character_list)
print(df)

pca_2 = PCA(n_components=2)
pca_2_result = pca_2.fit_transform(normal_matrix)

pca_df = pd.DataFrame(abs(pca_2.components_), columns=total_env, index=['pca1', 'pca2'])
print(pca_df)

kmeans = KMeans(n_clusters=2)
kmeans.fit(normal_matrix)
centroids = kmeans.cluster_centers_
centroids_pca = pca_2.transform(centroids)

def visualizing_results(pca_result, label, centroids_pca):
    """ Visualizing the clusters
    :param pca_result: PCA applied data
    :param label: K Means labels
    :param centroids_pca: PCA format K Means centroids
    """
    # ------------------ Using Matplotlib for plotting-----------------------
    x = pca_result[:, 0]
    y = pca_result[:, 1]

    plt.scatter(x, y, c=label, alpha=0.5, s=200)  # plot different colors per cluster
    plt.title('Featural classes')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')

    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, linewidths=1.5,
                color='red', edgecolors="black", lw=1.5)

    plt.show()

visualizing_results(pca_2_result, kmeans.labels_, centroids_pca)
