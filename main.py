import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
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

for count, string in enumerate(word_list):
    word_list[count] = f'#{string}#'

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

df = pd.DataFrame(normal_matrix, columns=total_env, index=character_list)
print(df)

pca_2 = PCA(n_components=2)
pca_2_result = pca_2.fit_transform(normal_matrix)

pca_df = pd.DataFrame(abs(pca_2.components_), columns=total_env, index=['pca1', 'pca2'])
print(pca_df)

def kmean_hyper_param_tuning(data):
    """
    Hyperparameter tuning to select the best from all the parameters on the basis of silhouette_score.
    param data: dimensionality reduced data after applying PCA
    :return: best number of clusters for the model (used for KMeans n_clusters)
    """
    # candidate values for our number of cluster
    parameters = [2, 3, 4, 5, 10]

    # instantiating ParameterGrid, pass number of clusters as input
    parameter_grid = ParameterGrid({'n_clusters': parameters})

    best_score = -1
    kmeans_model = KMeans()     # instantiating KMeans model
    silhouette_scores = []

    # evaluation based on silhouette_score
    for p in parameter_grid:
        kmeans_model.set_params(**p)    # set current hyper parameter
        kmeans_model.fit(data)          # fit model on wine dataset, this will find clusters based on parameter p

        ss = metrics.silhouette_score(data, kmeans_model.labels_)   # calculate silhouette_score
        silhouette_scores += [ss]       # store all the scores

        print('Parameter:', p, 'Score', ss)

        # check p which has the best score
        if ss > best_score:
            best_score = ss
            best_grid = p

    return best_grid['n_clusters']


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
    for i, txt in enumerate(character_list):
        plt.annotate(txt, (x[i], y[i]))

    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, linewidths=0.5,
                color='red', edgecolors="black", lw=0.5)

    plt.show()


optimum_clusters = kmean_hyper_param_tuning(normal_matrix)

kmeans = KMeans(n_clusters=optimum_clusters)
kmeans.fit(normal_matrix)
centroids = kmeans.cluster_centers_
centroids_pca = pca_2.transform(centroids)

visualizing_results(pca_2_result, kmeans.labels_, centroids_pca)
