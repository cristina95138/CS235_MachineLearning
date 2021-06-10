import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import random as rand
from sklearn.model_selection import train_test_split


data_names = ['Title','Title S&P 500 + GME Tickers','Title Sentiment Score','Title Negative Score','Title Neutral Score','Title Positive Score','Title Compound Score','Body','Body S&P 500 + GME Tickers','Body Sentiment','Body Negative Score','Body Neutral Score','Body Positive Score','Body Compound Score']
data = pd.read_csv('sa.csv',
                   names = data_names)

#k-means clustering
def kmeans_clustering(all_vals,K,max_iter = 100, tol = pow(10,-3)):
    assignments = []
    centroids = rand.sample(all_vals, K)
    all_sse = []
    iters = 0

    for i in range(max_iter):
        ++iters

        assignments = []
        clust = [[] for a in range(K)]
        sse = 0

        for k in all_vals:
            eucDist = -1
            index = 0

            for cnt,l in enumerate(centroids):
                calcDist = np.linalg.norm(np.array(k) - np.array(l))

                if cnt == 0:
                    eucDist = calcDist
                elif calcDist < eucDist:
                    index = cnt
                    eucDist = calcDist

            sse += eucDist
            assignments.append(index)
            clust[index].append(k)

        all_sse.append(sse)

        if i > 0:
            if np.absolute(all_sse[i] - all_sse[i-1])/all_sse[i-1] <= tol:
                break

        for cnt,m in enumerate(clust):
            mean = np.mean(m, axis=0).tolist()
            centroids[cnt] = mean

    return assignments, centroids, all_sse, iters

#GME Sentiment

GME_Neg = data['Title S&P 500 + GME Tickers'].where(data['Title S&P 500 + GME Tickers'] == 'GME')

pslenVals = (data[['Title Sentiment Score', 'petal_length']].values).tolist()
assignments, centroids, all_sse, iters = kmeans_clustering(pslenVals,3,max_iter = 100, tol = pow(10,-3))
clust = []

for cnt,len in enumerate(pslenVals):
    index = assignments[cnt]
    clust.append((len[0], len[1], index))

df = pd.DataFrame(clust, columns=['sepal_length', 'petal_length', 'cluster'])
plot = sb.scatterplot(x="sepal_length", y="petal_length", data=df, hue="cluster", palette="Set2")