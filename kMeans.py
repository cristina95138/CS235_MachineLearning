import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import random as rand
from sklearn.model_selection import train_test_split


data_names = ['Title S&P 500 + GME Tickers','Title Sentiment Score','Title Negative Score','Title Neutral Score','Title Positive Score','Title Compound Score','Body S&P 500 + GME Tickers','Body Sentiment','Body Negative Score','Body Neutral Score','Body Positive Score','Body Compound Score', 'Timestamp']
data = pd.read_csv('sa.csv',
                   names = data_names)
data_names_gme = ['Date','Open','High','Low','Close','Adj Close','Volume']
data_gme = pd.read_csv('GME.csv',
                   names = data_names_gme)

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

GME_Title = data.loc[data['Title S&P 500 + GME Tickers'].str.contains("GME")]
GME_Body = data.loc[data['Body S&P 500 + GME Tickers'].str.contains("GME")]

GME_Title_Jan = GME_Title.loc[GME_Title['Timestamp'].str.contains("2021-01")]
GME_Title_Feb = GME_Title.loc[GME_Title['Timestamp'].str.contains("2021-02")]
GME_Title_March = GME_Title.loc[GME_Title['Timestamp'].str.contains("2021-03")]
GME_Title_April = GME_Title.loc[GME_Title['Timestamp'].str.contains("2021-04")]
GME_Title_May = GME_Title.loc[GME_Title['Timestamp'].str.contains("2021-05")]

GME_Body_Jan = GME_Body.loc[GME_Body['Timestamp'].str.contains("2021-01")]
GME_Body_Feb = GME_Body.loc[GME_Body['Timestamp'].str.contains("2021-02")]
GME_Body_March = GME_Body.loc[GME_Body['Timestamp'].str.contains("2021-03")]
GME_Body_April = GME_Body.loc[GME_Body['Timestamp'].str.contains("2021-04")]
GME_Body_May = GME_Body.loc[GME_Body['Timestamp'].str.contains("2021-05")]

GME_Jan = data_gme.loc[data_gme['Date'].str.contains("2021-01")]
GME_Jan.apply(pd.to_numeric, errors='ignore')
GME_Feb = data_gme.loc[data_gme['Date'].str.contains("2021-02")]
GME_March = data_gme.loc[data_gme['Date'].str.contains("2021-03")]
GME_April = data_gme.loc[data_gme['Date'].str.contains("2021-04")]
GME_May = data_gme.loc[data_gme['Date'].str.contains("2021-05")]
GME_May = GME_May.apply(pd.to_numeric, errors='ignore')

pslenVals = (GME_May[['Open', 'Close']].values).tolist()
assignments, centroids, all_sse, iters = kmeans_clustering(pslenVals,3,max_iter = 100, tol = pow(10,-3))
clust = []

for cnt,len in enumerate(pslenVals):
    index = assignments[cnt]
    clust.append((len[0], len[1], index))

df = pd.DataFrame(clust, columns=['Open', 'Close', 'cluster'])
plot = sb.scatterplot(x="Open", y="Close", data=df, hue="cluster", palette="Set2")

plt.show()