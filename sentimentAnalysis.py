import pandas as pd
import re
from afinn import Afinn
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns

af = Afinn()
sid = SentimentIntensityAnalyzer()

tickers = []
file = open('tickers.txt', 'r')
fileLines = file.readlines()

for line in fileLines:
    line = line.strip()
    tickers.append(line)

# https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
def deEmojify(text):
    regrex_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U0001F1F2-\U0001F1F4"  # Macau flag
        u"\U0001F1E6-\U0001F1FF"  # flags
        u"\U0001F600-\U0001F64F"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U0001F1F2"
        u"\U0001F1F4"
        u"\U0001F620"
        u"\u200d"
        u"\u2640-\u2642"
        "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'',text)

reddit_wsb_df = pd.read_csv("reddit_wsb.csv")
reddit_wsb_df = reddit_wsb_df.dropna()

titles = reddit_wsb_df['title']
titles_list = [title.replace(title, deEmojify(title)) for title in titles]
all_title_tickers = []
for title in titles_list:
    title_tickers = []
    split_title = title.split()
    if set(split_title) & set(tickers):
        title_tickers.append(set(split_title) & set(tickers))
    all_title_tickers.append(title_tickers)

bodies = reddit_wsb_df['body']
bodies_list = [body.replace(body, deEmojify(body)) for body in bodies]
all_body_tickers = []
for body in bodies_list:
    body_tickers = []
    split_body = body.split()
    if set(split_body) & set(tickers):
        body_tickers.append(set(split_body) & set(tickers))
    all_body_tickers.append(body_tickers)

titles_sentiment_scores = [af.score(title) for title in titles_list]
bodies_sentiment_scores = [af.score(body) for body in bodies_list]

titles_polarity_scores = [sid.polarity_scores(title) for title in titles_list]
bodies_polarity_scores = [sid.polarity_scores(body) for body in bodies_list]

titles_polarity_scores_neg = [score['neg'] for score in titles_polarity_scores]
titles_polarity_scores_neu = [score['neu'] for score in titles_polarity_scores]
titles_polarity_scores_pos = [score['pos'] for score in titles_polarity_scores]
titles_polarity_scores_compound = [score['compound'] for score in titles_polarity_scores]

bodies_polarity_scores_neg = [score['neg'] for score in bodies_polarity_scores]
bodies_polarity_scores_neu = [score['neu'] for score in bodies_polarity_scores]
bodies_polarity_scores_pos = [score['pos'] for score in bodies_polarity_scores]
bodies_polarity_scores_compound = [score['compound'] for score in bodies_polarity_scores]

sentiment_df = pd.DataFrame({'Title': titles_list,
                             'Title S&P 500 Tickers': all_title_tickers,
                             'Title Sentiment Score': titles_sentiment_scores,
                             'Title Negative Score': titles_polarity_scores_neg,
                             'Title Neutral Score': titles_polarity_scores_neu,
                             'Title Positive Score': titles_polarity_scores_pos,
                             'Title Compound Score': titles_polarity_scores_compound,
                             'Body': bodies_list,
                             'Body S&P 500 Tickers': all_body_tickers,
                             'Body Sentiment': bodies_sentiment_scores,
                             'Body Negative Score': bodies_polarity_scores_neg,
                             'Body Neutral Score': bodies_polarity_scores_neu,
                             'Body Positive Score': bodies_polarity_scores_pos,
                             'Body Compound Score': bodies_polarity_scores_compound,
                             })

sentiment_df