from os import link
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd

pickle_data_in = open('processed_data.pkl','rb')
ds = pickle.load(pickle_data_in)
pickle_in = open("tfidf.pkl",'rb')
cosine_similarity = pickle.load(pickle_in)


def recomendation(idx):
    
    #get similarity values with other articles
    # similarity_score = list(enumerate(cosine_similarities[idx]))
    show_cos_sim = cosine_similarity[idx]
    print(show_cos_sim)
    similarity_score = sorted(list(enumerate(show_cos_sim)), key=lambda x: x[1], reverse=True)[1:11]
    # Get the scores of the n most similar news articles. Ignore the first movie.

    # similarity_score = similarity_score[1:10+1]

    Title = list()
    Link = list()
    news_indices = [i[0] for i in similarity_score]
    for i in range(len(news_indices)):
        Title.append(ds['title'].iloc[news_indices[i]])
        Link.append(ds['link'].iloc[news_indices[i]])
    return {Title[i]: Link[i] for i in range(len(Link))}
