import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
import re
nltk.download('stopwords')

data = pd.read_csv('result_final.csv')
print(data.shape)

data = data.drop_duplicates(subset=["title","text"], keep='first', inplace=False)
ds = data[['date','title','text','link']]
ds.insert(0,'id',range(0,data.shape[0]))
ds = ds.dropna()
print(ds.shape)

def remove_punctuation(text):
    Text = text.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    Text = tokenizer.tokenize(Text)
    Stop_word = set(stopwords.words("english"))
    Text = [w for w in Text if not w in Stop_word]
    Texts = [w for w in Text if w.isalpha()]
    Texts = "  ".join(Texts)
    return Texts

def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

ds['cleaned_desc'] = ds['text'].apply(func = remove_punctuation)
ds['cleaned_desc'] = ds.cleaned_desc.apply(func = remove_html)

pickle_data = open('processed_data.pkl','wb')
pickle.dump(ds,pickle_data)
pickle_data.close()

tf = TfidfVectorizer(analyzer='word',stop_words='english',max_df=0.8,min_df=0.0001,use_idf=True,ngram_range=(1,3))
tfidf_matrix = tf.fit_transform(ds['cleaned_desc'])

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

pickle_out = open("tfidf.pkl",'wb')
pickle.dump(cosine_similarities,pickle_out)
pickle_out.close()
