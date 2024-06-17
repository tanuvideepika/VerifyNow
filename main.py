import pandas as pd 
# import seaborn as sns 
# import matplotlib.pyplot as plt

data = pd.read_csv('News.csv',index_col=0) 
data.head()

data = data.drop(["title", "subject","date"], axis = 1)

# Shuffling 
data = data.sample(frac=1) 
data.reset_index(inplace=True) 
data.drop(["index"], axis=1, inplace=True)

from tqdm import tqdm 
import re 
import nltk 
nltk.download('punkt') 
nltk.download('stopwords') 
from nltk.corpus import stopwords 
# from nltk.tokenize import word_tokenize 
# from nltk.stem.porter import PorterStemmer 
# from wordcloud import WordCloud

def preprocess_text(text_data): 
    preprocessed_text = [] 
    for sentence in tqdm(text_data): 
        sentence = re.sub(r'[^\w\s]', '', sentence) 
        preprocessed_text.append(' '.join(token.lower() 
                                for token in str(sentence).split() 
                                if token not in stopwords.words('english'))) 

    return preprocessed_text


preprocessed_review = preprocess_text(data['text'].values) 
data['text'] = preprocessed_review

# Real 
consolidated = ' '.join( 
    word for word in data['text'][data['class'] == 1].astype(str)) 
 
# Fake 
consolidated = ' '.join( 
    word for word in data['text'][data['class'] == 0].astype(str)) 
 
# from sklearn.feature_extraction.text import CountVectorizer 

from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.linear_model import LogisticRegression 

x_train, x_test, y_train, y_test = train_test_split(data['text'], 
                                                    data['class'], 
                                                    test_size=0.20)

from sklearn.feature_extraction.text import TfidfVectorizer 

vectorization = TfidfVectorizer() 
x_train = vectorization.fit_transform(x_train) 
x_test = vectorization.transform(x_test)

from sklearn.linear_model import LogisticRegression 

model = LogisticRegression() 
model.fit(x_train, y_train) 

# testing the model 
print(accuracy_score(y_train, model.predict(x_train))) 
print(accuracy_score(y_test, model.predict(x_test))) 

from sklearn.tree import DecisionTreeClassifier 

model = DecisionTreeClassifier() 
model.fit(x_train, y_train) 

# testing the model 
print(accuracy_score(y_train, model.predict(x_train))) 
print(accuracy_score(y_test, model.predict(x_test))) 

import pickle

# Save vectorizer
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorization, file)

# Save model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)