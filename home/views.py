from django.http import HttpResponse
from django.template import loader
import nltk
from django.shortcuts import render
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')

import re
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle
stop_words = set(stopwords.words('english'))
if os.path.exists(r"C:\Users\jishnu\PycharmProjects\fake_news\model\model.pkl"):
    print("yes")
else:
    print("no")
#with open('../model.pkl', 'rb') as f:
#  clf2 = pickle.load(f)
clf2 = pickle.load(open("../fake_news/model/model.pkl", "rb"))
vectorizer=pickle.load(open("../fake_news/model/vectorizer.pickle", 'rb'))
#vectorizer = TfidfVectorizer()

def LemmSentence(sentence):
  lemma_words = []
  wordnet_lemmatizer = WordNetLemmatizer()
  word_tokens = word_tokenize(sentence)
  for word in word_tokens:
    if word not in stop_words:
      new_word = re.sub('[^a-zA-Z]', '', word)
      new_word = new_word.lower()
      new_word = wordnet_lemmatizer.lemmatize(new_word)
      lemma_words.append(new_word)
  return " ".join(lemma_words)


def output_lable(n):
  if n == 1:
    return "Fake News"
  elif n == 0:
    return "Not A Fake News"


def manual_testing(news):
  testing_news = {"text": [news]}
  new_def_test = pd.DataFrame(testing_news)
  new_def_test["text"] = new_def_test["text"].apply(LemmSentence)
  new_x_test = new_def_test["text"]
  new_xv_test = vectorizer.transform(new_x_test)
  pred_DT = clf2.predict(new_xv_test)
  return output_lable(pred_DT)
def index(request):
  template = loader.get_template('home.html')
  return HttpResponse(template.render())
def check(request):
  out =''
  if request.method == "POST":

    inputtxt = request.POST.get("getrow")
    out=manual_testing(inputtxt)
    print(inputtxt)
    print('output',out)


  return render(request, 'check.html',{'form': out})