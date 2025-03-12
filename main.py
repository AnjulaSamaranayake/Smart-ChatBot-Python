#Import the libraries
from newspaper import Article
import random
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings

warnings.filterwarnings('ignore')

#Download the punkt package
nltk.download('punkt', quiet=True)

#Get the Article
article = Article('https://www.mayoclinic.org/diseases-conditions/chronic-kidney-disease/symptoms-causes/syc-20354521')
article.download()
article.parse()
article.nlp()
corpus = article.text

#print the article text
print(corpus)

#Tokenization
text = corpus
sentences_list = nltk.sent_tokenize(text) # A list of sentences

#Print the list of sentences
print(sentences_list)

#A function to return a random greeting to a users greeting
def greeting_response(text):
    text = text.lower()

    #Users Greeting
    user_greetings = ['hi', 'hey', 'hello', 'hola']
    #Bots greeting response
    bot_greetings = ['hi', 'hey', 'hello']

    for word in text.splot():
        if word in user_greetings:
            return random.choice(bot_greetings)

def index_sort(list_var):
    length = len(list_var)
    list_index = list(range(0, length))

    x = list_var
    for i in range(length):
        for j in range(length):
            if x[list_index[i]] > x[list_index[j]]:
                #Swap
                temp = list_index[i]
                list_index[i] = list_index[j]
                list_index[j] = temp

                return list_index

#Create the bots response
def bot_response(user_input):
    user_input = user_input.lower()
    sentences_list.append(user_input)
    bot_response = ''
    cm = CountVectorizer().fit_transform(sentences_list)
    similarity_scores = cosine_similarity(cm[-1], cm)
    similarity_scores_list = similarity_scores.flatten()
    index = index_sort(similarity_scores_list)