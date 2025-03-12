# Import the libraries
from newspaper import Article
import random
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Download the punkt package
nltk.download('punkt', quiet=True)

# Get the Article
article = Article('https://www.mayoclinic.org/diseases-conditions/chronic-kidney-disease/symptoms-causes/syc-20354521')
article.download()
article.parse()
article.nlp()
corpus = article.text

# Tokenization
text = corpus
sentences_list = nltk.sent_tokenize(text)  # A list of sentences

# A function to return a random greeting to a user's greeting
def greeting_response(text):
    text = text.lower()

    # Users Greeting
    user_greetings = ['hi', 'hey', 'hello', 'hola']
    # Bots Greeting Response
    bot_greetings = ['hi', 'hey', 'hello']

    for word in text.split():
        if word in user_greetings:
            return random.choice(bot_greetings)
    return None

# Function to sort indexes based on values in descending order
def index_sort(list_var):
    length = len(list_var)
    list_index = list(range(0, length))

    x = list_var
    for i in range(length):
        for j in range(i + 1, length):
            if x[list_index[i]] < x[list_index[j]]:  # Fix comparison for descending order
                # Swap
                temp = list_index[i]
                list_index[i] = list_index[j]
                list_index[j] = temp

    return list_index  # Move outside of the loop

# Create the bot's response
def bot_response(user_input):
    user_input = user_input.lower()
    sentences_list.append(user_input)
    bot_response = ''
    cm = CountVectorizer().fit_transform(sentences_list)
    similarity_scores = cosine_similarity(cm[-1], cm)
    similarity_scores_list = similarity_scores.flatten()
    index = index_sort(similarity_scores_list)
    index = index[1:]
    response_flag = 0

    j = 0
    for i in range(len(index)):
        if similarity_scores_list[index[i]] > 0.0:
            bot_response += ' ' + sentences_list[index[i]]
            response_flag = 1
            j += 1
            if j > 2:
                break

    if response_flag == 0:
        bot_response = "I'm sorry, I don't understand you."

    sentences_list.pop()  # Ensure cleanup of user input
    return bot_response

# Start the chat
print('Doc Bot: I am Doc Bot. I will answer your questions about Kidney Disease.')

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'bye', 'quit', 'okay bye']:
        print("Doc Bot: Goodbye! Take care.")
        break
    response = greeting_response(user_input)
    if response:
        print(f"Doc Bot: {response}")
    else:
        print(f"Doc Bot: {bot_response(user_input)}")
