
import streamlit as st
import pandas as pd
import numpy as np
import random #for choosing random responses
import json #for reading the file random responses
import pickle
from nltk.stem import WordNetLemmatizer #reduce the word to a stem
from tensorflow.keras.models import load_model 

import nltk
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

intents = json.loads(open('intents.json').read())

lemmatizer = WordNetLemmatizer()

# load pickle files
words = pickle.load(open('words.pkl', 'rb')) 
classes = pickle.load(open('classes.pkl', 'rb'))

# load model
model = load_model('chatbot_model.h5')

#function to clean the sentences
def clean_up_sentence(sentence):
  sentence_words = nltk.wordpunct_tokenize(sentence)
  sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
  return sentence_words

#function to get the bag of words
def bag_of_words(sentence):
  sentence_words = clean_up_sentence(sentence)
  bag = [0]* len(words)
  for w in sentence_words:
    for i, word in enumerate(words):
      if word == w:
        bag[i] = 1
  return np.array(bag)

#function predict the class based on the sentence
def predict_class(sentence):
  bow = bag_of_words(sentence)
  res = model.predict(np.array([bow]))[0]
  ERROR_THRESHOLD = 0.25
  results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

  results.sort(key=lambda x: x[1], reverse=True)
  return_list = []
  for r in results:
    return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
  return return_list

#function getting a response the class based on the sentence
def get_reponse(intents_list, intents_json):
  tag = intents_list[0]['intent']
  list_of_intents = intents_json['intents']
  for i in list_of_intents:
    if i['tag'] == tag:
      result =random.choice(i['responses'])
      break
  return result


st.title("Sofia Your ULK Assistant Admissions Chat Bot")
html_temp = """
<div style="background-color:black;padding:10px">
<h3 style="color:white;">Hello, I am Sofia. I will answer your queries about ULK degree program.</h3>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

# result = ""
# categori = ""
# related = []

while True:
  message = st.text_input("Lets Chat, How can I help?", "")
  ints = predict_class(message)
  res = get_reponse(ints, intents)
  if st.button("Send"):
    st.success(' '+ message + '\n')
    st.success('Sofia: {}'.format(res)+ '\n')
    st.markdown(html_temp, unsafe_allow_html=True)
    st.write(message)
    st.write(res)
#     result = predict_news(url)
#     if result == [0]:
#         categori = 'Business'
#         related = df[df['content_category']=='business']["link"]
#     elif result == [1]:
#         categori = 'Entertainment'
#         related = df[df['content_category']=='entertainment']["link"]
#     elif result == [2]:
#         categori = 'Politics'
#         related = df[df['content_category']=='politics']["link"]
#     elif result == [3]:
#         categori = 'Sports'
#         related = df[df['content_category']=='sports']["link"]
# st.success('The article category prediction is: {}'.format(categori))

# front end elements of the web page 
# html_temp = """ 
# <div style ="background-color:black;padding:10px"> 
# <h1 style ="color:white;">Articles that are related</h1> 
# </div> 
# """ 
# st.markdown(html_temp, unsafe_allow_html=True)
# st.write(related)
