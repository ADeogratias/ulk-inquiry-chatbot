
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

# cache the model for use. So load it once and keep it in cache memory
@st.cache(allow_output_mutation=True)

# load model
def loadModel():
  model = load_model('chatbot_model.h5')
  return model

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

# get user input from text_input
def get_text():
    input_text = st.text_input("You: ","")
    return input_text

st.title("Laura Your ULK Admission Chat Bot")

html_temp = """
<div style="background-color:black;padding:10px">
<h5 style="color:white;"> What would you like to know about programs at ULK?</h5>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)
# st.text("What would you like to know about programs at ULK?")
st.text("type your question in the field below and press enter to know get your answer")

with st.spinner("Loading Model Into Memory..."):
  model = loadModel()

message = get_text()
if st.button("Ask"):
  ints = predict_class(message)
  res = get_reponse(ints, intents)
  st.write(message)
  with st.spinner("..."):
    st.success('Sofia: {}'.format(res)+ '\n')

else if message:
  ints = predict_class(message)
  res = get_reponse(ints, intents)
  st.write(message)
  with st.spinner("..."):
    st.success('Sofia: {}'.format(res)+ '\n')
