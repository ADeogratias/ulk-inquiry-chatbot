
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
    input_text = st.text_input("Laura: ","How can I help?")
    return input_text

st.title("Laura Your ULK Admission Chat Bot")
st.text("What would you like to know about programs at ULK?")
# html_temp = """
# <div style="background-color:black;padding:10px">
# <h3 style="color:white;"> Hello, I am Laura. I will answer your queries about ULK degree program.</h3>
# </div>
# """
# st.markdown(html_temp, unsafe_allow_html=True)

with st.spinner("Loading Model Into Memory..."):
  model = loadModel()

message = get_text()
# if st.button("Send"):
#   ints = predict_class(message)
#   res = get_reponse(ints, intents)
#   st.write(message)
#   with st.spinner("..."):
#     st.success('Sofia: {}'.format(res)+ '\n')

if message:
  ints = predict_class(message)
  res = get_reponse(ints, intents)
  st.write(message)
  with st.spinner("..."):
    st.success('Sofia: {}'.format(res)+ '\n')
    
# while True:
#   message = input("")
#   message = st.text_input("Lets Chat, How can I help?", "")
#   ints = predict_class(message)
#   res = get_reponse(ints, intents)
#   if st.button("Send"):
#     st.success(' '+ message + '\n')
#     st.success('Sofia: {}'.format(res)+ '\n')
#     st.write(message)
#     st.write(res)

# front end elements of the web page 
# html_temp = """ 
# <div style ="background-color:black;padding:10px"> 
# <h1 style ="color:white;">Articles that are related</h1> 
# </div> 
# """ 
# st.markdown(html_temp, unsafe_allow_html=True)
# st.write(related)
