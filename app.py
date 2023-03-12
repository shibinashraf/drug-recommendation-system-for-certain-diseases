import os
import joblib
import pandas as pd
import re
import nltk

nltk.download('wordnet')
nltk.download('stopwords')

import streamlit as st
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup


# Model saved with Keras model.save()
MODEL_PATH = 'model/passmodel.pkl'
TOKENIZER_PATH ='model/tfidfvectorizer.pkl'
DATA_PATH ='data/custom_dataset.csv'

# loading vectorizer
vectorizer = joblib.load(TOKENIZER_PATH)
# loading model
model = joblib.load(MODEL_PATH)
#getting stopwords
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()



st.set_page_config(page_title='PDDRS', page_icon='👨‍⚕️',layout = 'wide')
new_title = '<p style="font-family:sans-serif; color:#D30000; font-size:24px;"><marquee>Warning : For Educational purpose only. Not recommended for actual use.!</marquee></p>'
st.markdown(new_title, unsafe_allow_html=True)


st.markdown("""
<style>
.big-font {
    font-size:25px !important;
    color:#1D8096;
    margin-left:7%;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">• <a href = "https://www.who.int/news-room/fact-sheets/detail/depression">Depression</a></li><ul></p>', unsafe_allow_html=True)

st.title("💉 Patient Diagnosis and Drug Recommendation System 💉")
st.header("The system can detect the following diseases and recommend top drugs.")
lst = ['ADHD', 'Acne', 'Depression','Diabetes, Type 2','Migraine','Pneumonia']
st.markdown('<p class="big-font"> • <a href = "https://www.niams.nih.gov/health-topics/acne">Acne</a></li><ul></p>', unsafe_allow_html=True)
st.markdown('<p class="big-font">• <a href = "https://www.cdc.gov/ncbddd/adhd/facts.html">ADHD</a></li><ul></p>', unsafe_allow_html=True)
st.markdown('<p class="big-font">• <a href = "https://www.who.int/news-room/fact-sheets/detail/depression">Depression</a></li><ul></p>', unsafe_allow_html=True)
st.markdown('<p class="big-font">• <a href = "https://diabetes.org/diabetes/type-2">Diabetes, Type 2</a></li><ul></p>', unsafe_allow_html=True)
st.markdown('<p class="big-font">• <a href = "https://medlineplus.gov/ency/article/000709.htm">Migraine</a></li><ul></p>', unsafe_allow_html=True)
st.markdown('<p class="big-font">• <a href = "nhlbi.nih.gov/health/pneumonia">Pneumonia</a></li><ul></p>', unsafe_allow_html=True)




st.header("Enter Patient Condition:")
raw_text = st.text_input('')


def predict(raw_text):
    global predicted_cond
    global top_drugs
    if raw_text != "":
        clean_text = cleanText(raw_text)
        clean_lst = [clean_text]
        tfidf_vect = vectorizer.transform(clean_lst)
        prediction = model.predict(tfidf_vect)
        predicted_cond = prediction[0]
        df = pd.read_csv(DATA_PATH)
        top_drugs = top_drugs_extractor(predicted_cond,df)

def cleanText(raw_review):
    # 1. Delete HTML 
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. Make a space
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. lower letters
    words = letters_only.lower().split()
    # 5. Stopwords 
    meaningful_words = [w for w in words if not w in stop]
    # 6. lemmitization
    lemmitize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    # 7. space join words
    return( ' '.join(lemmitize_words))


def top_drugs_extractor(condition,df):
    df_top = df[(df['rating']>=9)&(df['usefulCount']>=90)].sort_values(by = ['rating', 'usefulCount'], ascending = [False, False])
    drug_lst = df_top[df_top['condition']==condition]['drugName'].head(4).tolist()
    drug_lst =[*set(drug_lst)]
    return drug_lst


predict_button = st.button("Predict")

if predict_button:
    predict(raw_text)
    st.header('Condition Predicted')
    st.subheader(predicted_cond)
    st.header('Top Recommended Drugs')
    for i in range(0,len(top_drugs)):
        st.subheader(top_drugs[i])


