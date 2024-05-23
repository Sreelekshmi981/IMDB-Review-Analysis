import streamlit as st
import pickle
import numpy as np
from PIL import Image

def main():
    st.title("IMDB REVIEW SENTIMENT ANALYSIS")
    image = Image.open("image.jpg")
    st.image(image, width=670)
    text=st.text_input("Review","Type here")
    model=pickle.load(open('model_sv.sav','rb'))
    vectorizer=pickle.load(open('vectorizer.sav','rb'))
    pred=st.button('PREDICT')
    if pred:
        vect = vectorizer.transform([text])
        vect1=vect.toarray()
        prediction=model.predict(vect1)
        if prediction==0:
            st.write("## Negative")
        else:
            st.write("## Postive")

main()
