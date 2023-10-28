import streamlit as st # Libraries 
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Image Classification", layout="wide")


# Create navigation bar
selected = option_menu(
    menu_title=None,
    options=["Home", "Image Classification Learner", "About"],
    icons=["house", "clipboard-data", "person"],
    default_index=1,
    orientation="horizontal"
)



#Creating the Home Page
def home_page():
        with st.container():
            st.title("Image Classifcation Web App")
        
        st.write("This image classigication learner uses a pre built prediction learner using teachable machine to predict what the image is!")



        st.write("1. Upload an image from your device."
                 )
        st.write("2. Make sure its an image from these 6 classes (Planets, Galaxy, Cosmos Space, Nebula, Stars and Constellations)")
        st.write("3. Make sure it is in JPEG format")
        st.write("4. Watch as the web app calculates what the picture is(Make sure its only related to space)")

#Creating the classification learner page
def class_learner():
            global model
            model = tf.keras.models.load_model("model.h5",compile=False)
            class1 = open("labels.txt","r").readlines()
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            st.header("Astronomy Image Classification Model")
            up_img = st.file_uploader("Upload an image", type=["jpg","jpeg"])
            if up_img is not None:
                  data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                  image = Image.open(up_img).convert("RGB")
                  st.image(up_img,use_column_width=True)
                  size = (224,224)
                  image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
                  image_array = np.asarray(image)
                  image_array1 = (image_array.astype(np.float32)/127.5) - 1
                  data[0] = image_array1
                  prediction = model.predict(data)
                  index = np.argmax(prediction)
                  class_name = class1[index]
                  confidence_score = prediction[0][index]
                  st.write("Predicted Class: ", class_name[2:])
                  st.write(f"Confidence Score:",confidence_score * 100, "%")
            else:
                  st.write("")


#creating the about page
def about():
    st.header("About me")
    st.write("")
    with st.container():
            st.write("My name is Advaith Ramakrishnan")
            st.write("I am a student studying a double degree in software engineering and business informatics in UC.")
            st.write("Contact Details: u3261011@uni.canberra.edu.au ")


if selected == "Home":
      home_page()
elif selected == "Image Classification Learner":
      class_learner() 
if selected == "About":
    about()
