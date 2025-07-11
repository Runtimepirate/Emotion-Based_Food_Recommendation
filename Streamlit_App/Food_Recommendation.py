import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np
import requests

DEEPFACE_BACKEND = 'opencv'

MOOD_TO_KEYWORD = {
    'happy': 'dessert',
    'sad': 'chocolate',
    'angry': 'salad',
    'surprise': 'sushi',
    'neutral': 'sandwich',
    'disgust': 'curry',
    'fear': 'soup'
}

def get_food_recommendation_mealdb(mood):
    keyword = MOOD_TO_KEYWORD.get(mood, "comfort food")
    url = f"https://www.themealdb.com/api/json/v1/1/search.php?s={keyword}"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        if data.get('meals'):
            return data['meals'][0]['strMeal']
        else:
            return "No recommendation found."
    except Exception as e:
        st.error("API error: " + str(e))
        return "No recommendation found."
    
st.set_page_config(page_title="Mood-Based Food Recommender", layout="centered")
st.title("🍽️ Mood-Based Food Recommendation App")
st.markdown("Capture your image below to detect your mood and get personalized food suggestions!")

uploaded_image = st.camera_input("Take a photo")

if uploaded_image is not None:

    image = Image.open(uploaded_image)
    img_np = np.array(image)

    with st.spinner("Analyzing face..."):
        try:
            result = DeepFace.analyze(img_np,
                                      actions=['emotion', 'age', 'gender'],
                                      enforce_detection=False,
                                      detector_backend=DEEPFACE_BACKEND)

            mood = result[0]['dominant_emotion']
            age = result[0]['age']
            gender_raw = result[0]['dominant_gender'] if 'dominant_gender' in result[0] else result[0]['gender']
            gender = "Man" if gender_raw.lower().startswith("m") else "Woman"
            food = get_food_recommendation_mealdb(mood)

            st.success("✅ Analysis complete!")
            st.markdown(f"**Mood:** {mood.capitalize()}")
            st.markdown(f"**Age:** {int(age)}")
            st.markdown(f"**Gender:** {gender}")
            st.markdown(f"**Recommended Food:** 🍽️ **{food}**")

        except Exception as e:
            st.error(f"Error: {e}")