import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
from deepface import DeepFace
import random

def recommend_food(mood):
    food_recommendations = {
        'happy': ['Ice Cream', 'Pizza', 'Burger'],
        'sad': ['Chocolate', 'Cake', 'Cookies'],
        'angry': ['Salad', 'Smoothie', 'Soup'],
        'surprise': ['Sushi', 'Pasta', 'Tacos'],
        'neutral': ['Sandwich', 'Fruit Salad', 'Grilled Chicken'],
        'disgust': ['Fried Rice', 'Curry', 'Pancakes'],
        'fear': ['Hot Chocolate', 'Tea', 'Warm Soup']
    }

    if mood in food_recommendations:
        return random.choice(food_recommendations[mood])
    else:
        return "Something light and comforting."

def detect_mood_and_recommend_food():
    camera = cv2.VideoCapture(0)
    
    while True:
        ret, frame = camera.read()

        if not ret:
            print("Failed to capture image")
            break
        
        result = DeepFace.analyze(frame, actions=['emotion','age','gender'], enforce_detection=False)
        
        if result:
            mood = result[0]['dominant_emotion']
            age = result[0]['age']
            gender = result[0]['gender']
            print(f"Detected mood: {mood}")
            print(f"Detected age: {age}")
            print(f"Detected gender: {gender}")

            food_suggestion = recommend_food(mood)
            print(f"Recommended food: {food_suggestion}")

            cv2.putText(frame, f'Mood: {mood}', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f'Age: {age}', (10, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f'Gender: {gender}', (10, 90), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f'Recommended food: {food_suggestion}', (10, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Mood, Age, Gender Detection', frame)
        #qcv2.imshow('Mood',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

    camera.release()
    cv2.destroyAllWindows()

detect_mood_and_recommend_food()
