# 🧠 Mood-Based Food Recommendation System

This project is a real-time mood, age, and gender detection system using the webcam, which recommends food items based on the user's detected emotion. It uses the **DeepFace** library to analyze facial expressions and suggests Indian food items tailored to the detected mood.

## 🎯 Features

- Real-time face detection using webcam
- Emotion, age, and gender detection using DeepFace
- Personalized food recommendations based on mood
- Fun and interactive user experience
- Indian food-centric suggestions

## 🚀 How It Works

1. The webcam captures live video feed.
2. DeepFace analyzes the frame to detect:
   - **Dominant Emotion**
   - **Age**
   - **Gender**
3. A food item is suggested based on the detected emotion.
4. The mood, age, gender, and food recommendation are displayed on the video feed.

## 🧪 Example Moods & Food Recommendations

| Mood      | Food Suggestions                  |
|-----------|-----------------------------------|
| Happy     | 'Ice Cream', 'Pizza', 'Burger'    |
| Sad       | 'Chocolate', 'Cake', 'Cookies'    |
| Angry     | 'Salad', 'Smoothie', 'Soup'       |
| Surprise  | 'Sushi', 'Pasta', 'Tacos'         |
| Neutral   | 'Sandwich', 'Fruit Salad', 'Grilled Chicken'    |
| Disgust   | 'Fried Rice', 'Curry', 'Pancakes'        |
| Fear      | 'Hot Chocolate', 'Tea', 'Warm Soup'   |

## 🛠️ Technologies Used

- Python
- OpenCV
- DeepFace (Facial Attribute Analysis)
- Random (Food selection)
