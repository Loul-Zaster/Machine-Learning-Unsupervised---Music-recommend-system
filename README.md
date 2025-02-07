# Music Recommendation System
1. Introduction
Music recommendation systems have become an essential feature in modern music streaming platforms, enhancing user experience by suggesting relevant songs based on listening history, preferences, and various content-based features. This project leverages machine learning techniques, particularly deep learning and similarity measures, to develop a content-based music recommendation system. The system incorporates pre-trained models for image feature extraction and similarity computations to recommend songs effectively.
The recommendation engine utilizes Spotify's API to fetch metadata, including album cover images, which are processed using a convolutional neural network (CNN). A precomputed similarity matrix helps determine the most relevant song recommendations based on the selected input.
2. Problem Definition
2.1 Problem Statement
The challenge is to develop an efficient and accurate music recommendation system that can provide personalized song suggestions to users. The system should:
•	Retrieve song metadata and album cover images from Spotify.
•	Extract meaningful features from album cover images using deep learning.
•	Compute song similarity using pre-trained models and cosine similarity.
•	Generate a ranked list of recommended songs based on the selected song input.
2.2 Challenges
•	Efficiently processing album cover images for feature extraction.
•	Implementing a robust similarity measure for content-based filtering.
•	Integrating Spotify's API to retrieve real-time song metadata.
•	Providing a user-friendly interface for easy interaction.
3. Solution Approach
3.1 System Architecture
The system consists of three major components:
1.	Data Collection: Spotify API is used to fetch song metadata, including album cover images.
2.	Feature Extraction: A pre-trained VGG16 model extracts deep features from the album covers.
3.	Recommendation Engine: A similarity matrix, computed using cosine similarity, determines the closest matches to the selected song.
4.	User Interface: Streamlit is used to provide a simple web-based interaction platform for users.
3.2 Technologies Used
•	Python for implementation.
•	Streamlit for web-based user interaction.
•	Spotipy (Spotify API) for fetching song data.
•	Keras & TensorFlow (VGG16) for deep feature extraction.
•	NumPy for numerical computations.
•	Pickle for storing precomputed similarity matrices.
3.3 Recommendation Methodology
•	Content-Based Filtering: Songs are recommended based on the similarity of extracted album cover features.
•	Cosine Similarity: Measures the closeness between feature vectors.
•	Deep Learning (VGG16): Extracts high-level features from images.
4. Code Implementation
4.1 Extracting Features from Album Covers
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import numpy as np
import requests
from io import BytesIO

def extract_image_features(img_url):
    model = VGG16(weights='imagenet', include_top=False)
    response = requests.get(img_url)
    img_data = BytesIO(response.content)
    img = image.load_img(img_data, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()
4.2 Computing Similarity & Recommendations
import pickle

def recommend(song, music, similarity):
    index = music[music['song'] == song].index[0]
    distances = sorted(enumerate(similarity[index]), key=lambda x: x[1], reverse=True)
    recommended_songs = [music.iloc[i[0]].song for i in distances[1:6]]
    return recommended_songs

# Load precomputed data
music = pickle.load(open('df.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Example usage
selected_song = "Shape of You"
recommendations = recommend(selected_song, music, similarity)
print(recommendations)
4.3 Streamlit Web Interface
import streamlit as st

st.header("Music Recommender System")
selected_song = st.selectbox("Choose a song", music['song'].values)

if st.button("Show Recommendations"):
    recommended_songs = recommend(selected_song, music, similarity)
    for song in recommended_songs:
        st.write(song)
Conclusion
This project demonstrates a content-based music recommendation system that integrates deep learning (VGG16) and similarity measures to suggest songs based on album covers. The implementation combines Spotify API, pre-trained CNN models, and cosine similarity to improve recommendation accuracy. Future enhancements can include hybrid approaches incorporating user listening history and collaborative filtering for better personalization.

