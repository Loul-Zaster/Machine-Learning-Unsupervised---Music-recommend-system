import pickle
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from io import BytesIO
import numpy as np

# Function to get Spotify access token
def get_spotify_token(client_id, client_secret):
    auth_response = requests.post(
        'https://accounts.spotify.com/api/token',
        data={
            'grant_type': 'client_credentials',
            'client_id': client_id,
            'client_secret': client_secret,
        }
    )
    if auth_response.status_code == 200:
        return auth_response.json().get('access_token')
    else:
        return None

CLIENT_ID = ""
CLIENT_SECRET = ""

# Fetch the Spotify access token dynamically
access_token = get_spotify_token(CLIENT_ID, CLIENT_SECRET)
sp = spotipy.Spotify(auth=access_token)

# Function to extract image features using VGG16
def extract_image_features(img_url):
    # Initialize VGG16 model pre-trained on ImageNet without the fully connected layer
    model = VGG16(weights='imagenet', include_top=False)
    
    # Download the image from the song's URL
    response = requests.get(img_url)
    img_data = BytesIO(response.content)
    
    # Load the image and resize it to (224x224) to fit VGG16 requirements
    img = image.load_img(img_data, target_size=(224, 224))
    x = image.img_to_array(img)
    
    # Expand dimensions to match the model input (1, 224, 224, 3)
    x = np.expand_dims(x, axis=0)
    
    # Preprocess the image to normalize for VGG16
    x = preprocess_input(x)
    
    # Extract image features using the VGG16 model
    features = model.predict(x)
    
    return features

# Function to get song album cover URL
def get_song_album_cover_url(song_name, artist_name):
    search_query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=search_query, type="track")

    if results and results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        album_cover_url = track["album"]["images"][0]["url"]
        track_url = track['external_urls']['spotify']
        return album_cover_url, track_url, track['id']
    else:
        return "https://i.postimg.cc/0QNxYz4V/social.png", None, None

# Modify the recommend function to use image features
def recommend(song):
    index = music[music['song'] == song].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_music_names = []
    recommended_music_posters = []
    recommended_music_links = []
    recommended_music_ids = []

    for i in distances[1:6]:
        artist = music.iloc[i[0]].artist
        album_cover_url, track_url, track_id = get_song_album_cover_url(music.iloc[i[0]].song, artist)
        image_features = extract_image_features(album_cover_url)  # Extract image features
        # You can further use these features to improve recommendations
        recommended_music_posters.append(album_cover_url)
        recommended_music_names.append(music.iloc[i[0]].song)
        recommended_music_links.append(track_url)
        recommended_music_ids.append(track_id)

    return recommended_music_names, recommended_music_posters, recommended_music_links, recommended_music_ids

st.header('Music Recommender System')

music = pickle.load(open('df.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

##########################################################
# if st.button('Show Recommendation'):
#     recommended_music_names, recommended_music_posters, recommended_music_links, recommended_music_ids = recommend(selected_song)

#     for i in range(5):
#         st.image(recommended_music_posters[i], width=100)
#         st.text(recommended_music_names[i])
        
#         play_button = st.button(f'Play {recommended_music_names[i]}', key=f'play_{i}')

#         if play_button:
#             st.session_state.track_id = recommended_music_ids[i]


# Input for searching songs
search_term = st.text_input("Song", "")

if search_term:
    filtered_music = music[music['song'].str.contains(search_term, case=False)]
    music_list = filtered_music['song'].values
else:
    music_list = music['song'].values

# Dropdown for selecting songs
selected_song = st.selectbox("Type or select a song from the dropdown", music_list)

if st.button('Show Recommendation'):
    if selected_song:
        # Extract song and artist names
        song_name = selected_song
        artist_name = music[music['song'] == selected_song]['artist'].values[0]

        # Recommendation logic
        recommended_music_names, recommended_music_posters, recommended_music_links, recommended_music_ids = recommend(selected_song)

        # Display recommended music
        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                st.image(recommended_music_posters[i], width=100)
                st.text(recommended_music_names[i])

                # Link to Spotify
                if recommended_music_links[i]:
                    st.markdown(f'<a href="{recommended_music_links[i]}" target="_blank">Open in Spotify</a>', unsafe_allow_html=True)
                else:
                    st.text("No Spotify link available")

                # Get track info and preview
                if recommended_music_ids[i]:
                    try:
                        track_info = sp.track(recommended_music_ids[i])
                        preview_url = track_info.get('preview_url')
                        if preview_url:
                            st.audio(preview_url, format='audio/mp3')
                        else:
                            st.warning("No preview available from Spotify.")
                    except spotipy.exceptions.SpotifyException as e:
                        st.error(f"Error fetching track info: {str(e)}")
                else:
                    st.text("No preview available")
