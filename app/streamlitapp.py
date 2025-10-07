# Import dependencies
import streamlit as st
import os
import imageio
import tensorflow as tf
import subprocess
from PIL import Image

from utils import load_data, num_to_char
from modelutil import load_model

# Set layout
st.set_page_config(layout='wide')

# Sidebar
with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipReader')
    st.info('This application is originally developed for the Lip Reading deep learning model.')

st.title('Lip Reading App')

# Load available video files
data_path = r"C:\Users\robin\OneDrive\Desktop\LipNet-main\data\data\s1"
if not os.path.exists(data_path):
    st.error(f"Missing folder: {data_path}")
    st.stop()

options = os.listdir(data_path)
if not options:
    st.warning('No videos found in ../data/s1')
    st.stop()

selected_video = st.selectbox('Choose video', options)

# Layout
col1, col2 = st.columns(2)

# Video Preview
with col1:
    st.info('The video below displays the converted video in mp4 format')
    file_path = os.path.join(data_path, selected_video)

    subprocess.run(['ffmpeg', '-i', file_path, '-vcodec', 'libx264', 'test_video.mp4', '-y'])

    with open('test_video.mp4', 'rb') as video_file:
        st.video(video_file.read())

# Model Prediction View
with col2:
    st.info('This is what the deep learning model sees while making a prediction')
    video, annotations = load_data(tf.convert_to_tensor(file_path))
    # imageio.mimsave('animation.gif', video, duration=100)
    st.image(Image.open(r"C:\\Users\\robin\\OneDrive\\Desktop\\LipNet-main\\app\\animation.gif"), width=400)
    
    st.info('This is the output of the machine learning model as tokens')
    model = load_model()
    yhat = model.predict(tf.expand_dims(video, axis=0))

    decoded_sparse = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0]

    decoded_dense = decoded_sparse.numpy()

    st.text(decoded_dense)

    st.info('Decode the raw tokens into words')
    for seq in decoded_dense:
        text = tf.strings.reduce_join(num_to_char(seq)).numpy().decode('utf-8')
        st.text(text)
