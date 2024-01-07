import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import speech_recognition as sr

# Download NLTK data (if not already downloaded)
nltk.download('vader_lexicon')

# Instantiate the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to perform audio sentiment analysis
def perform_audio_sentiment_analysis(text):
    sentiment_score = sia.polarity_scores(text)  # Analyze sentiment in transcribed text
    return sentiment_score

def main():
    st.title("Audio Sentiment Analysis")

    recognizer = sr.Recognizer()

    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        st.write("Speak something...")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        audio = recognizer.listen(source)  # Listen to microphone input

    # Recognize speech from the audio
    try:
        st.write("Recognizing...")
        text = recognizer.recognize_google(audio)  # Recognize speech using Google Web Speech API
        st.write(f"You said: {text}")
    except sr.UnknownValueError:
        st.write("Could not understand audio")
    except sr.RequestError:
        st.write("Could not request results; check your internet connection")

    # Perform audio sentiment analysis
    sentiment_score = perform_audio_sentiment_analysis(text)

    if sentiment_score:
        st.write(f"Sentiment Score: {sentiment_score}")
        if sentiment_score['compound'] > 0:
            st.write("Positive sentiment")
        elif sentiment_score['compound'] < 0:
            st.write("Negative sentiment")
        else:
            st.write("Neutral sentiment")

if __name__ == "__main__":
    main()
