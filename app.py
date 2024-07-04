import streamlit as st
from agents import load_data, AIAgent
import os

# Initialize the data loader
data_loader = load_data()

# Streamlit app
st.title("Youtube QA Agent")
st.write("Provide a YouTube link of a podcast, interviews or discussions to get started.")

# Get YouTube link from the user
youtube_link = st.text_input("YouTube Link:")

if st.button("Download and Transcribe"):
    if youtube_link:
        with st.spinner('Downloading and transcribing audio...'):
            try:
                # Download the audio
                data_loader.download_youtube_audio(youtube_link)

                # Transcribe the audio
                transcription_text = data_loader.transcribe_audio_files()
                st.session_state["transcription_text"] = transcription_text

                st.success("Transcription completed!")
                #st.write(transcription_text)

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a valid YouTube link.")

if "transcription_text" in st.session_state:
    st.write("Here are examples of questions you can ask.")
    response = ai_agent.generate_ex()
    st.write(response)
    st.write("You can now ask questions about the podcast.")
    question = st.text_input("Question:")
    if st.button("Ask"):
        if question:
            with st.spinner('Generating answer...'):
                ai_agent = AIAgent(text=st.session_state["transcription_text"])
                response = ai_agent.query(question)
                st.write(response)
        else:
            st.warning("Please enter a question.")

# Save transcription text for later use
if "transcription_text" in st.session_state:
    transcription_text = st.session_state["transcription_text"]
else:
    transcription_text = ""
