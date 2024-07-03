import whisper
import streamlit as st
from pytube import YouTube

import faiss

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.vectorstores.utils import DistanceStrategy


import os
import pandas as pd


# Import the Python SDK
import google.generativeai as genai
# Used to securely store your API key
import os
# Load the API key from Streamlit secrets
api_key = st.secrets["API_KEY "]
genai.configure(api_key=api_key)

class load_data:
  def __init__(self, output_path = 'C:\\Users\\PC\\Documents\\saif\\yt\\podcast'):
    self.output_path = output_path
    self.w_model = whisper.load_model("tiny")

  def download_youtube_audio(self, video_url):
    try:
        # Create a YouTube object
        yt = YouTube(video_url)

        # Get the audio stream with the highest quality
        audio_stream = yt.streams.filter(only_audio=True, file_extension='mp4').first()

        # Download the audio stream
        audio_stream.download(self.output_path)

        print("Download successful!")

    except Exception as e:
        print(f"Error: {e}")

  def transcribe_audio_files(self, folder_path = 'C:\\Users\\PC\\Documents\\saif\\yt\\podcast'):
    """
    Transcribe audio files in a specified folder and return a pandas DataFrame.

    Parameters:
    folder_path (str): The path to the folder containing audio files.
    model: The transcription model to use for transcribing the audio files.

    Returns:
    pd.DataFrame: A DataFrame with two columns: 'filename' and 'transcription_text'.
    """
    # List to store the transcriptions
    transcriptions = []

    # Check if the specified folder path exists
    if os.path.isdir(folder_path):
        # Iterate over all items in the folder
        for item in os.listdir(folder_path):
            audio_path = os.path.join(folder_path, item)
            # Transcribe the audio file using the provided model
            print("WORKING ON ____", item)
            result = self.w_model.transcribe(audio_path)

        return result["text"]

class AIAgent:
  def __init__(self, text, max_length=256, num_retrieved_docs = 3):
    self.max_length = max_length
    self.num_docs = num_retrieved_docs
    self.model = genai.GenerativeModel('gemini-pro')

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([text])

    # Create a vectorstore database using FAISS
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    self.vector_db = FAISS.from_documents(documents=docs, embedding=embeddings)
    #self.vector_db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="chroma_db")
    self.retriever = self.vector_db.as_retriever()

    self.template = "\n\nQuestion:\n{question}\n\nPrompt:\n{prompt}\n\nAnswer:\n{answer}\n\nContext:\n{context}"

  def retrieve(self, query):
      # Retrieve top k similar documents to query
      docs = self.retriever.get_relevant_documents(query)
      return docs

  def create_prompt(self, query, context):
    prompt = f"""
    You are an AI agent focused on answering questions about podcasts. Your task is to provide answers using the given context from the podcast transcript. Do not add any extra information beyond the answer itself.

    Given the following information:

    Question: {query}
    Context: {context}
    Answer:
    """
    return prompt

  def generate(self, query, retrieved_info):
    prompt = self.create_prompt(query, retrieved_info)

    answer = self.model.generate_content(prompt)

    print(prompt)
    return prompt, answer.text

  def query(self, query):
    context = self.retrieve(query)
    data = ""
    for doc in context:
      data += doc.page_content
      data = data[:1000]

    prompt, answer = self.generate(query, data)
    return self.template.format(question = query, prompt = prompt, answer = answer, context = data)
