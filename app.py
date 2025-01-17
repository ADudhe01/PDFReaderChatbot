import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import speech_recognition as sr

import speech_recognition as sr

def get_audio_input():
    """Capture audio input from the microphone and convert it to text."""
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening... Speak now.")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
            st.info("Processing speech...")
            user_text = recognizer.recognize_google(audio)
            st.success(f"Recognized: {user_text}")
            return user_text  # Return recognized text
    except sr.WaitTimeoutError:
        st.error("Listening timed out. Please try again.")
    except sr.RequestError:
        st.error("Could not request results. Check your internet connection.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    return None


def get_pdf_text(pdf_docs):
    text = ""
    
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
            
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    
    chunks = text_splitter.split_text(raw_text)
    
    return chunks
    
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever = vectorstore.as_retriever(),
        memory = memory
    )
    
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Reverse the chat history to display the latest interaction first
    paired_messages = [
        (st.session_state.chat_history[i], st.session_state.chat_history[i + 1])
        for i in range(0, len(st.session_state.chat_history) - 1, 2)
    ]
    for user_msg, bot_msg in reversed(paired_messages):
        st.write(user_template.replace("{{MSG}}", user_msg.content), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", bot_msg.content), unsafe_allow_html=True)

    
    
    
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
        
    if "audio_text" not in st.session_state:
        st.session_state.audio_text = ""
    
    st.header("Chat with multiple PDFs :books:")
    
    # Create columns for the input area
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Text input field
        user_question = st.text_input(
            "Ask a question about your documents:",
            value=st.session_state.audio_text,
            key="user_question"
        )

    with col2:
        # Audio input button
        if st.button("🎤 Speak", key="speak_button"):
            recognized_text = get_audio_input()
            if recognized_text:
                st.session_state.audio_text = recognized_text
                st.rerun()

    # Handle user input
    if st.button("Submit Question"):
        if user_question:
            handle_user_input(user_question)
            st.session_state.audio_text = ""
    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
        )
        
        if st.button("Process"):
            with st.spinner("Processing"):
                # get the pdf text
                raw_text = get_pdf_text(pdf_docs)
            
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
            
                # create vectorstore
                vectorstore = get_vectorstore(text_chunks)
                
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                
        
    

if __name__ == '__main__':
    main()