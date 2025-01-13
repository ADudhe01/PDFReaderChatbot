# PDF Reader Chatbot

This is a chatbot application built using **Streamlit**, **LangChain**, and **OpenAI**. It allows users to upload multiple PDFs, process the documents, and enables them to ask questions related to the content of the PDFs. The chatbot uses a conversational model to respond based on the text extracted from the uploaded PDFs.

## Features

- **PDF Upload**: Allows users to upload multiple PDFs for processing.
- **Text Extraction**: Extracts text from the uploaded PDFs using `PyPDF2`.
- **Text Chunking**: Splits the extracted text into chunks for better processing and retrieval using `CharacterTextSplitter` from LangChain.
- **Vector Database**: Uses FAISS (Facebook AI Similarity Search) to create a vector store for efficient document search and retrieval.
- **Conversational Bot**: Interacts with users based on the content of the PDFs and maintains conversation history using LangChain's `ConversationalRetrievalChain`.
- **OpenAI Integration**: Integrates OpenAI models (ChatGPT) to provide answers based on the uploaded documents.

## Prerequisites

Before running the app, make sure to install the necessary libraries:

```bash
pip install streamlit python-dotenv PyPDF2 langchain langchain_openai FAISS
```

Additionally, you will need a `.env` file with your OpenAI API Key for using the `ChatOpenAI` class.

## How It Works

- **Upload PDFs**: The user uploads one or more PDF files via the sidebar.
- **Text Extraction**: The app extracts the text content from the PDFs.
- **Text Chunking**: The extracted text is split into smaller chunks for efficient processing.
- **Vectorization**: The text chunks are embedded into vectors using OpenAI embeddings and stored in a FAISS vector store.
- **Conversational Interface**: The user can ask questions related to the document content. The app will respond using the embedded information from the PDFs.

## Running the App

1. Clone the repository to your local machine.
2. Navigate to the project directory and install the dependencies as mentioned above.
3. Run the app using Streamlit:

```bash
streamlit run app.py
```

4. Open your browser and go to `http://localhost:8501` to interact with the chatbot.

## Code Structure

- **app.py**: Main Streamlit app file that contains the core logic of the chatbot.
- **htmlTemplates.py**: Contains the HTML templates for styling the chat messages (e.g., `bot_template`, `user_template`).
- **.env**: Store your OpenAI API key for authentication.

## Usage

1. Upload your PDFs using the sidebar.
2. Once the PDFs are processed, type your questions in the main chat window.
3. The chatbot will provide responses based on the content of the uploaded PDFs.
