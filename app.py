import os
import base64
from dotenv import load_dotenv
import streamlit as st
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    ChatPromptTemplate,
    load_index_from_storage,
)
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load environment variables and configure settings
load_dotenv()
Settings.llm = HuggingFaceInferenceAPI(
    model_name="google/gemma-1.1-7b-it",
    tokenizer_name="google/gemma-1.1-7b-it",
    context_window=3000,
    token=os.getenv("HF_TOKEN"),
    max_new_tokens=512,
    generate_kwargs={"temperature": 0.1},
)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Constants
STORAGE_DIR = "./db"
DATA_DIR = "data"
PDF_PATH = os.path.join(DATA_DIR, "uploaded_pdf.pdf")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STORAGE_DIR, exist_ok=True)

def encode_pdf(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def create_pdf_viewer(base64_pdf):
    return f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

def process_pdf():
    docs = SimpleDirectoryReader(DATA_DIR).load_data()
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=STORAGE_DIR)

def query_pdf(user_query):
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    index = load_index_from_storage(storage_context)
    
    qa_template = ChatPromptTemplate.from_messages([
        (
            "user",
            """Your role is to be a Q&A assistant, focused on delivering accurate answers based on the given instructions and context. If a question falls outside the provided context or scope, politely guide the user to ask questions relevant to the document.
            Context:
            {context_str}
            Question:
            {query_str}
            """
        )
    ])
    
    query_engine = index.as_query_engine(text_qa_template=qa_template)
    response = query_engine.query(user_query)
    
    # Handle different response types
    if hasattr(response, 'response'):
        return response.response
    elif isinstance(response, dict) and 'response' in response:
        return response['response']
    elif isinstance(response, str):
        return response
    else:
        return "I apologize, but I couldn't find a relevant answer in the document."

# Streamlit UI
st.title("PDF Insight Engine")
st.markdown("Your Personal Document Assistant")
st.markdown("Let's explore your PDF together!")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [{'role': 'assistant', "content": 'Welcome! Please upload a PDF document, and I\'ll be ready to answer your questions about its content.'}]

with st.sidebar:
    st.title("Document Upload")
    uploaded_file = st.file_uploader("Choose a PDF file to analyze")
    if st.button("Process Document"):
        if uploaded_file:
            with st.spinner("Analyzing the document..."):
                with open(PDF_PATH, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                process_pdf()
                st.success("Analysis complete! You can now ask questions about the document.")
        else:
            st.warning("Please select a PDF file before processing.")

user_input = st.chat_input("What would you like to know about the document?")
if user_input:
    st.session_state.chat_history.append({'role': 'user', "content": user_input})
    bot_response = query_pdf(user_input)
    st.session_state.chat_history.append({'role': 'assistant', "content": bot_response})

for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.write(message['content'])
