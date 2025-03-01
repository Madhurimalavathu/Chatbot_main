import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from datasets import load_dataset

# Custom CSS for styling
st.markdown("""
<style>
    .stApp {
        background: #f8f5e6;
        background-image: radial-gradient(#d4d0c4 1px, transparent 1px);
        background-size: 20px 20px;
    }
    .college-font {
        font-family: 'Times New Roman', serif;
        color: #2c5f2d;
    }
    .user-msg {
        background: #ffffff !important;
        border-radius: 15px !important;
        border: 2px solid #2c5f2d !important;
    }
    .bot-msg {
        background: #fff9e6 !important;
        border-radius: 15px !important;
        border: 2px solid #ffd700 !important;
    }
</style>
""", unsafe_allow_html=True)

# Configure models
genai.configure(api_key="AIzaSyByL4NmHQaeXOC4__zKMiDlaFFh5kZcnvw")
gemini = genai.GenerativeModel('gemini-1.5-flash')
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Embedding model

# Function to load and process WikiQA dataset
@st.cache_data
def load_wikiqa_data():
    try:
        dataset = load_dataset("microsoft/wiki_qa")
        contexts = []

        for example in dataset['train']:
            question = example['question']
            answer = example['answer']
            context = f"Question: {question}\nAnswer: {answer}"
            contexts.append(context)

        embeddings = embedder.encode(contexts)
        index = faiss.IndexFlatL2(embeddings.shape[1])  # FAISS index for similarity search
        index.add(np.array(embeddings).astype('float32'))

        return contexts, index
    except Exception as e:
        st.error(f"Failed to load data. Error: {e}")
        st.stop()

# Load data and FAISS index
contexts, faiss_index = load_wikiqa_data()

# App Header
st.markdown('<h1 class="college-font">🗨️ WikiQA Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="college-font">Your Conversational QA Assistant</h3>', unsafe_allow_html=True)
st.markdown("---")

# Function to find closest matching context using FAISS
def find_closest_context(query, faiss_index, contexts):
    query_embedding = embedder.encode([query])
    _, I = faiss_index.search(query_embedding.astype('float32'), k=3)  # Top 3 matches
    matched_contexts = [contexts[i] for i in I[0]]
    return matched_contexts

# Function to generate a response using Gemini
def generate_response(query, contexts):
    prompt = f"""You are a helpful and knowledgeable chatbot for answering questions based on given contexts.
    Question: {query}
    Contexts: {contexts}
    - Provide a detailed and accurate answer.
    - If the question is unclear, ask for clarification.
    """
    response = gemini.generate_content(prompt)
    return response.text

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], 
                        avatar="🙋" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask anything based on the WikiQA dataset..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Finding the best answer..."):
        try:
            # Find closest matching contexts using FAISS
            matched_contexts = find_closest_context(prompt, faiss_index, contexts)
            
            # Generate a response using Gemini
            response = generate_response(prompt, matched_contexts)
            response = f"**Answer**:\n{response}"
        except Exception as e:
            response = f"Sorry, I couldn't generate a response. Error: {e}"
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
