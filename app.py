import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.prompts import PromptTemplate

from groq import Groq

# Load API key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

st.set_page_config(page_title="AI Document Assistant", layout="wide")

# Sidebar
st.sidebar.title("📌 Options")
if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.messages = []

st.sidebar.markdown("Upload a PDF and chat with it!")

# Title
st.title("📄 AI Document Assistant")
st.caption("Chat with your document using RAG + Groq")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:

    # Save file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ PDF uploaded successfully!")

    # Load and process only once
    if "vectorstore" not in st.session_state:
        with st.spinner("Processing document..."):
            loader = PyPDFLoader("temp.pdf")
            documents = loader.load()

            text_splitter = CharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100
            )
            docs = text_splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            vectorstore = FAISS.from_documents(docs, embeddings)
            st.session_state.vectorstore = vectorstore

    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

    # Prompt
    prompt = PromptTemplate.from_template(
        """Answer the question based only on the context below.

Context:
{context}

Question:
{question}
"""
    )

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    user_input = st.chat_input("Ask something about your document...")

    if user_input:
        # Store user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Retrieve docs
                retrieved_docs = retriever.invoke(user_input)
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])

                final_prompt = prompt.format(context=context, question=user_input)

                # Call Groq
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": final_prompt}],
                    model="llama-3.1-8b-instant"
                )

                response = chat_completion.choices[0].message.content

                st.write(response)

        # Store assistant response
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )