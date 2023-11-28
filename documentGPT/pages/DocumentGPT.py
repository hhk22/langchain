from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

st.title("DocumentGPT")

st.markdown(
    """
        Welcome!
                    
        Use this chatbot to ask questions to an AI about your files!

        Upload your files on the sidebar.
"""
)

def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    
    state_of_the_union = None
    with open('./rag/test.txt') as f:
        state_of_the_union = f.read()
    texts = state_of_the_union.split("\n\n")
    docs = splitter.create_documents(texts)

    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

file = st.file_uploader("Upload a file", type=["pdf", "txt", "docx"])
if file:
    retriver = embed_file(file)
    s = retriver.invoke("where do you live?")
    st.write(s)
