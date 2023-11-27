from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI()

cache_dir = LocalFileStore("./.cache")

state_of_the_union = None
with open('./rag/test.txt') as f:
    state_of_the_union = f.read()
texts = state_of_the_union.split("\n\n")

splitter = RecursiveCharacterTextSplitter()
docs = splitter.create_documents(texts)

embeddings = OpenAIEmbeddings()
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    embeddings, cache_dir
)

vectorstore = Chroma.from_documents(docs, cached_embeddings)

retriever = vectorstore.as_retriever()
print(retriever)

prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant. Answer questions using only the following context. If you don't know the answer just say you don't know, don't make it up:\n\n{context}",
        ),
        ("human", "{question}"),
    ])

chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
rst = chain.invoke("Describe Hyeonghwan")
print(rst)


