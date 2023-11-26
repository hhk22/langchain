from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore
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

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

rst = chain.run("Where does hyeonghwa live?")
print(rst)
rst = chain.run("describe Hyeonghwan")
print(rst)

