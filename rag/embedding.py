from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader, TextLoader
from langchain.storage import LocalFileStore
from dotenv import load_dotenv
load_dotenv()

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

rst = vectorstore.similarity_search("Where do you live?")
print(rst)