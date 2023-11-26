from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = UnstructuredFileLoader("./files/chater_one.docx")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)
loader.load_and_split(text_splitter=splitter)



