from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
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

map_doc_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use the following portion of a long document to see if any of the text is relevant to answer the question. Return any relevant text verbatim. If there is no relevant text, return : ''
            -------
            {context}
            """,
        ),
        ("human", "{question}"),
    ]
)


map_doc_chain = map_doc_prompt | llm

def map_document(inputs):
    documents = inputs["documents"]
    question = inputs["question"]
    results = []
    for document in documents:
        result = map_doc_chain.invoke({
            "context": document.page_content,
            "question": question
        }).content
        results.append(result)
    return "\n\n".join(results)

map_chain = { "documents": retriever, "question": RunnablePassthrough() } | RunnableLambda(map_document)

final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use the following portion of a long document to see if any of the text is relevant to answer the question. Return any relevant text verbatim. If there is no relevant text, return : ''
            -------
            {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

chain = {"context": map_chain, "question": RunnablePassthrough()} | final_prompt | llm
rst = chain.invoke("Describe Hyeonghwan")
print(rst)
