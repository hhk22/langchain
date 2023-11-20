from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0.1)

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=120,
    memory_key="chat_history",
    return_messages=True
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a helpful AI talking to a human"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ]
)

def load_memory():
    return memory.load_memory_variables({})["chat_history"]

chain = RunnablePassthrough.assign(chat_history=load_memory) | prompt | llm
rst = chain.invoke({
    "chat_history": memory.load_memory_variables({})["chat_history"],
    "question": "My name is Nico"
})
print(rst)
