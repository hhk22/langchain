from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
# from langchain.chains import LLMChain
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

# chain = LLMChain(
#     llm=llm,
#     memory=memory,
#     prompt=prompt,
#     verbose=True
# )

def load_memory(_):
    return memory.load_memory_variables({})["chat_history"]

def invoke_chain(question):
    rst = chain.invoke({"question": question})
    memory.save_context({"input": question}, {"output": rst.content})
    print(rst)


chain = RunnablePassthrough.assign(chat_history=load_memory) | prompt | llm
invoke_chain("My name is Nico!")
invoke_chain("What is my name?")

