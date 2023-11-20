from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

from langchain.chains import LLMChain

llm = ChatOpenAI(temperature=0.1)

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=180,
    memory_key="chat_history"
)

template = """
    You are a helpful AI talking to a human. 
    {chat_history}
    Human: {question}
    You: 
"""

chain = LLMChain(
    llm=llm,
    memory=memory,
    prompt=PromptTemplate.from_template(template)
)

rst = chain.predict(question="My name is Nico")
print(rst)

rst = chain.predict(question="What is Seoul?")
print(rst)

rst = chain.predict(question="What is my name?")
print(rst)

print(memory.load_memory_variables({}))