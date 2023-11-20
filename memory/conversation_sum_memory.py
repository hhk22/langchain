from langchain.memory import ConversationSummaryMemory
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0.1)

memory = ConversationSummaryMemory(llm=llm)

def add_message(input, output):
    memory.save_context({"input": input}, {"output": output})

def get_history():
    return memory.load_memory_variables({})

# add_message("Hi, I'm Hyeonghwan living in S.Korea", "Wow that is cool!")
# add_message("S.Korea is so beautiful", "I wish I could go!!")
# print(get_history())

from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=100,
    return_messages=True
)

def add_message(input, output):
    memory.save_context({"input": input}, {"output": output})

def get_history():
    return memory.load_memory_variables({})

add_message("Hi, I'm Hyeonghwan living in S.Korea", "Wow that is cool!")
add_message("Hi, I'm Hyeonghwan living in S.Korea", "Wow that is cool!")
add_message("Hi, I'm Hyeonghwan living in S.Korea", "Wow that is cool!")
add_message("Hi, I'm Hyeonghwan living in S.Korea", "Wow that is cool!")
add_message("Hi, I'm Hyeonghwan living in S.Korea", "Wow that is cool!")

print(get_history())
