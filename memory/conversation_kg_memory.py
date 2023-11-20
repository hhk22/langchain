from langchain.memory import ConversationKGMemory
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0.1)

memory = ConversationKGMemory(
    llm=llm,
    return_messages=True
)

def add_message(input, output):
    memory.save_context({"input": input}, {"output": output})

add_message("Hi, I'm Hyeonghwan living in S.Korea", "Wow that is cool!")

rst = memory.load_memory_variables({"input": "who is Hyeonghwan"})
print(rst)

