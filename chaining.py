from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
load_dotenv()

chat = ChatOpenAI(
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import BaseOutputParser

class CommaOutputParser(BaseOutputParser):
    def parse(self, text):
        return list(map(str.strip, text.split(",")))

template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are list generating machine. Everything you are asked will be answered with a comma list of max 12. DO NOT reply with anything else."),
        ("human", "{question}")
    ]
)

chain = template | chat | CommaOutputParser()
output = chain.invoke({"question": "How many planets?"})
print(output)