from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
load_dotenv()

from langchain.prompts import PromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain.prompts.example_selector.base import BaseExampleSelector

from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.schema import BaseOutputParser

chat = ChatOpenAI(
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

examples = [
    {
        "question": "What do you know about France?",
        "answer": """
        Here is what I know:
        Capital: Paris
        Language: French
        Food: Wine and Cheese
        Currency: Euro
        """,
    },
    {
        "question": "What do you know about Italy?",
        "answer": """
        I know this:
        Capital: Rome
        Language: Italian
        Food: Pizza and Pasta
        Currency: Euro
        """,
    },
    {
        "question": "What do you know about Greece?",
        "answer": """
        I know this:
        Capital: Athens
        Language: Greek
        Food: Souvlaki and Feta Cheese
        Currency: Euro
        """,
    },
]

class RandomExampleSelector(BaseExampleSelector):
    def __init__(self, example):
        self.examples = example
    
    def add_example(self, example: Dict[str, str]) -> Any:
        self.examples.append(example)

    def select_examples(self, input_variables):
        from random import choice
        return [choice(self.examples)]
        

example_prompt = PromptTemplate.from_template("Human: {question}\nAI: {answer}")
example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=80
)

example_selector = RandomExampleSelector(
    example=examples
)

prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    example_selector=example_selector,
    suffix="Human: What do you know about {country}",
    input_variables=["country"]
)

chain = prompt | chat
output = chain.invoke({"country":"Brazil"})

print(output)
