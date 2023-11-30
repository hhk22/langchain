from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

prompt = PromptTemplate.from_template("A {word} is a")

llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 50
    }
)

chain = prompt | llm
res = chain.invoke({
    "word": "potato"
})
print(res)