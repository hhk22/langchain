from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler, get_openai_callback
from langchain.globals import set_llm_cache, set_debug
from langchain.cache import InMemoryCache
from langchain.llms.loading import load_llm
from dotenv import load_dotenv
load_dotenv()

set_llm_cache(InMemoryCache())
set_debug(True)

# chat = load_llm("model.json")

chat = OpenAI(
    temperature=0.1,
    max_tokens=450,
    model="gpt-3.5-turbo-16k"
)
chat.save("model.json")

# with get_openai_callback() as usage:
#     chat.predict("What is the receipe for ramen ?")
#     print(usage)