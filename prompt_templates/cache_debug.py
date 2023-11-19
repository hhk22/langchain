from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.globals import set_llm_cache, set_debug
from langchain.cache import InMemoryCache
from dotenv import load_dotenv
load_dotenv()


set_llm_cache(InMemoryCache())
set_debug(True)

chat = ChatOpenAI(
    temperature=0.1
)

chat.predict("How do you make an pasta?")