from langchain_ollama import OllamaLLM

from prompt_templates import get_prev_and_final_prompt
from vector_store import get_vector_store
from retreival import map_and_reduce
import warnings

warnings.filterwarnings("ignore")

llm = OllamaLLM(
    model="llama3.2:1b",
    temperature=0.3,
)
vector_store = get_vector_store()

def ask_llm(user_query: str):

    prev_prompt, final_prompt = get_prev_and_final_prompt()

    answer = map_and_reduce(
        max_tokens=200,
        user_query= user_query,
        k= 4,
        llm = llm,
        vector_store= vector_store,
        prev_prompt= prev_prompt,
        final_prompt= final_prompt,
    )

    return answer

if __name__ == "__main__":
    print(ask_llm(user_query="what is nvidia company good at"))