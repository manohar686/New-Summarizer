from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from loader_and_splitter import load_and_split

encoder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chunks = load_and_split("./data/nvidia_news.txt")

vector_store = Chroma.from_texts( 
    texts=chunks,
    embedding=encoder,
    collection_name="nvidia_news",
    persist_directory="./data/chroma_db",   
)

def get_vector_store():
    return vector_store