from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split(file_path: str) -> List[str] :
    txt_loader = TextLoader(file_path)
    text_content = txt_loader.load()[0].page_content

    splitter = RecursiveCharacterTextSplitter(
    separators= ["\n\n", "\n", "."],
    chunk_size= 100,
    chunk_overlap= False,
    )

    chunks = splitter.split_text(text_content)
    return chunks