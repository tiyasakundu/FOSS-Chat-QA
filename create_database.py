from langchain_community.document_loaders import DirectoryLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from dotenv import load_dotenv
import os
import shutil

DATA_PATH = './data/fossology.wiki'
CHROMA_PATH = 'chroma'

def main():
    load_dotenv()
    create_database()

def create_database():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob='*.md')
    documents=loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 500,
    length_function = len,
    add_start_index = True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks

def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL"),
            model="bge-m3",
        ), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == '__main__':
    main()