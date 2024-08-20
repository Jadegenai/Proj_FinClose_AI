from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import streamlit as st
import os

api_key = st.secrets["OpenAI_Secret_Key"]

existing_files_directory = "Flat_Files"
included_extensions = ['pdf']


def create_embeddings():
    st.markdown("Start embedding the Flat files")
    # Load multiple files
    all_files = []
    for fn in os.listdir(existing_files_directory):
        if any(fn.endswith(ext) for ext in included_extensions):
            file_path = os.path.join(existing_files_directory, fn)
            all_files.append(file_path)

    st.markdown("Start chunking the files")
    page_list = []
    for pdf_path in all_files:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        page_list.append(pages)

    flat_list = [item for sublist in page_list for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=5)
    texts = text_splitter.split_documents(flat_list)
    st.markdown("File Chunking Completed")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    docsearch = FAISS.from_documents(texts, embeddings)

    # save the faiss index
    docsearch.save_local("kb_faiss_index")
    st.markdown("All the Flat files have been Embedded")
