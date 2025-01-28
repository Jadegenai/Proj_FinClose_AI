from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import create_PDF_embeddings as create_embedding


# Declare Global Variables
openai_api_key = st.secrets["OpenAI_Secret_Key"]
st_UserName = st.secrets["streamlit_username"]
st_Password = st.secrets["streamlit_password"]
llm_model_name = "gpt-4-turbo"


def get_faiss():
    "get the loaded FAISS embeddings"
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
        return FAISS.load_local("kb_faiss_index", embeddings, allow_dangerous_deserialization=True)
    except:
        create_embedding.create_embeddings()
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
        return FAISS.load_local("kb_faiss_index", embeddings, allow_dangerous_deserialization=True)


LETTER_TEMPLATE = """ Your task is to answer the questions releted to Company's Unaccounted Transactions Report that user has asked by taking in consideration \context provided to you.
You take your time to think and provide the correct answer. If the user has asked for total amount then you need to sum the amount for that particular Supplier.
Provide answer based on the \context, and if you can't find anything relevant to the \question asked by the user , just say "I'm sorry, I couldn't find that."
Context: ```{context}```
Question: ```{question}```
"""

LETTER_PROMPT = PromptTemplate(input_variables=["question", "context"], template=LETTER_TEMPLATE, )

llm = ChatOpenAI(
    model_name=llm_model_name,
    temperature=0.3,
    max_tokens=1500,
    openai_api_key=openai_api_key
)


def letter_chain(question):
    """returns a question answer chain for FAISS vectordb"""
    docsearch = get_faiss()
    retreiver = docsearch.as_retriever(  #
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=retreiver,
                                           chain_type="stuff",  # "stuff", "map_reduce","refine", "map_rerank"
                                           return_source_documents=True,
                                           # chain_type_kwargs={"prompt": LETTER_PROMPT}
                                           )
    return qa_chain({"query": question})
