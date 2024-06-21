from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA
import pinecone
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OpenAI_Secret_Key")
pinecone_key = os.getenv("Pinecone_Secret_Key")
pinecone_env_name = os.getenv("Pinecone_Environment_Name")

LETTER_TEMPLATE = """ Your task is to answer the questions releted to Company's Unaccounted Transactions Report that user has asked by taking in consideration \context provided to you.
You take your time to think and provide the correct answer. If the user has asked for total amount then you need to sum the amount for that particular Supplier.
Provide answer based on the \context, and if you can't find anything relevant to the \question asked by the user , just say "I'm sorry, I couldn't find that."
Context: ```{context}```
Question: ```{question}```
"""

LETTER_PROMPT = PromptTemplate(input_variables=["question", "context"], template=LETTER_TEMPLATE, )

llm = ChatOpenAI(
    model_name="gpt-4-turbo",
    temperature=0.2,
    max_tokens=1000, 
    openai_api_key=api_key
)


def get_pinecone():
    " get the pinecone embeddings"
    pinecone.init(api_key=pinecone_key, environment=pinecone_env_name)
    index_name = "boomi"
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return Pinecone.from_existing_index(index_name,embeddings)


def letter_chain(question):
    """returns a question answer chain for pinecone vectordb"""
    
    docsearch = get_pinecone()
    retreiver = docsearch.as_retriever(#
        search_type="similarity", 
        search_kwargs={"k":3}
    )
    qa_chain = RetrievalQA.from_chain_type(llm,
                                            retriever=retreiver,
                                           chain_type="stuff", #"stuff", "map_reduce","refine", "map_rerank"
                                           return_source_documents=True,
                                           #chain_type_kwargs={"prompt": LETTER_PROMPT}
                                          )
    return qa_chain({"query": question})


